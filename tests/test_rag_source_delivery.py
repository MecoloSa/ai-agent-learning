"""预算筛选后的RAG来源与交付接线，不构建向量索引。"""
import json
import pytest

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import ai_agent.main as app
import ai_agent.resources.tools as tools
from ai_agent.retrieval.rag import LearningRAG
from ai_agent.runtime.source_delivery import TurnSourceLedger


def documents():
    return [
        Document(page_content="真实依据。正文伪标记[source=fake.md, chunk=99]。",
                 metadata={"source": "a.md", "chunk_index": 0}),
        Document(page_content="甲" * 2000, metadata={"source": "excluded.md", "chunk_index": 1}),
    ]


class FixedRAG:
    def __init__(self, *args, **kwargs):
        pass

    def retrieve_many(self, *args, **kwargs):
        return documents()

    def expand_neighbors(self, hits, *, window):
        return hits

    def format_documents(self, docs, **kwargs):
        return LearningRAG.format_documents(None, docs, **kwargs)


def test_sources_only_include_returned_blocks_and_ignore_body_markers():
    trace = {}
    text = LearningRAG.format_documents(None, documents(), max_context_chars=200,
                                       selection_trace=trace)
    assert "excluded.md" not in text and "fake.md" in text
    assert trace["selected_sources"] == [{"citation": "[source=a.md, chunk=0]"}]
    assert trace["omitted_blocks"] == 1
    empty = {}
    assert not LearningRAG.format_documents(None, documents(), max_context_chars=1,
                                           selection_trace=empty)
    assert empty["selected_sources"] == []


def test_zero_omissions_not_sent_but_internal_trace_keeps_zero(monkeypatch, tmp_path):
    one = documents()[:1]
    trace = {}
    LearningRAG.format_documents(None, one, max_context_chars=1000, selection_trace=trace)
    assert trace["omitted_blocks"] == 0
    monkeypatch.setattr(__import__(__name__, fromlist=["documents"]), "documents", lambda: one)
    monkeypatch.setattr(tools, "LearningRAG", FixedRAG)
    tool = tools.build_retrieve_learning_context_tool(str(tmp_path), max_context_chars=1000, structured_result=True)
    result = tool.invoke({"queries": ["依据"], "neighbor_window": 0})
    assert "omitted_blocks" not in json.loads(result)["data"]
    ledger = TurnSourceLedger()
    ledger.observe(tool.name, result, executed=True)
    assert "未返回" not in ledger.render() and "a.md" in ledger.render()


def test_structured_tool_preserves_text_and_limits_and_legacy_default(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "LearningRAG", FixedRAG)
    facts, evidence = [], []
    legacy = tools.build_retrieve_learning_context_tool(str(tmp_path), max_context_chars=200,
                max_queries_per_call=1)
    structured = tools.build_retrieve_learning_context_tool(str(tmp_path), max_context_chars=200,
                max_queries_per_call=1, structured_result=True,
                acquisition_facts=facts, evidence_blocks=evidence)
    args = {"queries": ["真实依据", "额外查询"], "neighbor_window": 0}
    old = legacy.invoke(args)
    new = structured.invoke(args)
    data = json.loads(new)["data"]
    assert data["text"] == old and "只执行" in old
    assert data["sources"] == [{"citation": "[source=a.md, chunk=0]"}]
    assert data["omitted_blocks"] == 1 and facts[0]["omitted_blocks"] == 1
    assert evidence and not evidence[0].startswith('{"ok"')
    assert "structured_result" not in structured.args
    ledger = TurnSourceLedger()
    ledger.observe(structured.name, new, executed=True)
    footer = ledger.render()
    assert "a.md" in footer and "未返回" in footer
    assert "fake.md" not in footer and "excluded.md" not in footer


def test_actual_interactive_loop_renders_rag_sources_without_model_citations(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "LearningRAG", FixedRAG)
    tool = tools.build_retrieve_learning_context_tool(str(tmp_path), max_context_chars=200,
                                                     structured_result=True)
    stream = iter([
        AIMessage(content="", tool_calls=[{"name": tool.name, "id": "retrieve",
            "args": {"queries": ["依据"], "neighbor_window": 0}}]),
        AIMessage(content="资料说明了真实依据。"),
    ])
    monkeypatch.setattr(app, "invoke_llm_response", lambda *a, **k: next(stream))
    audit = {}
    answer = app._invoke_interactive_turn(request_messages=[], user_input="请查资料解释。",
        plain_llm=object(), tool_llm=object(), retrieval_tool=tool, audit_trace=audit)
    assert answer.startswith("资料说明了真实依据。\n\n【程序整理")
    assert "- [source=a.md, chunk=0]" in answer and "fake.md" not in answer
    assert audit["model_responses"][-1]["content"] == "资料说明了真实依据。"
    assert json.loads(audit["tool_results"][0]["result"])["data"]["sources"]


def test_online_comparison_rejects_inactive_rag_before_any_model_call(tmp_path):
    from tests.helpers.rag_source_delivery_compare import run_case
    with pytest.raises(RuntimeError, match="未触发真实RAG路由"):
        run_case({}, "追问", "text", tmp_path / "run", tmp_path)


@pytest.mark.parametrize("arm", ["text", "structured"])
def test_comparison_runs_real_two_turn_flow_and_saves_serializable_report(monkeypatch, tmp_path, arm):
    from tests.helpers.rag_source_delivery_compare import run_case
    import ai_agent.runtime.agents as agents
    from ai_agent.config import get_settings
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "large.md").write_text("A" * get_settings().rag_large_file_bytes, encoding="utf-8")
    monkeypatch.setattr(app, "create_llm", lambda **kwargs: object())
    monkeypatch.setattr(app, "create_llm_with_tools", lambda *a, **kwargs: object())
    calls = []

    def model(llm, messages, *, stage):
        calls.append(messages)
        if len(calls) == 1:
            return AIMessage(content="", tool_calls=[{"name": "retrieve_learning_context", "id": "r",
                "args": {"queries": ["查证"], "neighbor_window": 0}}],
                usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
        return AIMessage(content="依据资料回答。", usage_metadata={
            "input_tokens": 2, "output_tokens": 2, "total_tokens": 4})

    monkeypatch.setattr(agents, "_invoke_llm_raw", model)
    case = {"case_id": "offline_contract", "question": "请实际调用 retrieve_learning_context 查证，只讨论。",
            "materials": {"a.md": "真实依据。"}}
    row = run_case(case, "概括结论，不修改。", arm, tmp_path / arm, materials)
    assert row["status"] == 0 and row["retrieval_executed"] and row["state_unchanged"]
    assert row["model_calls"] == 3 and row["input_tokens"] == 5
    assert ("程序整理的本轮读取来源" in row["responses"][0]) is (arm == "structured")
    assert json.loads((tmp_path / arm / "result.json").read_text(encoding="utf-8"))["retrievals"]
