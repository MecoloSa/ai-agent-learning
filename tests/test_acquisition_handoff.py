"""获取限制交接：固定输入、真实工具接线，本地模型/检索替身。"""
import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import ai_agent.resources.tools as tools
import ai_agent.runtime.agents as agents


@pytest.mark.parametrize("mode", ["hit", "no_hit", "budget_empty"])
def test_query_scope_and_results_are_facts_not_evidence(tmp_path, monkeypatch, mode):
    evidence = "[source=notes.md, chunk=0]\n叙述视角决定信息范围。"
    queries_seen = []

    class FakeRAG:
        def __init__(self, *args, **kwargs):
            pass

        def retrieve_many(self, pairs, **kwargs):
            queries_seen.extend(pairs)
            return [] if mode == "no_hit" else [Document(page_content="实际命中文本")]

        def expand_neighbors(self, hits, *, window):
            return hits

        def format_documents(self, docs, *, max_context_chars, selection_trace=None):
            return evidence if mode == "hit" else ""

    monkeypatch.setattr(tools, "LearningRAG", FakeRAG)
    blocks, facts = [], []
    tool = tools.build_retrieve_learning_context_tool(
        str(tmp_path), max_queries_per_call=1,
        evidence_blocks=blocks, acquisition_facts=facts,
    )
    result = tool.invoke({"queries": ["叙述视角定义", "人物态度定义"]})
    assert len(queries_seen) == 1 and "本次只执行" in result
    assert blocks == ([evidence] if mode == "hit" else [])
    assert facts == [{
        "tool": "retrieve_learning_context",
        "status": {"hit": "query_limit", "no_hit": "no_hits", "budget_empty": "context_budget"}[mode],
        "requested_queries": 2, "executed_queries": 1,
        "query_limit": 1, "hit_count": 0 if mode == "no_hit" else 1,
        "returned_chars": len(evidence) if mode == "hit" else 0,
    }]
    assert "acquisition_facts" not in tool.args


@pytest.mark.parametrize("partial", [False, True])
def test_read_records_only_actual_truncation(tmp_path, partial):
    (tmp_path / "notes.md").write_text("甲乙丙", encoding="utf-8")
    facts = []
    reader = {t.name: t for t in tools.build_agent_tools(str(tmp_path), acquisition_facts=facts)}["read_learning_file"]
    result = reader.invoke({"relative_path": "notes.md", "max_chars": 1 if partial else 10})
    assert json.loads(result)["data"]["truncated"] is partial
    assert facts == ([{
        "tool": "read_learning_file", "status": "truncated",
        "source": "notes.md", "scope": "prefix", "returned_chars": 1,
    }] if partial else [])
    assert "acquisition_facts" not in reader.args


def test_failed_then_successful_read_reaches_planner_separately(tmp_path, monkeypatch):
    from ai_agent.runtime.context_handoff import unpack_material_context

    (tmp_path / "notes.md").write_text("叙述视角决定信息范围。", encoding="utf-8")
    responses = iter([
        AIMessage(content="", tool_calls=[{"id": "bad", "name": "read_learning_file", "args": {"relative_path": "missing.md"}, "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"id": "good", "name": "read_learning_file", "args": {"relative_path": "notes.md"}, "type": "tool_call"}]),
        AIMessage(content="资料分析完成。[source=notes.md]"),
    ])
    monkeypatch.setattr(agents, "create_llm_with_tools", lambda _: object())
    monkeypatch.setattr(agents, "_invoke_llm_raw", lambda *args, **kwargs: next(responses))
    analysis, context = agents.analyse_materials_agentic(str(tmp_path), "理解叙述视角")
    evidence, facts = unpack_material_context(context)
    assert "missing.md" not in evidence and "notes.md" in evidence
    assert facts == [{"tool": "read_learning_file", "status": "failed", "reason": "FileNotFoundError", "source": "missing.md"}]
    assert str(tmp_path) not in json.dumps(facts)

    requests, trace = [], {}
    monkeypatch.setattr(agents, "needs_rag", lambda _: False)
    monkeypatch.setattr(agents, "create_llm", lambda: object())

    def invoke(llm, messages, *, stage):
        requests.extend(messages)
        return "计划正文"

    monkeypatch.setattr(agents, "_invoke_llm", invoke)
    agents.create_study_plan(analysis, context, "理解叙述视角", 1, material_dir=str(tmp_path), mode="agent", plan_evidence_trace=trace)
    assert trace["initial_evidence"] == evidence
    assert trace["initial_acquisition_facts"] == facts
    user_message = requests[1].content
    assert "初始资料获取范围与限制（程序记录，非原文证据）" in user_message
    assert user_message.index("missing.md") > user_message.index("初始资料获取范围与限制")
    # 保存可读观察，方便直接核对原始输入、证据和真正提供给规划模型的消息。
    (tmp_path / "handoff-observation.json").write_text(json.dumps({
        "execution_mode": "offline_stub",
        "user_input": "理解叙述视角",
        "input_sequence": ["读取missing.md", "读取notes.md", "分析", "规划"],
        "material_text": "叙述视角决定信息范围。",
        "evidence": evidence,
        "acquisition_facts": facts,
        "request_messages": [{"role": m.type, "content": m.content} for m in requests],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def test_handoff_preserves_plain_context_and_roundtrips_restricted_context():
    from ai_agent.runtime.context_handoff import pack_material_context, unpack_material_context

    plain = '[source=notes.md]\n原文\n\n---\n\n{"ok": true}'
    assert pack_material_context(plain, []) == plain
    assert unpack_material_context(plain) == (plain, [])
    facts = [{"tool": "read_learning_file", "status": "truncated", "source": "notes.md"}]
    assert unpack_material_context(pack_material_context(plain, facts)) == (plain, facts)


def test_interactive_preload_separates_facts_and_on_demand_keeps_them_out(monkeypatch):
    import ai_agent.main as main
    from ai_agent.runtime.context_handoff import pack_material_context

    context = pack_material_context("[source=notes.md]\n原文", [{"status": "truncated", "source": "notes.md"}])
    for retrieval, eligible, expected in [(False, False, True), (True, True, False)]:
        from dataclasses import replace
        monkeypatch.setattr(main, "get_settings", lambda: replace(agents.get_settings(), interactive_preload_materials=False))
        prompt = main._build_interactive_system_prompt("分析", "计划", context, "目标", retrieval_available=retrieval, on_demand_eligible=eligible)
        assert ("初始资料获取范围与限制" in prompt) is expected
        assert "ai_agent_material_context" not in prompt


@pytest.mark.parametrize("limited", [False, True])
def test_partial_evidence_budget_records_actual_block_counts(tmp_path, monkeypatch, limited):
    from ai_agent.retrieval.rag import LearningRAG

    docs = [
        Document(page_content="短证据", metadata={"source": "a.md", "chunk_index": 0}),
        Document(page_content="乙" * 500, metadata={"source": "b.md", "chunk_index": 0}),
    ]
    first = LearningRAG.format_documents(None, docs[:1], max_context_chars=1000)

    class FakeRAG:
        format_documents = LearningRAG.format_documents

        def __init__(self, *args, **kwargs):
            pass

        def retrieve_many(self, *args, **kwargs):
            return docs

        def expand_neighbors(self, hits, *, window):
            return hits

    monkeypatch.setattr(tools, "LearningRAG", FakeRAG)
    facts, blocks = [], []
    tool = tools.build_retrieve_learning_context_tool(
        str(tmp_path), max_context_chars=len(first) if limited else 1000,
        acquisition_facts=facts, evidence_blocks=blocks,
    )
    expected = first if limited else LearningRAG.format_documents(None, docs, max_context_chars=1000)
    assert tool.invoke({"queries": ["固定测试主题"]}) == expected
    assert blocks == [expected]
    assert facts == ([{
        "tool": "retrieve_learning_context", "status": "context_budget",
        "requested_queries": 1, "executed_queries": 1,
        "query_limit": tools.get_settings().plan_max_queries_per_tool_call,
        "hit_count": 2, "returned_chars": len(first),
        "input_blocks": 2, "selected_blocks": 1, "omitted_blocks": 1,
    }] if limited else [])


@pytest.mark.parametrize("args,reason", [
    ({"queries": [" "]}, "no_valid_queries"),
    ({"queries": ["定义"], "neighbor_window": 2}, "invalid_neighbor_window"),
])
def test_unexecuted_query_is_not_counted_as_executed(tmp_path, monkeypatch, args, reason):
    def unexpected(*args, **kwargs):
        raise AssertionError("不应创建检索实例")

    monkeypatch.setattr(tools, "LearningRAG", unexpected)
    facts = []
    tool = tools.build_retrieve_learning_context_tool(str(tmp_path), acquisition_facts=facts)
    assert "检索未执行" in tool.invoke(args)
    assert facts == [{"tool": "retrieve_learning_context", "status": "not_executed", "reason": reason, "executed_queries": 0}]


@pytest.mark.parametrize("payload", [
    {"ai_agent_material_context": 2, "evidence": "原文", "acquisition_facts": []},
    {"ai_agent_material_context": 1, "evidence": "原文", "acquisition_facts": "错误字段"},
])
def test_known_but_invalid_handoff_fails_instead_of_becoming_evidence(payload):
    from ai_agent.runtime.context_handoff import unpack_material_context

    with pytest.raises(ValueError, match="资料交接格式"):
        unpack_material_context(json.dumps(payload))
