"""分析观察与下游证据分离：真实文件工具，模型与向量查询使用本地替身。"""
import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

import ai_agent.resources.tools as tools
import ai_agent.runtime.agents as agents


def call(name, args, identity):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": identity, "type": "tool_call",
    }])


@pytest.mark.parametrize("read_body", [False, True])
def test_real_analysis_passes_observations_to_model_but_only_reads_downstream(
    tmp_path, monkeypatch, read_body,
):
    body = "叙述视角决定读者能接触的信息。"
    (tmp_path / "notes.md").write_text(body, encoding="utf-8")
    responses = [
        call("list_learning_files", {}, "list"),
        call("search_learning_files", {"keyword": "叙述视角"}, "search"),
    ]
    if read_body:
        responses.append(call("read_learning_file", {"relative_path": "notes.md"}, "read"))
    responses.append(AIMessage(content="分析完成；已读取正文。" if read_body else "只定位了文件，尚未读取正文。"))
    iterator = iter(responses)
    observed = []

    def invoke(llm, messages, *, stage):
        observed[:] = [m for m in messages if isinstance(m, ToolMessage)]
        return next(iterator)

    monkeypatch.setattr(agents, "create_llm_with_tools", lambda _: object())
    monkeypatch.setattr(agents, "_invoke_llm_raw", invoke)
    analysis, context = agents.analyse_materials_agentic(str(tmp_path), "理解叙述视角")
    assert [m.tool_call_id for m in observed] == (["list", "search", "read"] if read_body else ["list", "search"])
    assert "notes.md" in observed[0].content
    assert "notes.md:1:" in observed[1].content
    if read_body:
        assert context == observed[2].content
        data = json.loads(context)["data"]
        assert data["text"] == body
        assert data["source"] == "notes.md"
        assert data["scope"] == "prefix" and data["truncated"] is False
    else:
        assert context == ""

    # 接着调用真实规划入口，检查真正发出的初始证据，而非只检查局部列表。
    monkeypatch.setattr(agents, "needs_rag", lambda _: False)
    monkeypatch.setattr(agents, "create_llm", lambda: object())
    requests = []

    def plan_invoke(llm, messages, *, stage):
        requests.extend(messages)
        return "计划正文：阅读并口头解释叙述视角。"

    monkeypatch.setattr(agents, "_invoke_llm", plan_invoke)
    trace = {}
    agents.create_study_plan(
        analysis, context, "理解叙述视角", 1,
        material_dir=str(tmp_path), mode="agent", plan_evidence_trace=trace,
    )
    assert trace["initial_evidence"] == context
    assert "notes.md:1:" not in requests[1].content
    assert "--- 调用 " not in requests[1].content
    if read_body:
        assert context in requests[1].content


def test_collector_is_scoped_to_each_factory_and_keeps_excerpt_limits(tmp_path):
    (tmp_path / "a.md").write_text("甲乙丙", encoding="utf-8")
    first, second = [], []
    toolsets = [{t.name: t for t in tools.build_agent_tools(str(tmp_path), evidence_blocks=sink)} for sink in (first, second)]
    assert "evidence_blocks" not in toolsets[0]["read_learning_file"].args
    toolsets[0]["read_learning_file"].invoke({"relative_path": "a.md", "max_chars": 1})
    assert len(first) == 1 and second == []
    data = json.loads(first[0])["data"]
    assert data["text"] == "甲" and data["truncated"] is True
    toolsets[1]["list_learning_files"].invoke({})
    toolsets[1]["search_learning_files"].invoke({"keyword": "甲"})
    toolsets[1]["summarize_text_statistics"].invoke({"text": "任意分析输入"})
    assert second == []


def test_empty_or_failed_read_does_not_publish_evidence(tmp_path):
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    sink = []
    reader = {t.name: t for t in tools.build_agent_tools(str(tmp_path), evidence_blocks=sink)}["read_learning_file"]
    reader.invoke({"relative_path": "empty.md"})
    with pytest.raises(FileNotFoundError):
        reader.invoke({"relative_path": "missing.md"})
    assert sink == []


@pytest.mark.parametrize("mode", ["hit", "no_hit", "no_text"])
def test_retrieval_publishes_only_formatted_evidence_not_notices(tmp_path, monkeypatch, mode):
    evidence = "[source=notes.md, chunk=0]\n叙述视角决定信息范围。"

    class FakeRAG:
        def __init__(self, *args, **kwargs):
            pass

        def retrieve_many(self, query_pairs, **kwargs):
            return [] if mode == "no_hit" else [Document(page_content="正文")]

        def expand_neighbors(self, hits, *, window):
            return hits

        def format_documents(self, docs, *, max_context_chars, selection_trace=None):
            return evidence if mode == "hit" else ""

    monkeypatch.setattr(tools, "LearningRAG", FakeRAG)
    sink = []
    tool = tools.build_retrieve_learning_context_tool(str(tmp_path), max_queries_per_call=1, evidence_blocks=sink)
    result = tool.invoke({"queries": ["叙述视角定义", "人物态度定义"]})
    assert "本次只执行" in result
    assert sink == ([evidence] if mode == "hit" else [])
    if mode == "hit":
        assert evidence in result and "本次只执行" not in sink[0]


def test_agent_factory_connects_retrieval_to_the_same_collector(tmp_path, monkeypatch):
    evidence = "[source=notes.md, chunk=0]\n原始资料"

    class FakeRAG:
        def __init__(self, *args, **kwargs):
            pass

        def retrieve_many(self, *args, **kwargs):
            return [Document(page_content="原始资料")]

        def expand_neighbors(self, hits, *, window):
            return hits

        def format_documents(self, docs, *, max_context_chars, selection_trace=None):
            return evidence

    monkeypatch.setattr(tools, "needs_rag", lambda _: True)
    monkeypatch.setattr(tools, "LearningRAG", FakeRAG)
    sink = []
    bound = {t.name: t for t in tools.build_agent_tools(str(tmp_path), evidence_blocks=sink)}
    result = bound["retrieve_learning_context"].invoke({"queries": ["定义"]})
    assert result == evidence and sink == [evidence]
    assert "evidence_blocks" not in bound["retrieve_learning_context"].args
