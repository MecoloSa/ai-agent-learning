"""历史原文回读：固定回复验证来源范围、工具协议及真实交互接线，无网络。"""
import builtins
from dataclasses import replace
import importlib
import json
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError
import pytest

import ai_agent.main as app
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.history_archive import ConversationArchive
from ai_agent.persistence.history_session import ArchiveCapture
from ai_agent.persistence.memory_state import ArchiveSource, HistoryArchiveState, MemorySnapshot
from ai_agent.persistence.state_store import load_learning_state

FIRST = "这轮练习名为松针练习。今天最多20分钟，只阅读与口头说明，不绘图、不编程。这里只讨论，不修改正式记录。"
CORRECTION = "更正：今天最多35分钟，其他条件保持。这里只更正对话条件，不修改任何计划或记录。"
QUESTION = "请实际调用 read_conversation_turn，读取来源1第2轮，逐字引用用户更正时间的完整原话；不要修改计划、任务或反馈。"


@pytest.fixture
def api():
    return importlib.import_module("ai_agent.persistence.history_readback")


@pytest.fixture
def prepared(tmp_path, api):
    root = tmp_path / "archives"
    capture = ArchiveCapture(root, None, True)
    capture.capture(FIRST, "了解，不修改正式记录。")
    capture.capture(CORRECTION, "了解，条件更正为35分钟，不修改正式记录。")
    access = api.HistoryReadAccess(root)
    access.start_turn(capture.state)
    tools = {tool.name: tool for tool in access.tools}
    return root, capture, access, tools


def invoke(tools, name="read_conversation_turn", **args):
    return json.loads(tools[name].invoke(args))


def test_list_returns_only_association_metadata(prepared):
    _, capture, access, tools = prepared
    message = json.loads(access.source_message().content)
    assert message["source_count"] == 1 and CORRECTION not in access.source_message().content
    result = invoke(tools, "list_conversation_sources")["data"]
    assert result["metadata_only"] is True and result["next_offset"] is None
    assert result["sources"] == [{"source_index": 1, **capture.state.sources[0].model_dump()}]
    assert "citation" not in result and CORRECTION not in json.dumps(result, ensure_ascii=False)


def test_read_preserves_exact_text_ids_citation_and_file(prepared):
    root, capture, _, tools = prepared
    ref = capture.state.sources[0]
    path = root / f"{ref.archive_id}.sqlite3"
    before = path.read_bytes()
    data = invoke(tools, source_index=1, turn_number=2)["data"]
    assert data["user_text"] == CORRECTION and data["truncated"] is False
    assert data["evidence_kind"] == "archived_conversation"
    assert data["citation"] == f"[history={ref.archive_id}/{ref.session_id}, turn=2]"
    assert data["citation_ref"] == "[history-ref=1:2]"
    assert data["user_message_id"] == f"{data['turn_id']}:user"
    assert data["assistant_message_id"] == f"{data['turn_id']}:assistant"
    assert path.read_bytes() == before


@pytest.mark.parametrize("args", [
    {"source_index": True, "turn_number": 1},
    {"source_index": "1", "turn_number": 1},
    {"source_index": 0, "turn_number": 1},
    {"source_index": 1, "turn_number": 1.5},
    {"source_index": 1, "turn_number": 0},
    {"source_index": 1, "turn_number": 1, "max_chars": 4001},
    {"source_index": 1, "turn_number": 1, "root": "outside"},
    {"source_index": 1, "turn_number": 1, "archive_id": "f" * 32},
    {"source_index": 1, "turn_number": 1, "session_id": "f" * 32},
])
def test_invalid_arguments_rejected_before_storage(prepared, monkeypatch, args):
    _, _, _, tools = prepared
    monkeypatch.setattr(ConversationArchive, "read_turn", lambda *a, **k: pytest.fail("不能接触数据库"))
    with pytest.raises(ValidationError):
        tools["read_conversation_turn"].invoke(args)


def test_future_turn_and_unknown_source_rejected_before_storage(prepared, monkeypatch):
    _, capture, access, tools = prepared
    ref = capture.state.sources[0].model_copy(update={"through_turn": 1})
    access.start_turn(HistoryArchiveState(sources=(ref,)))
    monkeypatch.setattr(ConversationArchive, "read_turn", lambda *a, **k: pytest.fail("越界不能读取"))
    assert invoke(tools, source_index=1, turn_number=2)["error"]["code"] == "turn_out_of_scope"
    assert invoke(tools, source_index=2, turn_number=1)["error"]["code"] == "source_out_of_scope"


def test_current_turn_sources_are_fixed_until_next_refresh(prepared):
    _, capture, access, tools = prepared
    capture.capture("第三轮原话", "第三轮回复")
    assert invoke(tools, source_index=1, turn_number=3)["error"]["code"] == "turn_out_of_scope"
    access.start_turn(capture.state)
    assert invoke(tools, source_index=1, turn_number=3)["data"]["user_text"] == "第三轮原话"


def test_budget_rejects_without_partial_text(prepared):
    _, _, _, tools = prepared
    result = invoke(tools, source_index=1, turn_number=2, max_chars=1)
    assert result["error"]["code"] == "read_limit_exceeded"
    assert "data" not in result and CORRECTION not in json.dumps(result, ensure_ascii=False)


def test_missing_archive_is_not_recreated(prepared):
    root, capture, _, tools = prepared
    path = root / f"{capture.state.sources[0].archive_id}.sqlite3"
    path.unlink()
    assert invoke(tools, source_index=1, turn_number=1)["error"]["code"] == "archive_unavailable"
    assert not path.exists()


def test_unknown_database_format_does_not_expose_path(prepared):
    root, capture, _, tools = prepared
    path = root / f"{capture.state.sources[0].archive_id}.sqlite3"
    path.write_text("PRIVATE_DATABASE_DETAIL", encoding="utf-8")
    result = invoke(tools, source_index=1, turn_number=1)
    assert result["error"]["code"] == "archive_unavailable"
    assert str(root) not in str(result) and "PRIVATE_DATABASE_DETAIL" not in str(result)


def test_missing_turn_is_failure_not_empty_success(prepared):
    _, capture, access, tools = prepared
    ref = capture.state.sources[0].model_copy(update={"through_turn": 3})
    access.start_turn(HistoryArchiveState(sources=(ref,)))
    assert invoke(tools, source_index=1, turn_number=3)["error"]["code"] == "turn_missing"


def test_catalog_paginates_without_reading_databases(api, tmp_path):
    access = api.HistoryReadAccess(tmp_path)
    refs = tuple(ArchiveSource(archive_id=uuid4().hex, session_id=uuid4().hex, through_turn=1) for _ in range(11))
    access.start_turn(HistoryArchiveState(sources=refs))
    tools = {t.name: t for t in access.tools}
    first = invoke(tools, "list_conversation_sources")["data"]
    second = invoke(tools, "list_conversation_sources", offset=10)["data"]
    assert len(first["sources"]) == 10 and first["next_offset"] == 10
    assert second["sources"][0]["source_index"] == 11 and second["next_offset"] is None
    assert invoke(tools, "list_conversation_sources", offset=12)["error"]["code"] == "invalid_offset"


def test_disabling_capture_keeps_saved_sources_readable(prepared):
    _, capture, access, tools = prepared
    access.start_turn(HistoryArchiveState(enabled=False, sources=capture.state.sources))
    assert invoke(tools, source_index=1, turn_number=2)["data"]["user_text"] == CORRECTION
    access.start_turn(None)
    assert invoke(tools, "list_conversation_sources")["data"]["sources"] == []


def test_real_toolcall_returns_matching_tool_message(prepared):
    _, _, _, tools = prepared
    result = tools["read_conversation_turn"].invoke({
        "type": "tool_call", "id": "history-read-1", "name": "read_conversation_turn",
        "args": {"source_index": 1, "turn_number": 2},
    })
    assert isinstance(result, ToolMessage) and result.tool_call_id == "history-read-1"
    assert json.loads(result.content)["data"]["user_text"] == CORRECTION


def call(name, args, ident):
    return {"name": name, "args": args, "id": ident, "type": "tool_call"}


def test_loop_lists_reads_and_returns_evidence_in_tool_messages(prepared, monkeypatch):
    _, _, access, _ = prepared
    requests = []
    def model(llm, messages, *, stage):
        requests.append(list(messages))
        observed = [m for m in messages if isinstance(m, ToolMessage)]
        if not observed:
            return AIMessage(content="", tool_calls=[call("list_conversation_sources", {}, "catalog")])
        if len(observed) == 1:
            assert json.loads(observed[0].content)["data"]["sources"][0]["through_turn"] == 2
            return AIMessage(content="", tool_calls=[call("read_conversation_turn", {"source_index": 1, "turn_number": 2}, "read")])
        data = json.loads(observed[-1].content)["data"]
        return AIMessage(content=f"> {data['user_text']}\n{data['citation']}")
    monkeypatch.setattr(app, "invoke_llm_response", model)
    audit = {}
    result = app._invoke_interactive_turn(
        request_messages=[HumanMessage(content="摘要只说讨论过安排，不包含原话。"), access.source_message()],
        user_input="请先列出历史来源，再读来源1第2轮并引用原话。不要修改记录。",
        plain_llm=object(), tool_llm=object(), retrieval_tool=None, extra_tools=access.tools, audit_trace=audit,
    )
    assert CORRECTION in result and CORRECTION not in str([m.content for m in requests[0]])
    assert [r["tool_name"] for r in audit["tool_results"]] == ["list_conversation_sources", "read_conversation_turn"]
    assert all(r["executed"] for r in audit["tool_results"])
    assert audit["finished"] and audit["final_answer"] == result


@pytest.mark.parametrize("number,expected", [(2, CORRECTION), (3, "未取得可确认的成功结果")])
def test_explicit_execution_guard_accepts_success_only(prepared, monkeypatch, number, expected):
    _, _, access, _ = prepared
    def model(llm, messages, *, stage):
        observed = [m for m in messages if isinstance(m, ToolMessage)]
        if not observed:
            return AIMessage(content="", tool_calls=[call("read_conversation_turn", {"source_index": 1, "turn_number": number}, "read")])
        result = json.loads(observed[-1].content)
        return AIMessage(content=result["data"]["user_text"] if result["ok"] else "我已读取成功。")
    monkeypatch.setattr(app, "invoke_llm_response", model)
    result = app._invoke_interactive_turn(request_messages=[], user_input=QUESTION,
        plain_llm=object(), tool_llm=object(), retrieval_tool=None, extra_tools=access.tools)
    assert expected in result


def test_existing_global_tool_budget_applies(prepared, monkeypatch):
    _, _, access, _ = prepared
    settings = replace(app.get_settings(), interactive_max_retrieval_calls=2)
    monkeypatch.setattr(app, "get_settings", lambda: settings)
    def model(llm, messages, *, stage):
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="仅按已有成功结果回答。")
        return AIMessage(content="", tool_calls=[
            call("list_conversation_sources", {}, "list"),
            call("read_conversation_turn", {"source_index": 1, "turn_number": 1}, "read1"),
            call("read_conversation_turn", {"source_index": 1, "turn_number": 2}, "read2"),
        ])
    monkeypatch.setattr(app, "invoke_llm_response", model)
    audit = {}
    app._invoke_interactive_turn(request_messages=[], user_input="查看历史", plain_llm=object(),
        tool_llm=object(), retrieval_tool=None, extra_tools=access.tools, audit_trace=audit)
    rows = audit["tool_results"]
    assert [r["executed"] for r in rows] == [True, True, False]
    assert json.loads(rows[-1]["result"])["error"]["code"] == "budget_exceeded"


def test_real_session_binds_refreshes_and_routes_both_tools(api, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "needs_rag", lambda _: False)
    monkeypatch.setattr(app, "create_llm", lambda: object())
    bound = []
    monkeypatch.setattr(app, "create_llm_with_tools", lambda tools: bound.extend(t.name for t in tools) or object())
    catalogs = []
    def turn(**kw):
        tools = {t.name: t for t in kw["extra_tools"]}
        catalogs.append(invoke(tools, "list_conversation_sources")["data"])
        assert "read_conversation_turn" in tools
        if len(catalogs) == 2:
            assert invoke(tools, source_index=1, turn_number=1)["data"]["user_text"] == "第一轮原话"
        return "只解释，不修改记录。"
    monkeypatch.setattr(app, "_invoke_interactive_turn", turn)
    inputs = iter(["第一轮原话", "第二轮查询", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    state = LearningState(learning_goal="原目标", plan_markdown="原计划")
    path = tmp_path / "current.json"
    app.run_interactive_session("", "原计划", "", "原目标", "window", str(tmp_path),
        initial_state=state, current_state_path=path, archive_history=True, history_archive_root=tmp_path / "archives")
    assert [r["total_sources"] for r in catalogs] == [0, 1]
    assert "list_conversation_sources" in bound and "read_conversation_turn" in bound
    saved = load_learning_state(path)
    assert saved.plan_markdown == "原计划" and not saved.tasks and not saved.feedback_records
    assert saved.conversation_memory.history_archive.sources[0].through_turn == 2
