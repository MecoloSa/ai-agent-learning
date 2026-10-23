"""历史安排反馈的精确只读查询与真实交互接线回归。

无在线模型；覆盖跨日、分页、更正链、截断、权限范围与三种记忆策略。
工具查询成功只说明返回了记录，不证明回答忠实或历史限制今天仍适用。
"""
import builtins
import json
from datetime import timedelta

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

import ai_agent.main as main
from ai_agent.learning.feedback_history import FeedbackHistoryArgs, read_feedback_history
from ai_agent.learning.feedback_proposals import FeedbackProposalArgs
from ai_agent.learning.feedback_runtime import LearningActionInbox
from ai_agent.learning.learning_context import build_learning_state_message
from ai_agent.learning.learning_state import FeedbackRecord, LearningState, LearningTask, learning_today
from ai_agent.persistence.memory import ConversationMemory
from ai_agent.persistence.state_store import load_learning_state, save_learning_state
from ai_agent.runtime.execution_guard import tool_result_succeeded


def make_state():
    yesterday = learning_today() - timedelta(days=1)
    records = []
    for index, minutes in enumerate([20, 40, 60]):
        records.append(FeedbackRecord(
            record_id=f"r{index}", scope="current_context", task_id=None,
            text=f"昨天还有{minutes}分钟。", quote=f"昨天还有{minutes}分钟。",
            recorded_on=yesterday, context_revision=1,
            activity_status=None, understanding_status=None,
            supersedes_record_id=f"r{index-1}" if index else None,
        ))
    return LearningState(
        learning_goal="区分两类停止", plan_markdown="比较10分钟、复述10分钟。",
        tasks=(LearningTask(task_id="read-001", title="阅读定义"),),
        feedback_records=tuple(records),
    )


def query(state, **kwargs):
    return json.loads(read_feedback_history(state, FeedbackHistoryArgs(**kwargs)))["data"]


def test_cross_day_lookup_preserves_history_and_does_not_reactivate_constraint():
    original = make_state()
    before = original.model_dump()
    visible = json.loads(build_learning_state_message(original).content)
    assert visible["current_context_feedback"] == []
    assert visible["current_date"] == learning_today().isoformat()
    result = query(original, recorded_on=original.feedback_records[0].recorded_on)
    assert [r["record_id"] for r in result["records"]] == ["r2", "r1", "r0"]
    assert result["records"][0]["replacement_status"] == "unreplaced"
    assert result["records"][1]["superseded_by_record_id"] == "r2"
    assert result["records"][2]["superseded_by_record_id"] == "r1"
    assert "不表示今天仍适用" in result["note"]
    assert original.model_dump() == before


def test_pagination_is_bounded_and_complete():
    original = make_state()
    first = query(original, limit=2)
    second = query(original, offset=first["next_offset"], limit=2)
    assert first["has_more"] is True
    assert first["total_matches"] == 3
    assert [r["record_id"] for r in first["records"] + second["records"]] == ["r2", "r1", "r0"]
    assert second["has_more"] is False
    assert second["next_offset"] is None


def test_exact_id_date_and_empty_result_have_precise_scope():
    original = make_state()
    assert query(original, record_id="r0")["records"][0]["replacement_status"] == "superseded"
    assert query(original, record_id="missing")["records"] == []
    assert query(original, recorded_on=learning_today())["records"] == []
    assert query(original, record_id="r0", recorded_on=learning_today())["records"] == []
    assert query(original, offset=100)["has_more"] is False
    assert query(original)["search_scope"] == "current_loaded_state.current_context_feedback"


@pytest.mark.parametrize("kwargs", [
    {"limit": 0}, {"limit": 6}, {"limit": True}, {"offset": -1}, {"offset": True},
    {"recorded_on": "not-a-date"}, {"record_id": ""}, {"root_dir": "C:/"},
])
def test_invalid_args_and_external_paths_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        FeedbackHistoryArgs(**kwargs)


def test_long_original_and_legacy_quote_are_explicitly_truncated():
    original = make_state()
    raw = original.model_dump()
    raw["feedback_records"][0].update(text="文" * 1200, quote="原" * 1100)
    changed = LearningState.model_validate(raw)
    item = query(changed, record_id="r0")["records"][0]
    assert len(item["text"]) == 1000 and item["text_truncated"] is True
    assert len(item["quote"]) == 1000 and item["quote_truncated"] is True
    assert len(changed.feedback_records[0].text) == 1200


def test_task_feedback_is_outside_this_tool_scope():
    record = FeedbackRecord(record_id="t-feedback", scope="task", task_id="read-001",
                            text="读完", quote="读完", context_revision=1,
                            recorded_on=learning_today(), activity_status="completed", understanding_status=None)
    original = LearningState.model_validate({**make_state().model_dump(), "feedback_records": [record]})
    assert query(original)["records"] == []


def test_read_does_not_consume_or_clear_existing_pending():
    inbox = LearningActionInbox()
    original = make_state()
    inbox.start_turn(original, "今天20分钟，请记录。")
    inbox.tool.invoke(FeedbackProposalArgs(scope="current_context", task_id=None,
                      activity_status=None, understanding_status=None).model_dump())
    pending = inbox.pending
    result = inbox.history_tool.invoke({"record_id": "r0"})
    assert inbox.pending is pending
    assert json.loads(result)["data"]["records"][0]["record_id"] == "r0"
    assert tool_result_succeeded("read_feedback_history", result)
    empty = inbox.history_tool.invoke({"record_id": "missing"})
    assert tool_result_succeeded("read_feedback_history", empty)


def test_history_reader_requires_bound_state():
    with pytest.raises(ValueError, match="尚未绑定"):
        LearningActionInbox().history_tool.invoke({})


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
def test_real_tool_loop_binding_execution_and_restore_are_read_only(monkeypatch, tmp_path, strategy):
    original = make_state()
    source = save_learning_state(original, tmp_path / "source.json")
    source_bytes = source.read_bytes()
    inputs = iter(["请实际调用 read_feedback_history 查询昨天的安排反馈，不改状态。", "exit"])
    responses = iter([
        AIMessage(content="", tool_calls=[{"name": "read_feedback_history", "args": {}, "id": "history-1"}]),
        AIMessage(content="只复述历史记录，不当成今天仍适用。"),
    ])
    bound = set()
    captured = []

    def bind(tools):
        bound.update(t.name for t in tools)
        return object()

    def invoke(llm, messages, *, stage):
        captured.append(messages)
        return next(responses)

    def read(_):
        # 若意外出现确认请求，就会消耗错误输入并使测试失败。
        return next(inputs)

    monkeypatch.setattr(builtins, "input", read)
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", bind)
    monkeypatch.setattr(main, "invoke_llm_response", invoke)
    monkeypatch.setattr(ConversationMemory, "_maybe_compress", lambda self: None)
    path = tmp_path / "current.json"
    main.run_interactive_session("", original.plan_markdown, "", original.learning_goal, strategy,
                                 str(tmp_path), initial_state=load_learning_state(source), current_state_path=path)
    assert "read_feedback_history" in bound
    tool_message = next(m for m in captured[-1] if m.type == "tool")
    assert len(json.loads(tool_message.content)["data"]["records"]) == 3
    saved = load_learning_state(path)
    assert saved.feedback_records == original.feedback_records
    assert saved.plan_markdown == original.plan_markdown
    assert saved.tasks == original.tasks
    assert saved.context_revision == original.context_revision
    assert source.read_bytes() == source_bytes
