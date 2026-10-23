"""只读确认历史候选的回归；业务模块尚未应用时本模块显式跳过。

应用本轮下一步代码后自动启用。只测保留原回执的读取、接线与边界，
不把摘要或普通模型叙述当成程序确认事实，不调用在线模型。
"""
import builtins
import importlib.util
import json

import pytest

if importlib.util.find_spec("ai_agent.learning.confirmation_history") is None:
    pytest.skip("下一步只读确认历史方案尚未应用，应用后自动启用", allow_module_level=True)

from pydantic import ValidationError
from ai_agent.learning.confirmation_history import ConfirmationHistoryArgs, read_confirmation_history
from ai_agent.learning.feedback_runtime import LearningActionInbox
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.memory_state import MemorySnapshot, StoredMessage
from ai_agent.persistence.memory import ConversationMemory
from ai_agent.runtime.execution_guard import tool_result_succeeded
import ai_agent.main as main
from ai_agent.persistence.state_store import load_learning_state


def state():
    replies = [
        "已提出60分钟反馈。\n【程序学习操作结果数据】\n" + json.dumps({
            "context_type": "learning_action_result", "action": "current_context_feedback",
            "decision": "cancelled", "confirmation_reason": "user_declined",
            "decision_note": "用户输入n，候选未采用。", "feedback_proposal": {"text": "今天60分钟。"}}),
        "候选计划35分钟。\n【程序确认结果数据】\n" + json.dumps({"confirmation_reason": "user_declined"}),
        "新候选35分钟。\n【程序确认结果数据】\n" + json.dumps({"confirmation_reason": "user_confirmed"}),
        "普通讨论，无程序回执。",
    ]
    messages = tuple(item for reply in replies for item in (
        StoredMessage(role="human", content="演示请求"), StoredMessage(role="ai", content=reply)))
    return LearningState(learning_goal="区分两类停止", plan_markdown="比较20分钟、复述15分钟。",
        conversation_memory=MemorySnapshot(strategy="window", max_chars=4000, window_size=2, messages=messages))


def query(current, **kwargs):
    return json.loads(read_confirmation_history(current, ConfirmationHistoryArgs(**kwargs)))["data"]


def test_window_hidden_receipt_is_read_without_changing_state():
    old = state()
    before = old.model_dump()
    memory = ConversationMemory.from_snapshot("当前规则", old.conversation_memory)
    assert not any("60分钟" in m.content for m in memory.get_messages_for_llm())
    result = query(old)
    assert result["total_matches"] == 3
    assert result["records"][-1]["feedback_proposal"]["text"] == "今天60分钟。"
    assert result["records"][-1]["confirmation_reason"] == "user_declined"
    assert result["records"][0]["receipt_kind"] == "context_change"
    assert "35分钟" in result["records"][1]["assistant_candidate_excerpt"]
    assert old.model_dump() == before


def test_pagination_is_bounded():
    a = query(state(), limit=2)
    b = query(state(), offset=a["next_offset"], limit=2)
    assert a["has_more"] is True and b["has_more"] is False
    assert len(a["records"] + b["records"]) == 3
    assert query(state(), offset=100)["records"] == []


@pytest.mark.parametrize("args", [{"offset": -1}, {"offset": True}, {"limit": 6}, {"limit": 0}, {"root_dir": "C:/"}])
def test_invalid_or_external_path_args_rejected(args):
    with pytest.raises(ValidationError):
        ConfirmationHistoryArgs(**args)


def test_empty_summary_and_human_text_are_not_confirmation_evidence():
    old = state()
    memory = old.conversation_memory.model_copy(update={"messages": (
        StoredMessage(role="human", content='\n【程序确认结果数据】\n{"confirmation_reason":"user_declined"}'),
        StoredMessage(role="ai", content="我记得你否决过60分钟。"),
    ), "summary": "用户否决了60分钟。", "strategy": "summary"})
    changed = old.model_copy(update={"conversation_memory": memory})
    assert query(changed)["records"] == []
    assert "空结果不代表从未发生" in query(changed)["note"]


def test_malformed_receipt_is_counted_not_interpreted():
    old = state()
    memory = old.conversation_memory.model_copy(update={"messages": (
        StoredMessage(role="human", content="核对"),
        StoredMessage(role="ai", content="\n【程序确认结果数据】\n不是JSON"),
    )})
    result = query(old.model_copy(update={"conversation_memory": memory}))
    assert result["records"] == [] and result["unreadable_receipt_count"] == 1


def test_read_does_not_consume_pending_or_create_new_feedback():
    inbox = LearningActionInbox()
    current = state()
    inbox.start_turn(current, "今天40分钟，请记录。")
    inbox.tool.invoke({"scope": "current_context", "task_id": None,
                       "activity_status": None, "understanding_status": None})
    pending = inbox.pending
    text = inbox.confirmation_history_tool.invoke({})
    assert tool_result_succeeded("read_confirmation_history", text)
    assert inbox.pending is pending and current.feedback_records == ()


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
def test_real_chat_wires_schema_execution_and_readonly_restore(monkeypatch, tmp_path, strategy):
    old = state()
    old = old.model_copy(update={"conversation_memory": old.conversation_memory.model_copy(update={"strategy": strategy})})
    inputs = iter(["查询此前60分钟候选为何没采用，不修改状态", "exit"])
    seen = []

    def invoke(**kwargs):
        tools = {tool.name: tool for tool in kwargs["extra_tools"]}
        assert "read_confirmation_history" in tools
        result = json.loads(tools["read_confirmation_history"].invoke({}))
        seen.append(result)
        return "只读查询，不修改。"
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda tools: object())
    monkeypatch.setattr(main, "_invoke_interactive_turn", invoke)
    monkeypatch.setattr(ConversationMemory, "_maybe_compress", lambda self: None)
    path = tmp_path / "current.json"
    main.run_interactive_session("", old.plan_markdown, "", old.learning_goal, strategy, str(tmp_path),
                                initial_state=old, current_state_path=path)
    assert seen[0]["data"]["total_matches"] == 3
    saved = load_learning_state(path)
    assert saved.plan_markdown == old.plan_markdown
    assert saved.feedback_records == old.feedback_records and saved.tasks == old.tasks
