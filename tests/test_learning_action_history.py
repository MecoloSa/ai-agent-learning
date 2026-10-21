"""学习操作结果回归：确认结论应关联操作、任务和最新状态。

本文件配合本轮指导应用后运行；假模型验证数据流，不证明在线语义正确。
"""

import builtins
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import ai_agent.main as main
import ai_agent.learning.feedback_runtime as runtime
import ai_agent.persistence.memory as memory_module
from ai_agent.learning.feedback_proposals import FeedbackProposalArgs, prepare_feedback_proposal
from ai_agent.learning.learning_state import LearningState, LearningTask
from ai_agent.learning.task_creation import TaskCreationArgs, prepare_task_creation, commit_task_creation
from ai_agent.persistence.state_store import load_learning_state


def state():
    return LearningState(
        learning_goal="区分两类停止", plan_markdown="原计划不变",
        tasks=(LearningTask(task_id="read-001", title="阅读定义", activity_status="completed",
                            understanding_status="self_reported_unclear"),),
    )


def receipt(text):
    return json.loads(text.split("\n【程序学习操作结果数据】\n", 1)[1])


@pytest.mark.parametrize("decision", ["confirmed", "cancelled", "invalid"])
def test_creation_receipt_uses_actual_state_not_proposal(decision):
    original = state()
    pending = prepare_task_creation(original, TaskCreationArgs(title="对比两类停止"), user_message="加入待办")
    current = commit_task_creation(original, pending) if decision == "confirmed" else original
    result = receipt(runtime.build_learning_action_history("候选尚未生效", current, pending, decision))
    assert result["action"] == "task_creation"
    assert result["decision"] == decision
    assert result["task_id"] == pending.task_id
    assert result["proposal_pending"] is False
    assert result["persistence"] == "not_checked_here"
    if decision == "confirmed":
        assert result["task_after"]["activity_status"] == "not_started"
    else:
        assert result["task_after"] is None
    assert current.plan_markdown == original.plan_markdown


@pytest.mark.parametrize("scope", ["task", "current_context"])
def test_feedback_receipt_distinguishes_task_and_context(scope):
    original = state()
    args = FeedbackProposalArgs(
        scope=scope, task_id="read-001" if scope == "task" else None,
        activity_status=None,
        understanding_status="self_reported_clear" if scope == "task" else None,
    )
    pending = prepare_feedback_proposal(original, args, user_message="现在清楚了" if scope == "task" else "今天40分钟")
    updated = runtime.commit_feedback_proposal(original, pending)
    result = receipt(runtime.build_learning_action_history("提议", updated, pending, "confirmed"))
    assert result["action"] == ("task_feedback" if scope == "task" else "current_context_feedback")
    if scope == "task":
        assert result["task_after"]["activity_status"] == "completed"
        assert result["task_after"]["understanding_status"] == "self_reported_clear"
    else:
        assert result["task_after"] is None
        assert updated.tasks == original.tasks


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
@pytest.mark.parametrize("answer", ["y", "n"])
def test_confirmed_result_reaches_next_turn_and_saved_memory(monkeypatch, tmp_path, strategy, answer):
    original = state()
    path = tmp_path / "current.json"
    inputs = iter(["把对比两类停止加入待办，先不要改计划", answer, "只复述已登记任务", "exit"])
    seen = []

    def invoke(**kwargs):
        seen.append(kwargs["request_messages"])
        if len(seen) == 1:
            inbox = kwargs["feedback_inbox"]
            inbox.task_tool.invoke({"title": "对比两类停止", "source_id": "current"})
            return "已为您提交创建任务建议，等待确认。"
        return "按当前正式记录回答。"

    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(main, "_invoke_interactive_turn", invoke)
    monkeypatch.setattr(memory_module.ConversationMemory, "_maybe_compress", lambda self: None)
    main.run_interactive_session("", original.plan_markdown, "", original.learning_goal,
                                 strategy, str(tmp_path), initial_state=original, current_state_path=path)
    saved = load_learning_state(path)
    history = next(m.content for m in seen[1] if m.type == "ai" and "程序学习操作结果数据" in m.content)
    result = receipt(history)
    assert result["decision"] == ("confirmed" if answer == "y" else "cancelled")
    assert result["proposal_pending"] is False
    current = next(json.loads(m.content) for m in seen[1] if m.type == "human"
                   and str(m.content).startswith('{"context_type": "current_learning_state"'))
    assert len(current["tasks"]) == (2 if answer == "y" else 1)
    assert len(saved.tasks) == len(current["tasks"])
    assert saved.plan_markdown == original.plan_markdown
    assert "程序学习操作结果数据" in saved.conversation_memory.messages[1].content
    assert saved.conversation_memory.strategy == strategy


def test_summary_prompt_preserves_final_decision_not_candidate(monkeypatch):
    calls = []

    def summarize(messages):
        calls.append(messages)
        return AIMessage(content="已确认登记对比任务；不是待确认。")

    monkeypatch.setattr(memory_module, "create_llm", lambda: SimpleNamespace(invoke=summarize))
    monkeypatch.setattr(memory_module, "get_settings", lambda: SimpleNamespace(log_payloads=False))
    original = state()
    pending = prepare_task_creation(original, TaskCreationArgs(title="对比两类停止"), user_message="加入待办")
    current = commit_task_creation(original, pending)
    text = runtime.build_learning_action_history("等待确认", current, pending, "confirmed")
    memory = memory_module.ConversationMemory(system_prompt="当前规则", strategy="summary", max_chars=1, window_size=6)
    memory.add_user("加入待办")
    memory.add_ai(text)
    memory.add_user("继续讨论")
    memory.add_ai("只讨论，不修改")
    assert "程序学习操作结果数据" in calls[0][0].content
    assert '"decision": "confirmed"' in calls[0][1].content
    assert memory.summary == "已确认登记对比任务；不是待确认。"
