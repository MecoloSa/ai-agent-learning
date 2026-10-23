"""取消任务聊天工具：真实候选、明确确认、共享预算和保存恢复。

模型使用替身，验证控制流而非真实模型的意图理解准确率。
本文件需配合本轮业务接线代码运行。
"""
import builtins
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

import ai_agent.main as main
from ai_agent.learning.feedback_runtime import LearningActionInbox, build_learning_action_history, has_unbacked_learning_proposal
from ai_agent.learning.learning_state import LearningState, LearningTask, update_learning_context
from ai_agent.learning.task_cancellation import (
    TaskCancellationArgs, prepare_task_cancellation, commit_task_cancellation, review_task_cancellation,
)
from ai_agent.persistence.memory import ConversationMemory
from ai_agent.persistence.state_store import load_learning_state
from ai_agent.runtime.execution_guard import tool_result_succeeded


def state():
    return LearningState(
        learning_goal="区分两类停止", plan_markdown="原计划保留",
        tasks=(LearningTask(task_id="draw-003", title="画出两种停止的流转", activity_status="in_progress",
                            understanding_status="self_reported_unclear", latest_feedback="画到一半", feedback_source="user_report"),),
    )


def pending(original=None):
    return prepare_task_cancellation(original or state(), TaskCancellationArgs(task_id="draw-003"),
                                     user_message="取消画图任务，先不改计划")


@pytest.mark.parametrize("extra", [{"text": "模型改写原文"}, {"is_cancelled": True}, {"context_revision": 99}, {"activity_status": "completed"}])
def test_model_cannot_supply_origin_or_execution_fields(extra):
    with pytest.raises(ValidationError):
        TaskCancellationArgs.model_validate({"task_id": "draw-003", **extra})


def test_prepare_is_readonly_and_commit_preserves_history():
    original = state()
    candidate = pending(original)
    assert not original.tasks[0].is_cancelled
    result = commit_task_cancellation(original, candidate)
    assert result.tasks[0].is_cancelled
    assert result.tasks[0].activity_status == "in_progress"
    assert result.tasks[0].latest_feedback == "画到一半"
    assert result.plan_markdown == original.plan_markdown
    assert result.context_revision == original.context_revision
    with pytest.raises(ValueError, match="已经取消"):
        commit_task_cancellation(result, candidate)
    with pytest.raises(ValueError, match="版本"):
        commit_task_cancellation(update_learning_context(original, learning_goal="另一目标"), candidate)


def test_source_resolution_is_program_owned_and_unknown_source_rejected():
    inbox = LearningActionInbox()
    inbox.start_turn(state(), "就取消刚才那个", history_messages=[HumanMessage(content="取消画图，不改计划")])
    result = json.loads(inbox.cancellation_tool.invoke({"task_id": "draw-003", "source_id": "previous_user_1"}))
    assert result["ok"]
    assert inbox.pending.user_message == "取消画图，不改计划"
    inbox.start_turn(state(), "取消")
    result = json.loads(inbox.cancellation_tool.invoke({"task_id": "draw-003", "source_id": "invented"}))
    assert not result["ok"]
    assert inbox.pending is None


def test_missing_and_cancelled_tasks_are_rejected_without_pending():
    inbox = LearningActionInbox()
    for original, task_id in [(state(), "missing"), (commit_task_cancellation(state(), pending()), "draw-003")]:
        inbox.start_turn(original, "取消")
        result = json.loads(inbox.cancellation_tool.invoke({"task_id": task_id}))
        assert not result["ok"]
        assert inbox.pending is None


@pytest.mark.parametrize("first,second", [("cancel", "feedback"), ("cancel", "create"), ("feedback", "cancel"), ("create", "cancel")])
def test_three_actions_share_one_pending_slot(first, second):
    inbox = LearningActionInbox()
    inbox.start_turn(state(), "操作")
    tools = {
        "cancel": (inbox.cancellation_tool, {"task_id": "draw-003"}),
        "create": (inbox.task_tool, {"title": "复述"}),
        "feedback": (inbox.tool, {"scope": "task", "task_id": "draw-003", "activity_status": None, "understanding_status": None}),
    }
    tool, args = tools[first]
    assert json.loads(tool.invoke(args))["ok"]
    original_pending = inbox.pending
    tool, args = tools[second]
    assert json.loads(tool.invoke(args))["error"]["code"] == "pending_exists"
    assert inbox.pending is original_pending


@pytest.mark.parametrize("answer", ["n", ""])
def test_declining_cancellation_proposal_preserves_registered_task(monkeypatch, answer):
    original = state()
    monkeypatch.setattr(builtins, "input", lambda _: answer)
    result, decision = review_task_cancellation(original, pending(original))
    assert result is original
    assert decision == "cancelled"


def test_eof_declines_proposal(monkeypatch):
    def eof(_):
        raise EOFError
    monkeypatch.setattr(builtins, "input", eof)
    original = state()
    result, decision = review_task_cancellation(original, pending(original))
    assert result is original and decision == "cancelled"


def test_cancellation_claim_and_receipt_use_correct_action_type():
    proposal = pending()
    assert not has_unbacked_learning_proposal("已提交取消任务建议。", proposal)
    assert has_unbacked_learning_proposal("已提交取消任务建议。", None)
    assert has_unbacked_learning_proposal("已提交创建任务建议。", proposal)
    result = commit_task_cancellation(state(), proposal)
    text = build_learning_action_history("等待确认", result, proposal, "confirmed")
    receipt = json.loads(text.split("\n【程序学习操作结果数据】\n")[1])
    assert receipt["action"] == "task_cancellation"
    assert receipt["task_after"]["is_cancelled"]
    assert receipt["decision"] == "confirmed"


def test_tool_success_means_pending_not_cancelled():
    assert tool_result_succeeded("propose_task_cancellation", '{"ok":true,"data":{"status":"pending"}}')
    assert not tool_result_succeeded("propose_task_cancellation", '{"ok":false,"error":{}}')


def test_progress_feedback_on_cancelled_task_is_rejected_before_confirmation(monkeypatch):
    from ai_agent.learning.feedback_proposals import FeedbackProposalArgs, prepare_feedback_proposal
    from ai_agent.learning.feedback_runtime import review_feedback_proposal
    original = commit_task_cancellation(state(), pending())
    proposal = prepare_feedback_proposal(original, FeedbackProposalArgs(
        scope="task", task_id="draw-003", activity_status="completed", understanding_status=None,
    ), user_message="把画图任务标为完成")
    monkeypatch.setattr(builtins, "input", lambda _: pytest.fail("非法进度候选不应请求确认"))
    result, decision = review_feedback_proposal(original, proposal)
    assert result is original
    assert decision == "invalid"


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
@pytest.mark.parametrize("answer", ["y", "n"])
def test_chat_cancel_confirm_save_next_turn_and_restore(monkeypatch, tmp_path, strategy, answer):
    original = state()
    path = tmp_path / "current.json"
    inputs = iter(["取消画图任务，先不要改计划", answer, "只查询任务是否取消", "exit"])
    responses = iter([
        AIMessage(content="", tool_calls=[{"name": "propose_task_cancellation", "args": {"task_id": "draw-003"}, "id": "cancel-1"}]),
        AIMessage(content="已提交取消任务建议。"),
        AIMessage(content="以最新状态为准。"),
    ])
    captured, bound = [], []

    def invoke(llm, messages, *, stage):
        captured.append(list(messages))
        return next(responses)

    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda tools: bound.extend(t.name for t in tools) or object())
    monkeypatch.setattr(main, "invoke_llm_response", invoke)
    monkeypatch.setattr(ConversationMemory, "_maybe_compress", lambda self: None)
    main.run_interactive_session("", original.plan_markdown, "", original.learning_goal, strategy, str(tmp_path),
                                 initial_state=original, current_state_path=path)
    saved = load_learning_state(path)
    assert "propose_task_cancellation" in bound
    assert saved.tasks[0].is_cancelled is (answer == "y")
    assert saved.tasks[0].activity_status == "in_progress"
    assert saved.tasks[0].latest_feedback == "画到一半"
    assert saved.plan_markdown == original.plan_markdown
    assert saved.context_revision == original.context_revision
    current = next(json.loads(m.content) for m in captured[-1] if m.type == "human"
                   and str(m.content).startswith('{"context_type": "current_learning_state"'))
    assert current["tasks"][0]["is_cancelled"] is (answer == "y")
    history = saved.conversation_memory.messages[1].content
    receipt = json.loads(history.split("\n【程序学习操作结果数据】\n")[1])
    assert receipt["action"] == "task_cancellation"
    assert receipt["decision"] == ("confirmed" if answer == "y" else "cancelled")
    assert saved.conversation_memory.strategy == strategy
    source_bytes = path.read_bytes()
    restart_inputs = iter(["只查询，不修改", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(restart_inputs))
    monkeypatch.setattr(main, "invoke_llm_response", lambda *a, **kw: AIMessage(content="当前记录不变。"))
    restored_path = tmp_path / "restored.json"
    main.run_interactive_session("", saved.plan_markdown, "", saved.learning_goal, None, str(tmp_path),
                                 initial_state=saved, current_state_path=restored_path)
    assert load_learning_state(restored_path).tasks == saved.tasks
    assert path.read_bytes() == source_bytes


def test_same_turn_plan_change_and_cancellation_are_both_withheld(monkeypatch, tmp_path):
    original = state()
    inputs = iter(["取消任务并修改计划", "exit"])
    def invoke(**kwargs):
        kwargs["feedback_inbox"].cancellation_tool.invoke({"task_id": "draw-003"})
        return "===NEW_PLAN_START===新计划===NEW_PLAN_END==="
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(main, "_invoke_interactive_turn", invoke)
    monkeypatch.setattr(ConversationMemory, "_maybe_compress", lambda self: None)
    path = tmp_path / "current.json"
    main.run_interactive_session("", original.plan_markdown, "", original.learning_goal, "window", str(tmp_path),
                                 initial_state=original, current_state_path=path)
    saved = load_learning_state(path)
    assert not saved.tasks[0].is_cancelled
    assert saved.plan_markdown == original.plan_markdown
