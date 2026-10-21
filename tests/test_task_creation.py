"""自然语言创建任务的契约、共用收件箱和真实会话接线回归。

所有模型响应使用替身，不证明真实模型总能识别创建意图。
候选不改状态；确认后默认状态、保存、下一轮及恢复记录应保持一致。
"""

import builtins
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

import ai_agent.main as main
from ai_agent.learning.feedback_runtime import FeedbackInbox, LearningActionInbox, has_unbacked_learning_proposal
from ai_agent.learning.learning_commands import handle_learning_command
from ai_agent.learning.learning_state import LearningState, LearningTask, update_learning_context
from ai_agent.persistence.memory import ConversationMemory
from ai_agent.persistence.state_store import load_learning_state
from ai_agent.learning.task_creation import TaskCreationArgs, prepare_task_creation, commit_task_creation, review_task_creation


def state():
    return LearningState(learning_goal="理解ReAct", plan_markdown="先阅读，再比较")


def args(**changes):
    return {"title": "对比两种终止条件", **changes}


def prepare(current=None):
    return prepare_task_creation(
        current or state(), TaskCreationArgs.model_validate(args()),
        user_message="把对比两种终止条件加入待办，先不要改计划。",
    )


@pytest.mark.parametrize("extra", [
    {"task_id": "模型指定"}, {"activity_status": "completed"},
    {"understanding_status": "self_reported_clear"}, {"learning_goal": "新目标"},
    {"title": " "},
])
def test_model_cannot_control_identity_progress_or_goal(extra):
    with pytest.raises(ValidationError):
        TaskCreationArgs.model_validate(args(**extra))


def test_candidate_is_read_only_and_commit_preserves_context():
    original = state()
    pending = prepare(original)
    assert original.tasks == ()
    assert pending.task_id.startswith("task-")
    result = commit_task_creation(original, pending)
    task = result.tasks[0]
    assert task.task_id == pending.task_id
    assert task.activity_status == "not_started"
    assert task.understanding_status == "unknown"
    assert task.latest_feedback == ""
    assert result.plan_markdown == original.plan_markdown
    assert result.learning_goal == original.learning_goal
    assert result.context_revision == original.context_revision
    assert result.feedback_records == original.feedback_records
    assert original.tasks == ()
    with pytest.raises(ValueError):
        commit_task_creation(result, pending)


def test_changed_revision_and_missing_source_are_rejected():
    original = state()
    with pytest.raises(ValueError, match="变化"):
        commit_task_creation(update_learning_context(original, plan_markdown="新计划"), prepare(original))
    with pytest.raises(ValueError, match="为空"):
        prepare_task_creation(original, TaskCreationArgs.model_validate(args()), user_message=" ")


@pytest.mark.parametrize("answer", ["n", "", None])
def test_cancel_and_eof_do_not_register(monkeypatch, answer):
    def read(_):
        if answer is None:
            raise EOFError
        return answer
    monkeypatch.setattr(builtins, "input", read)
    original = state()
    result, status = review_task_creation(original, prepare(original))
    assert result is original
    assert status == "cancelled"


def test_duplicate_title_has_actionable_tool_result():
    original = LearningState(
        learning_goal="目标", plan_markdown="计划",
        tasks=(LearningTask(task_id="old-1", title="Read ReAct"),),
    )
    inbox = LearningActionInbox()
    inbox.start_turn(original, "新增read   react任务")
    result = json.loads(inbox.task_tool.invoke({"title": "read   react"}))
    assert result["ok"] is False
    assert "old-1" in result["error"]["message"]
    assert inbox.pending is None
    assert len(original.tasks) == 1


@pytest.mark.parametrize("first", ["feedback", "task"])
def test_two_tool_types_share_one_pending_slot(first):
    inbox = LearningActionInbox()
    inbox.start_turn(state(), "今天只剩20分钟；把对比两种终止条件加入待办")
    feedback = {"scope": "current_context", "task_id": None,
                "activity_status": None, "understanding_status": None}
    first_tool, first_args, second_tool, second_args = (
        (inbox.tool, feedback, inbox.task_tool, args()) if first == "feedback"
        else (inbox.task_tool, args(), inbox.tool, feedback)
    )
    assert json.loads(first_tool.invoke(first_args))["ok"] is True
    pending = inbox.pending
    assert json.loads(second_tool.invoke(second_args))["error"]["code"] == "pending_exists"
    assert inbox.pending is pending
    assert FeedbackInbox is LearningActionInbox


def test_historical_user_source_can_authorize_task_proposal():
    inbox = LearningActionInbox()
    inbox.start_turn(state(), "就按刚才说的加入待办", history_messages=[
        HumanMessage(content="我想练习对比两种终止条件"),
        AIMessage(content="我可以建议别的活动"),
    ])
    result = json.loads(inbox.task_tool.invoke(args(source_id="previous_user_1")))
    assert result["ok"] is True
    assert inbox.pending.user_message == "我想练习对比两种终止条件"


def test_manual_task_command_keeps_original_semantics():
    result, message = handle_learning_command(state(), '/task {"task_id":"manual-1","title":"阅读"}')
    assert "已登记任务" in message
    assert result.tasks[0].task_id == "manual-1"
    failed, message = handle_learning_command(result, '/task {"task_id":"manual-1","title":"另一任务"}')
    assert failed is result
    assert "未执行" in message


def test_execution_claim_must_match_pending_type():
    assert has_unbacked_learning_proposal("已为您提交创建任务建议。", None)
    assert not has_unbacked_learning_proposal("已为您提交创建任务建议。", prepare())
    assert not has_unbacked_learning_proposal("已为您提交创建任务记录。", prepare())
    assert not has_unbacked_learning_proposal("当前登记的任务如下。", None)


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
@pytest.mark.parametrize("answer", ["y", "n"])
def test_chat_create_confirm_save_and_next_turn(monkeypatch, tmp_path, strategy, answer):
    original = state()
    path = tmp_path / "current.json"
    inputs = iter(["把对比两种终止条件加入待办，先不要改计划。", answer, "只复述登记任务，不修改", "exit"])
    responses = iter([
        AIMessage(content="", tool_calls=[{"name": "propose_task_creation", "args": args(), "id": "create-1"}]),
        AIMessage(content="已为您提交创建任务建议。"),
        AIMessage(content="按当前正式任务表回答。"),
    ])
    captured = []
    bound = []

    def read(_):
        value = next(inputs)
        if value == answer:
            assert not path.exists()
        return value

    def invoke(llm, messages, *, stage):
        captured.append(list(messages))
        return next(responses)

    monkeypatch.setattr(builtins, "input", read)
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda tools: bound.extend(t.name for t in tools) or object())
    monkeypatch.setattr(main, "invoke_llm_response", invoke)
    monkeypatch.setattr(ConversationMemory, "_maybe_compress", lambda self: None)
    main.run_interactive_session(
        "", original.plan_markdown, "", original.learning_goal, strategy, str(tmp_path),
        initial_state=original, current_state_path=path,
    )
    result = load_learning_state(path)
    assert "propose_task_creation" in bound
    assert len(result.tasks) == (1 if answer == "y" else 0)
    assert result.plan_markdown == original.plan_markdown
    assert result.context_revision == original.context_revision
    assert len(result.conversation_memory.messages) == 4
    records = [json.loads(m.content) for m in captured[-1] if m.type == "human" and str(m.content).startswith("{")]
    current = next(r for r in records if r.get("context_type") == "current_learning_state")
    assert len(current["tasks"]) == len(result.tasks)
    assert ("confirmed" if answer == "y" else "cancelled") in result.conversation_memory.messages[1].content
    if result.tasks:
        assert current["tasks"][0]["task_id"] == result.tasks[0].task_id
        assert result.tasks[0].activity_status == "not_started"
        source_bytes = path.read_bytes()
        restored_path = tmp_path / "restored-current.json"
        restart_inputs = iter(["只查看已有任务，不创建", "exit"])
        monkeypatch.setattr(builtins, "input", lambda _: next(restart_inputs))
        monkeypatch.setattr(main, "invoke_llm_response", lambda *a, **kw: AIMessage(content="当前登记任务保持不变。"))
        main.run_interactive_session(
            "", result.plan_markdown, "", result.learning_goal, None, str(tmp_path),
            initial_state=result, current_state_path=restored_path,
        )
        restored = load_learning_state(restored_path)
        assert restored.tasks == result.tasks
        assert restored.conversation_memory.strategy == strategy
        assert len(restored.conversation_memory.messages) == 6
        assert path.read_bytes() == source_bytes
