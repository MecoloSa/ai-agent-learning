"""程序附加原话的契约回归；保留文件名以兼容已有测试命令。

不再测试被撤回的引号兼容热路径。模型只选来源，不复制或改写用户原文。
离线测试不证明模型一定选对来源或理解正确，确认预览仍须核对。
"""

import builtins
import json

import pytest
from pydantic import ValidationError

from ai_agent.learning.feedback_proposals import FeedbackProposalArgs
from ai_agent.learning.feedback_runtime import LearningActionInbox, commit_feedback_proposal, review_feedback_proposal
from ai_agent.learning.learning_context import build_learning_state_message
from ai_agent.learning.learning_state import LearningState, LearningTask
from ai_agent.persistence.state_store import save_learning_state, load_learning_state
from ai_agent.learning.task_creation import TaskCreationArgs, commit_task_creation, review_task_creation


def state():
    return LearningState(
        learning_goal="理解两种停止条件", plan_markdown="原计划",
        tasks=(LearningTask(task_id="t1", title="阅读文章"),),
    )


def feedback_args(**changes):
    return {
        "source_id": "current", "scope": "current_context", "task_id": None,
        "activity_status": None, "understanding_status": None, **changes,
    }


@pytest.mark.parametrize("model, base", [
    (TaskCreationArgs, {"title": "对比停止条件"}),
    (FeedbackProposalArgs, feedback_args()),
])
def test_schema_does_not_request_quote(model, base):
    assert "quote" not in model.model_json_schema()["properties"]
    assert "quote" not in model.model_validate(base).model_dump()


@pytest.mark.parametrize("field", ["quote", "text", "user_message"])
@pytest.mark.parametrize("model, base", [
    (TaskCreationArgs, {"title": "对比停止条件"}),
    (FeedbackProposalArgs, feedback_args()),
])
def test_model_cannot_supply_original_text(model, base, field):
    with pytest.raises(ValidationError):
        model.model_validate({**base, field: "今天剩40分钟，已经完全理解"})


def test_task_creation_no_longer_requires_repeating_quotes(monkeypatch, capsys):
    raw = '把“对比正常结束与保护性停止”作为新任务加入待办，先不要改学习计划。'
    inbox = LearningActionInbox()
    original = state()
    inbox.start_turn(original, raw)
    response = json.loads(inbox.task_tool.invoke({
        "title": "对比正常结束与保护性停止", "source_id": "current",
    }))
    assert response["data"]["status"] == "pending"
    assert inbox.pending.user_message == raw
    assert len(original.tasks) == 1
    monkeypatch.setattr(builtins, "input", lambda _: "n")
    result, decision = review_task_creation(original, inbox.pending)
    assert result is original
    assert decision == "cancelled"
    assert raw in capsys.readouterr().out


def test_record_original_text_and_keep_v5_compatible(tmp_path):
    raw = '今天只剩“20分钟”，先不要改计划。'
    inbox = LearningActionInbox()
    original = state()
    inbox.start_turn(original, raw)
    inbox.tool.invoke(feedback_args())
    confirmed = commit_feedback_proposal(original, inbox.pending)
    record = confirmed.feedback_records[0]
    assert record.quote == raw
    assert record.text == raw
    assert original.feedback_records == ()
    path = tmp_path / "feedback.json"
    save_learning_state(confirmed, path)
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "learning-state-v7"
    restored = load_learning_state(path)
    assert restored.feedback_records == confirmed.feedback_records
    data = json.loads(build_learning_state_message(restored).content)
    assert data["current_context_feedback"][0]["text"] == raw
    assert "quote" not in data["current_context_feedback"][0]


def test_old_distinct_quote_is_retained_in_context():
    inbox = LearningActionInbox()
    inbox.start_turn(state(), "今天只剩20分钟，先不要改计划")
    inbox.tool.invoke(feedback_args())
    confirmed = commit_feedback_proposal(state(), inbox.pending)
    old_record = confirmed.feedback_records[0].model_dump()
    old_record["quote"] = "今天只剩20分钟"
    old = LearningState.model_validate({
        **confirmed.model_dump(), "feedback_records": [old_record],
    })
    data = json.loads(build_learning_state_message(old).content)
    assert data["current_context_feedback"][0]["quote"] == "今天只剩20分钟"


def test_authentic_source_does_not_prove_suggested_status(monkeypatch, capsys):
    raw = "我还是不理解，先不要改计划。"
    inbox = LearningActionInbox()
    original = state()
    inbox.start_turn(original, raw)
    inbox.tool.invoke(feedback_args(
        scope="task", task_id="t1", understanding_status="self_reported_clear",
    ))
    assert inbox.pending.user_message == raw
    assert original.tasks[0].understanding_status == "unknown"
    monkeypatch.setattr(builtins, "input", lambda _: "n")
    result, decision = review_feedback_proposal(original, inbox.pending)
    assert result is original
    assert decision == "cancelled"
    preview = capsys.readouterr().out
    assert raw in preview
    assert "self_reported_clear" in preview


def test_confirmed_task_creation_still_uses_program_id_and_default_status():
    inbox = LearningActionInbox()
    original = state()
    inbox.start_turn(original, "请新增对比两种停止条件的任务")
    inbox.task_tool.invoke({"title": "对比两种停止条件"})
    result = commit_task_creation(original, inbox.pending)
    assert result.tasks[-1].task_id == inbox.pending.task_id
    assert result.tasks[-1].activity_status == "not_started"
    assert result.plan_markdown == original.plan_markdown
