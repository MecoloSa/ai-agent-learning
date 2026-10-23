"""取消任务的状态契约：保留历史、拒绝隐式重启、兼容旧存档。

本轮只验证底层数据能力，尚不代表聊天取消工具已接入。
"""
import json

import pytest
from pydantic import ValidationError

from ai_agent.learning.learning_state import LearningState, LearningTask, UserFeedback, apply_user_feedback
from ai_agent.learning.learning_context import build_learning_state_message
from ai_agent.learning.task_cancellation import TaskCancellation, cancel_learning_task
from ai_agent.persistence.state_store import load_learning_state, save_learning_state


def state(activity="in_progress"):
    return LearningState(
        learning_goal="区分两类停止", plan_markdown="原计划",
        tasks=(
            LearningTask(task_id="t1", title="画图", activity_status=activity,
                         understanding_status="self_reported_unclear",
                         latest_feedback="图画了一半", feedback_source="user_report"),
            LearningTask(task_id="t2", title="复述区别"),
        ),
    )


def cancel(original):
    return cancel_learning_task(original, TaskCancellation(task_id="t1", text="取消画图任务，先不改计划"))


@pytest.mark.parametrize("activity", ["not_started", "in_progress", "completed"])
def test_cancellation_preserves_progress_feedback_and_other_state(activity):
    original = state(activity)
    updated = cancel(original)
    assert updated.tasks[0].is_cancelled is True
    assert updated.tasks[0].cancellation_reason == "取消画图任务，先不改计划"
    assert updated.tasks[0].activity_status == activity
    assert updated.tasks[0].understanding_status == "self_reported_unclear"
    assert updated.tasks[0].latest_feedback == "图画了一半"
    assert updated.tasks[0].feedback_source == "user_report"
    assert updated.tasks[1] == original.tasks[1]
    assert updated.plan_markdown == original.plan_markdown
    assert updated.context_revision == original.context_revision
    assert updated.feedback_records == original.feedback_records
    assert original.tasks[0].is_cancelled is False


def test_missing_task_and_repeated_cancellation_do_not_mutate_state():
    original = state()
    with pytest.raises(ValueError, match="不存在"):
        cancel_learning_task(original, TaskCancellation(task_id="missing", text="取消"))
    updated = cancel(original)
    with pytest.raises(ValueError, match="已经取消"):
        cancel(updated)
    assert updated.tasks[0].cancellation_reason == "取消画图任务，先不改计划"


@pytest.mark.parametrize("payload", [
    {"task_id": "t1", "text": " "},
    {"task_id": " ", "text": "取消"},
    {"task_id": "t1", "text": "取消", "activity_status": "completed"},
])
def test_cancellation_parameters_reject_empty_or_unrelated_fields(payload):
    with pytest.raises(ValidationError):
        TaskCancellation.model_validate(payload)


@pytest.mark.parametrize("changes", [
    {"is_cancelled": True},
    {"is_cancelled": False, "cancellation_reason": "已取消"},
    {"is_cancelled": "true", "cancellation_reason": "取消"},
])
def test_cancel_flag_and_reason_cannot_contradict(changes):
    with pytest.raises(ValidationError):
        LearningTask(task_id="t1", title="画图", **changes)


def test_progress_feedback_cannot_implicitly_reopen_cancelled_task():
    updated = cancel(state())
    with pytest.raises(ValueError, match="已取消"):
        apply_user_feedback(updated, UserFeedback(task_id="t1", text="继续做", activity_status="completed"))
    assert updated.tasks[0].activity_status == "in_progress"


def test_understanding_can_be_corrected_without_reopening():
    updated = cancel(state())
    corrected = apply_user_feedback(updated, UserFeedback(
        task_id="t1", text="虽不继续画图，但现在理解了", understanding_status="self_reported_clear",
    ))
    assert corrected.tasks[0].is_cancelled is True
    assert corrected.tasks[0].activity_status == "in_progress"
    assert corrected.tasks[0].understanding_status == "self_reported_clear"
    assert corrected.tasks[0].cancellation_reason == updated.tasks[0].cancellation_reason


def test_context_exposes_cancellation_without_hiding_historical_progress():
    message = json.loads(build_learning_state_message(cancel(state())).content)
    task = message["tasks"][0]
    assert task["is_cancelled"] is True
    assert task["cancellation_reason"] == "取消画图任务，先不改计划"
    assert task["activity_status"] == "in_progress"


@pytest.mark.parametrize("version", [f"learning-state-v{i}" for i in range(1, 6)])
def test_old_snapshots_default_to_not_cancelled(tmp_path, version):
    data = state().model_dump(mode="json")
    for task in data["tasks"]:
        task.pop("is_cancelled")
        task.pop("cancellation_reason")
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": version, "state": data}), encoding="utf-8")
    original_bytes = path.read_bytes()
    restored = load_learning_state(path)
    assert all(not task.is_cancelled and task.cancellation_reason == "" for task in restored.tasks)
    assert path.read_bytes() == original_bytes


def test_v6_save_and_restore_preserves_cancelled_task(tmp_path):
    updated = cancel(state())
    path = save_learning_state(updated, tmp_path / "new.json")
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "learning-state-v7"
    assert load_learning_state(path) == updated
