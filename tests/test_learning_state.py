"""学习状态模块的离线测试。

不调用 LLM、不读取学习资料、不写入持久化文件。
验证数据更新规则，而不是验证模型理解反馈的能力。
"""

import pytest
from pydantic import ValidationError

from ai_agent.learning.learning_state import (
    LearningState,
    LearningTask,
    UserFeedback,
    apply_user_feedback,
)


def make_state() -> LearningState:
    return LearningState(
        learning_goal="理解文章观点",
        plan_markdown="第1天阅读，第2天比较观点。",
        tasks=(
            LearningTask(task_id="task-001", title="阅读文章"),
            LearningTask(task_id="task-002", title="比较观点"),
        ),
    )


def test_reading_completed_does_not_mean_understood():
    state = make_state()

    print("\n[阶段1] 创建初始学习状态")
    print(state.model_dump_json(indent=2))

    feedback = UserFeedback(
        task_id="task-001",
        text="文章读完了，但两个观点的区别没理解。",
        activity_status="completed",
        understanding_status="self_reported_unclear",
    )

    print("\n[阶段2] 创建结构化反馈")
    print("注意：状态由测试明确填写，并非从反馈文本自动推断。")
    print(feedback.model_dump_json(indent=2))

    print("\n[阶段3] 调用 apply_user_feedback")
    updated = apply_user_feedback(state, feedback)

    print("\n[阶段4] 查看任务更新前后")
    # zip() 把新旧任务按顺序配对；当前更新函数不改变任务顺序。
    # strict=True 要求两个序列长度相同，避免任务意外丢失时被静默忽略。
    for before, after in zip(state.tasks, updated.tasks, strict=True):
        print(f"\n任务：{before.task_id} / {before.title}")
        print(
            f"  活动进度：{before.activity_status}"
            f" -> {after.activity_status}"
        )
        print(
            f"  理解情况：{before.understanding_status}"
            f" -> {after.understanding_status}"
        )
        print(f"  最新反馈：{after.latest_feedback!r}")
        print(f"  反馈来源：{after.feedback_source}")

    print("\n[阶段5] 自动检查更新规则")

    # 打印只便于观察；真正判断对错的是下面的断言。
    assert updated.tasks[0].activity_status == "completed"
    assert updated.tasks[0].understanding_status == "self_reported_unclear"
    assert updated.tasks[0].feedback_source == "user_report"

    assert updated.tasks[1] == state.tasks[1]
    assert state.tasks[0].activity_status == "not_started"
    assert updated.plan_markdown == state.plan_markdown
    assert updated.learning_goal == state.learning_goal

    print("检查通过：目标任务正确更新，其他任务、旧状态和计划均未误改。")


def test_unmentioned_understanding_is_preserved():
    state = make_state()
    state = apply_user_feedback(
        state,
        UserFeedback(
            task_id="task-001",
            text="目前还没理解。",
            understanding_status="self_reported_unclear",
        ),
    )

    updated = apply_user_feedback(
        state,
        UserFeedback(
            task_id="task-001",
            text="今天已经读完。",
            activity_status="completed",
        ),
    )

    # 第二次只反馈活动进度，不能抹掉第一次记录的理解困难。
    assert updated.tasks[0].understanding_status == "self_reported_unclear"


def test_unknown_task_is_rejected():
    state = make_state()

    with pytest.raises(ValueError, match="任务不存在"):
        apply_user_feedback(
            state,
            UserFeedback(task_id="missing", text="已经完成。"),
        )

    assert state.tasks[0].activity_status == "not_started"


def test_duplicate_task_ids_are_rejected():
    with pytest.raises(ValidationError, match="task_id 不能重复"):
        LearningState(
            learning_goal="理解文章",
            plan_markdown="阅读并比较。",
            tasks=(
                LearningTask(task_id="same", title="阅读"),
                LearningTask(task_id="same", title="比较"),
            ),
        )


@pytest.mark.parametrize("bad_fields", [
    {"activity_status": "finished"},  # 未定义的状态值。
    {"activity_stauts": "completed"},  # 拼错字段不能被静默忽略。
    {"text": "   "},                   # 空白反馈没有有效内容。
])
def test_invalid_feedback_is_rejected(bad_fields):
    data = {"task_id": "task-001", "text": "已经读完。"}
    data.update(bad_fields)

    with pytest.raises(ValidationError):
        UserFeedback(**data)


def test_user_can_correct_previous_progress():
    state = apply_user_feedback(
        make_state(),
        UserFeedback(
            task_id="task-001",
            text="已经完成。",
            activity_status="completed",
        ),
    )

    corrected = apply_user_feedback(
        state,
        UserFeedback(
            task_id="task-001",
            text="刚才说错了，其实只读了一半。",
            activity_status="in_progress",
        ),
    )

    # 不强制进度只能向前：用户需要能够纠正误报。
    assert corrected.tasks[0].activity_status == "in_progress"
    assert state.tasks[0].activity_status == "completed"


def test_direct_mutation_is_rejected():
    state = make_state()

    with pytest.raises(ValidationError):
        state.tasks[0].activity_status = "completed"


def test_state_can_round_trip_through_json():
    state = make_state()

    # 只验证未来保存/加载所需的序列化能力，尚未真正写入磁盘。
    restored = LearningState.model_validate_json(state.model_dump_json())

    assert restored == state
