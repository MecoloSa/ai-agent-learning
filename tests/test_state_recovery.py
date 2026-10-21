"""状态恢复入口测试。

验证恢复后继续更新、另存快照，以及绕过计划生成。
假输入和假模型不访问在线服务。
"""

import builtins
import sys

import pytest

import ai_agent.main as main_module
import ai_agent.learning.learning_commands as commands_module
from ai_agent.learning.learning_state import (
    LearningState,
    LearningTask,
    UserFeedback,
)
from ai_agent.persistence.state_store import save_learning_state, load_learning_state


def make_state():
    return LearningState(
        learning_goal="理解文章观点",
        plan_markdown="阅读文章。",
        tasks=(
            LearningTask(
                task_id="task-001",
                title="阅读文章",
                activity_status="in_progress",
                understanding_status="self_reported_unclear",
                latest_feedback="只读了一半，没理解。",
                feedback_source="user_report",
            ),
        ),
    )


def test_recovery_updates_and_saves_without_model(monkeypatch, tmp_path):
    original = make_state()
    old_path = tmp_path / "old.json"
    save_learning_state(original, old_path)

    inputs = iter([
        "/state",
        "文章讲了什么？",
        '/feedback {"task_id":"task-001","text":"文章读完了。",'
        '"activity_status":"completed"}',
        "/save-state",
        "exit",
    ])

    def unexpected_model(*args, **kwargs):
        raise AssertionError("手动恢复流程不应创建模型")

    monkeypatch.setattr(main_module, "create_llm", unexpected_model)
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(
        commands_module,
        "resolve_project_path",
        lambda value: tmp_path / "new.json",
    )

    updated = main_module.run_state_recovery_session(
        load_learning_state(old_path)
    )

    assert updated.tasks[0].activity_status == "completed"
    assert updated.tasks[0].understanding_status == "self_reported_unclear"
    assert load_learning_state(tmp_path / "new.json") == updated
    assert load_learning_state(old_path) == original


def test_recovery_report_reuses_confirmation(monkeypatch):
    state = make_state()
    inputs = iter([
        "/report task-001 文章读完了。",
        "y",
        "exit",
    ])
    roles = []

    def fake_create_llm(*, role="standard"):
        roles.append(role)
        return object()

    def fake_proposal(state, task_id, text, llm):
        return UserFeedback(
            task_id=task_id,
            text=text,
            activity_status="completed",
        )

    monkeypatch.setattr(main_module, "create_llm", fake_create_llm)
    monkeypatch.setattr(main_module, "propose_user_feedback", fake_proposal)
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    updated = main_module.run_state_recovery_session(state)

    assert roles == ["feedback"]
    assert updated.tasks[0].activity_status == "completed"
    assert updated.tasks[0].understanding_status == "self_reported_unclear"
    assert state.tasks[0].activity_status == "in_progress"


def test_cli_resume_skips_plan_generation(monkeypatch, tmp_path):
    state = make_state()
    path = tmp_path / "state.json"
    save_learning_state(state, path)
    received = []

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("恢复模式不能重新生成计划")

    def fake_session(restored):
        received.append(restored)
        return restored

    monkeypatch.setattr(
        sys, "argv",
        ["learning-agent", "--resume-state", str(path)],
    )
    monkeypatch.setattr(
        main_module, "run_learning_assistant", unexpected_generation,
    )
    monkeypatch.setattr(
        main_module, "run_state_recovery_session", fake_session,
    )

    main_module.main()

    assert received == [state]


@pytest.mark.parametrize("file_exists", [False, True])
def test_invalid_snapshot_does_not_start_session(
    monkeypatch, tmp_path, capsys, file_exists,
):
    path = tmp_path / "invalid.json"
    if file_exists:
        path.write_text("{broken", encoding="utf-8")

    def unexpected_call(*args, **kwargs):
        raise AssertionError("快照无效时不能启动会话或新任务")

    monkeypatch.setattr(
        sys, "argv",
        ["learning-agent", "--resume-state", str(path)],
    )
    monkeypatch.setattr(
        main_module, "run_state_recovery_session", unexpected_call,
    )
    monkeypatch.setattr(
        main_module, "run_learning_assistant", unexpected_call,
    )

    main_module.main()

    assert "无法恢复快照" in capsys.readouterr().out


def test_goal_and_resume_cannot_be_used_together(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        [
            "learning-agent",
            "--goal", "新目标",
            "--resume-state", "state.json",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main_module.main()

    assert exc.value.code == 2
