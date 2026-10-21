"""学习状态自动保存测试。

使用假输入、假反馈解析和临时目录，不访问在线模型或真实学习记录。
验证内容变化判断、失败提示，以及恢复会话中的自动保存接线。
"""

import builtins

import pytest

import ai_agent.main as main_module
from ai_agent.learning.learning_state import (
    LearningState,
    LearningTask,
    UserFeedback,
    apply_user_feedback,
)
from ai_agent.persistence.state_store import save_learning_state, load_learning_state


def make_state():
    return LearningState(
        learning_goal="理解文章观点",
        plan_markdown="阅读并比较观点。",
        tasks=(LearningTask(task_id="task-001", title="阅读文章"),),
    )


def test_equal_content_does_not_save(monkeypatch):
    before = make_state()
    after = LearningState.model_validate(before.model_dump())
    assert after is not before

    def unexpected_save(state):
        raise AssertionError("内容相同时不应保存")

    monkeypatch.setattr(
        main_module, "save_current_learning_state", lambda state, _path: (unexpected_save)(state),
    )

    assert main_module._autosave_learning_change(before, after) is None


def test_changed_state_is_saved_without_mutation(monkeypatch, tmp_path):
    before = make_state()
    after = apply_user_feedback(
        before,
        UserFeedback(
            task_id="task-001",
            text="读了一半。",
            activity_status="in_progress",
        ),
    )
    path = tmp_path / "auto.json"

    monkeypatch.setattr(
        main_module,
        "save_current_learning_state",
        lambda state, _path: (lambda state: save_learning_state(state, path))(state),
    )

    assert main_module._autosave_learning_change(before, after) == path
    assert load_learning_state(path) == after
    assert before.tasks[0].activity_status == "not_started"
    assert after.tasks[0].activity_status == "in_progress"


def test_failed_autosave_keeps_memory_and_warns(monkeypatch, capsys):
    before = make_state()
    after = apply_user_feedback(
        before,
        UserFeedback(
            task_id="task-001",
            text="读完了。",
            activity_status="completed",
        ),
    )

    def fail_save(state):
        raise OSError("模拟磁盘写入失败")

    monkeypatch.setattr(main_module, "save_current_learning_state", lambda state, _path: (fail_save)(state))

    assert main_module._autosave_learning_change(before, after) is None

    output = capsys.readouterr().out
    assert "自动保存失败" in output
    assert "尚未保存到磁盘" in output
    assert "自动保存成功" not in output
    assert after.tasks[0].activity_status == "completed"


@pytest.mark.parametrize(
    "operation, expected_saves",
    [
        ("manual_feedback", 1),
        ("confirmed_report", 1),
        ("cancelled_report", 0),
        ("query", 0),
        ("invalid_feedback", 0),
    ],
)
def test_recovery_saves_only_actual_changes(
    monkeypatch, tmp_path, operation, expected_saves,
):
    state = make_state()
    saved_states = []

    if operation == "manual_feedback":
        inputs = [
            '/feedback {"task_id":"task-001","text":"读完了。",'
            '"activity_status":"completed"}',
            "exit",
        ]
    elif operation in {"confirmed_report", "cancelled_report"}:
        inputs = [
            "/report task-001 读完了。",
            "y" if operation == "confirmed_report" else "n",
            "exit",
        ]
    elif operation == "query":
        inputs = ["/state", "exit"]
    else:
        inputs = [
            '/feedback {"task_id":"missing","text":"读完了。"}',
            "exit",
        ]

    answers = iter(inputs)
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))

    def fake_proposal(state, task_id, text, llm):
        return UserFeedback(
            task_id=task_id,
            text=text,
            activity_status="completed",
        )

    def fake_save(current):
        path = tmp_path / f"auto-{len(saved_states)}.json"
        save_learning_state(current, path)
        saved_states.append(current)
        return path

    # 模型职责被假函数替代；仍执行真实确认、状态校验和保存流程。
    monkeypatch.setattr(main_module, "create_llm", lambda **kwargs: object())
    monkeypatch.setattr(main_module, "propose_user_feedback", fake_proposal)
    monkeypatch.setattr(main_module, "save_current_learning_state", lambda state, _path: (fake_save)(state))

    updated = main_module.run_state_recovery_session(state)

    assert len(saved_states) == expected_saves
    if expected_saves:
        assert updated.tasks[0].activity_status == "completed"
        assert load_learning_state(tmp_path / "auto-0.json") == updated
    else:
        assert updated == state

    print(f"\n操作={operation}，实际自动保存次数={len(saved_states)}")

def test_interactive_task_registration_autosaves(monkeypatch, tmp_path):
    """普通会话登记任务后，必须实际保存，而不只是打印登记成功。"""
    inputs = iter([
        '/task {"task_id":"task-001","title":"理解ReAct循环机制"}',
        "/state",
        "exit",
    ])
    saved_states = []
    path = tmp_path / "registered.json"

    def fake_save(state):
        save_learning_state(state, path)
        saved_states.append(state)
        return path

    def unexpected_chat(**kwargs):
        raise AssertionError("状态命令不应进入模型问答")

    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(main_module, "create_llm", lambda: object())
    monkeypatch.setattr(main_module, "save_current_learning_state", lambda state, _path: (fake_save)(state))
    monkeypatch.setattr(
        main_module, "_invoke_interactive_turn", unexpected_chat,
    )

    main_module.run_interactive_session(
        analysis="资料分析",
        plan="阅读并解释ReAct循环。",
        material_context="测试资料",
        learning_goal="理解ReAct",
        strategy="window",
        material_dir="materials",
    )

    # 登记保存一次；紧接着的 /state 查询不应再次保存。
    assert len(saved_states) == 1
    restored = load_learning_state(path)
    assert restored == saved_states[0]
    assert restored.tasks[0].task_id == "task-001"
    assert restored.tasks[0].activity_status == "not_started"
