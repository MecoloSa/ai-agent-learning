"""当前学习上下文的更新、保存和兼容测试。

重点防止初始记录被覆盖，以及改计划后登记任务导致版本重置。
所有模型行为均使用替身，不调用在线服务。
"""

import builtins
import json

import pytest
from pydantic import ValidationError

import ai_agent.main as main_module
import ai_agent.learning.learning_commands as commands_module
from ai_agent.learning.learning_state import (
    LearningState,
    LearningTask,
    UserFeedback,
    apply_user_feedback,
    update_learning_context,
)
from ai_agent.persistence.state_store import save_learning_state, load_learning_state


def make_state():
    return LearningState(
        learning_goal="旧目标",
        plan_markdown="旧计划",
        tasks=(
            LearningTask(
                task_id="task-001",
                title="阅读文章",
                activity_status="completed",
            ),
        ),
    )


def test_update_preserves_initial_context_and_progress():
    state = make_state()
    updated = update_learning_context(
        state,
        learning_goal="新目标",
        plan_markdown="新计划",
    )

    assert updated.learning_goal == "新目标"
    assert updated.plan_markdown == "新计划"
    assert updated.initial_learning_goal == "旧目标"
    assert updated.initial_plan_markdown == "旧计划"
    assert updated.context_revision == 2
    assert updated.tasks_need_review is True
    assert updated.tasks == state.tasks
    assert state.context_revision == 1

    again = update_learning_context(updated, plan_markdown="第三版计划")
    assert again.initial_plan_markdown == "旧计划"
    assert again.context_revision == 3


def test_same_content_does_not_increment_revision():
    state = make_state()
    updated = update_learning_context(state, plan_markdown=" 旧计划 ")
    assert updated is state


@pytest.mark.parametrize("changes", [
    {"learning_goal": "   "},
    {"plan_markdown": ""},
])
def test_invalid_context_does_not_change_original(changes):
    state = make_state()

    with pytest.raises(ValidationError):
        update_learning_context(state, **changes)

    assert state.plan_markdown == "旧计划"
    assert state.context_revision == 1


def test_register_and_feedback_preserve_revision_metadata():
    state = update_learning_context(make_state(), plan_markdown="新计划")
    registered, message = commands_module.handle_learning_command(
        state,
        '/task {"task_id":"task-002","title":"比较观点"}',
    )

    assert "已登记任务" in message
    assert registered.initial_plan_markdown == "旧计划"
    assert registered.context_revision == 2
    assert registered.tasks_need_review is True

    updated = apply_user_feedback(
        registered,
        UserFeedback(task_id="task-002", text="开始比较了。",
                     activity_status="in_progress"),
    )
    assert updated.context_revision == 2
    assert updated.tasks_need_review is True


def test_current_snapshot_preserves_current_and_initial_context(tmp_path):
    state = update_learning_context(make_state(), plan_markdown="新计划")
    path = tmp_path / "current.json"
    save_learning_state(state, path)

    assert json.loads(path.read_text(encoding="utf-8"))[
        "schema_version"
    ] == "learning-state-v5"
    assert load_learning_state(path) == state


def test_legacy_snapshot_uses_available_context_as_baseline(tmp_path):
    path = tmp_path / "v1.json"
    path.write_text(
        json.dumps({
            "schema_version": "learning-state-v1",
            "state": {
                "learning_goal": "旧文件目标",
                "plan_markdown": "旧文件计划",
                "tasks": [],
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    restored = load_learning_state(path)
    assert restored.learning_goal == "旧文件目标"
    assert restored.initial_learning_goal == "旧文件目标"
    assert restored.initial_plan_markdown == "旧文件计划"
    assert restored.context_revision == 1


def test_chat_update_register_save_and_restore(monkeypatch, tmp_path):
    path = tmp_path / "saved.json"
    inputs = iter([
        '/task {"task_id":"task-001","title":"阅读文章"}',
        '/feedback {"task_id":"task-001","text":"已读完。",'
        '"activity_status":"completed"}',
        "请修改目标和计划",
        "y",
        "解释新计划",
        '/task {"task_id":"task-002","title":"比较观点"}',
        "/save-state",
        "exit",
    ])
    requests = []

    def fake_turn(**kwargs):
        requests.append(kwargs["request_messages"])
        if len(requests) == 1:
            return (
                "===NEW_GOAL_START===新目标===NEW_GOAL_END==="
                "===NEW_PLAN_START===新计划===NEW_PLAN_END==="
            )
        return "这是对新计划的解释。"

    monkeypatch.setattr(main_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(main_module, "create_llm", lambda: object())
    monkeypatch.setattr(main_module, "_invoke_interactive_turn", fake_turn)
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(
        commands_module, "resolve_project_path", lambda value: path,
    )

    result = main_module.run_interactive_session(
        analysis="资料分析",
        plan="旧计划",
        material_context="资料证据",
        learning_goal="旧目标",
        strategy="window",
        material_dir="materials",
    )

    restored = load_learning_state(path)
    assert result == ("新计划", "新目标")
    assert restored.plan_markdown == "新计划"
    assert restored.learning_goal == "新目标"
    assert restored.initial_plan_markdown == "旧计划"
    assert restored.initial_learning_goal == "旧目标"
    assert restored.context_revision == 2
    assert restored.tasks_need_review is True
    assert restored.tasks[0].activity_status == "completed"
    assert len(restored.tasks) == 2

    # 第二次普通问答必须已经使用新目标、新计划，而不只是文件保存正确。
    prompt = str(requests[1][0].content)
    assert "当前学习目标：新目标" in prompt
    assert "当前学习计划：\n新计划" in prompt

def show_state_change(stage, before, after):
    """打印实际对象的字段变化，不参与业务更新或测试判定。"""
    print(f"\n{'=' * 16} {stage} {'=' * 16}")

    fields = [
        ("initial_learning_goal", "初始目标"),
        ("initial_plan_markdown", "初始计划"),
        ("learning_goal", "当前目标"),
        ("plan_markdown", "当前计划"),
        ("context_revision", "修订号"),
        ("tasks_need_review", "任务待核对"),
    ]

    for field, label in fields:
        # getattr 根据字段名读取属性，避免为每个字段重复写打印代码。
        old = getattr(before, field)
        new = getattr(after, field)
        marker = "保留" if old == new else "变化"
        print(f"[{marker}] {label}：{old!r} → {new!r}")

    # 按 ID 对照，而不是按列表位置，避免把不同任务错配。
    old_tasks = {task.task_id: task for task in before.tasks}
    new_tasks = {task.task_id: task for task in after.tasks}

    print("\n任务对照：")
    for task_id, task in new_tasks.items():
        old = old_tasks.get(task_id)
        if old is None:
            print(f"[新增] {task_id} / {task.title}")
            continue

        marker = "保留" if old == task else "变化"
        print(f"[{marker}] {task_id} / {task.title}")
        print(
            f"  活动：{old.activity_status}"
            f" → {task.activity_status}"
        )
        print(
            f"  理解：{old.understanding_status}"
            f" → {task.understanding_status}"
        )
        print(
            f"  反馈：{old.latest_feedback!r}"
            f" → {task.latest_feedback!r}"
        )

    for task_id in old_tasks.keys() - new_tasks.keys():
        print(f"[移除] {task_id}")

def test_visualize_context_update_and_restore(tmp_path):
    """观察真实状态更新与保存恢复；不调用 LLM。"""
    original = LearningState(
        learning_goal="理解文章中的两种观点",
        plan_markdown="第1天阅读文章，第2天比较观点。",
        tasks=(
            LearningTask(
                task_id="task-001",
                title="阅读文章",
                activity_status="completed",
                understanding_status="self_reported_unclear",
                latest_feedback="已经读完，但两种观点的区别没理解。",
                feedback_source="user_report",
            ),
        ),
    )

    show_state_change("阶段1：建立初始状态", original, original)
    assert original.initial_plan_markdown == original.plan_markdown
    assert original.context_revision == 1

    # 本测试直接提供新目标/计划，隔离验证状态机制。
    # 它不证明模型能够正确理解“调整计划”的自然语言请求。
    revised = update_learning_context(
        original,
        learning_goal="重点分清两种观点的区别",
        plan_markdown="保留已完成阅读，第2天用半小时比较两种观点。",
    )

    show_state_change("阶段2：调整目标和计划", original, revised)

    assert revised.initial_learning_goal == original.learning_goal
    assert revised.initial_plan_markdown == original.plan_markdown
    assert revised.context_revision == 2
    assert revised.tasks_need_review is True
    assert revised.tasks == original.tasks

    # 重复提交相同内容，不应产生一个虚假的新版本。
    unchanged = update_learning_context(
        revised,
        plan_markdown=revised.plan_markdown,
    )
    show_state_change("阶段3：重复相同计划", revised, unchanged)

    assert unchanged is revised
    assert unchanged.context_revision == 2

    # 这是本轮重要的关联修复：
    # 登记任务只改变任务表，不能重置初始记录或修订号。
    registered, message = commands_module.handle_learning_command(
        revised,
        '/task {"task_id":"task-002","title":"用自己的话比较两种观点"}',
    )
    print(f"\n命令结果：{message}")
    show_state_change("阶段4：登记新任务", revised, registered)

    assert registered.initial_plan_markdown == original.plan_markdown
    assert registered.context_revision == 2
    assert registered.tasks_need_review is True
    assert registered.tasks[0] == original.tasks[0]
    assert len(registered.tasks) == 2

    path = tmp_path / "visual-state.json"
    save_learning_state(registered, path)
    restored = load_learning_state(path)

    saved_data = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n保存格式：{saved_data['schema_version']}")
    show_state_change("阶段5：保存后重新读取", registered, restored)

    assert saved_data["schema_version"] == "learning-state-v5"
    assert restored == registered

    print("\n全部检查通过：初始版本保留、当前版本更新、任务未重置、恢复一致。")

def test_visualize_context_update_and_restore(tmp_path):
    """观察真实状态更新与保存恢复；不调用 LLM。"""
    original = LearningState(
        learning_goal="理解文章中的两种观点",
        plan_markdown="第1天阅读文章，第2天比较观点。",
        tasks=(
            LearningTask(
                task_id="task-001",
                title="阅读文章",
                activity_status="completed",
                understanding_status="self_reported_unclear",
                latest_feedback="已经读完，但两种观点的区别没理解。",
                feedback_source="user_report",
            ),
        ),
    )

    show_state_change("阶段1：建立初始状态", original, original)
    assert original.initial_plan_markdown == original.plan_markdown
    assert original.context_revision == 1

    # 本测试直接提供新目标/计划，隔离验证状态机制。
    # 它不证明模型能够正确理解“调整计划”的自然语言请求。
    revised = update_learning_context(
        original,
        learning_goal="重点分清两种观点的区别",
        plan_markdown="保留已完成阅读，第2天用半小时比较两种观点。",
    )

    show_state_change("阶段2：调整目标和计划", original, revised)

    assert revised.initial_learning_goal == original.learning_goal
    assert revised.initial_plan_markdown == original.plan_markdown
    assert revised.context_revision == 2
    assert revised.tasks_need_review is True
    assert revised.tasks == original.tasks

    # 重复提交相同内容，不应产生一个虚假的新版本。
    unchanged = update_learning_context(
        revised,
        plan_markdown=revised.plan_markdown,
    )
    show_state_change("阶段3：重复相同计划", revised, unchanged)

    assert unchanged is revised
    assert unchanged.context_revision == 2

    # 这是本轮重要的关联修复：
    # 登记任务只改变任务表，不能重置初始记录或修订号。
    registered, message = commands_module.handle_learning_command(
        revised,
        '/task {"task_id":"task-002","title":"用自己的话比较两种观点"}',
    )
    print(f"\n命令结果：{message}")
    show_state_change("阶段4：登记新任务", revised, registered)

    assert registered.initial_plan_markdown == original.plan_markdown
    assert registered.context_revision == 2
    assert registered.tasks_need_review is True
    assert registered.tasks[0] == original.tasks[0]
    assert len(registered.tasks) == 2

    path = tmp_path / "visual-state.json"
    save_learning_state(registered, path)
    restored = load_learning_state(path)

    saved_data = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n保存格式：{saved_data['schema_version']}")
    show_state_change("阶段5：保存后重新读取", registered, restored)

    assert saved_data["schema_version"] == "learning-state-v5"
    assert restored == registered

    print("\n全部检查通过：初始版本保留、当前版本更新、任务未重置、恢复一致。")
