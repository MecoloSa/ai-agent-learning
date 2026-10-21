"""验证普通问答能接收已有学习状态。

使用假问答捕获真实组装的消息，不调用模型、不保存真实快照。
这里只验证状态接线；CLI恢复、资料绑定与真实回答质量留给后续验收。
"""

import builtins
import json

import pytest

import ai_agent.main as main_module
import ai_agent.persistence.memory as memory_module
from ai_agent.learning.learning_state import LearningState, LearningTask


def make_restored_state():
    return LearningState(
        learning_goal="当前目标：理解ReAct",
        plan_markdown="当前计划：解释工具结果与终止条件。",
        initial_learning_goal="最初目标",
        initial_plan_markdown="最初计划",
        context_revision=4,
        tasks_need_review=True,
        tasks=(LearningTask(
            task_id="task-001",
            title="理解ReAct",
            activity_status="in_progress",
            understanding_status="self_reported_unclear",
            latest_feedback="不理解循环何时结束。",
            feedback_source="user_report",
        ),),
    )


@pytest.fixture
def chat_spy(monkeypatch, tmp_path):
    requests, saved = [], []

    def fake_turn(**kwargs):
        requests.append(kwargs["request_messages"])
        return "这里只解释，不修改计划或目标。"

    def fake_save(state):
        saved.append(state)
        return tmp_path / "not-written.json"

    monkeypatch.setattr(main_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(
        main_module, "create_llm", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        main_module,
        "create_llm_with_tools",
        lambda *a, **kw: object(),
    )
    monkeypatch.setattr(
        main_module, "_invoke_interactive_turn", fake_turn
    )
    monkeypatch.setattr(
        main_module, "save_current_learning_state", lambda state, _path: (fake_save)(state)
    )

    # 只测状态初始化与传递，不让summary调用真实摘要模型。
    # 摘要算法本身仍由已有test_memory.py覆盖。
    monkeypatch.setattr(
        memory_module.ConversationMemory,
        "_maybe_compress",
        lambda self: None,
    )
    return requests, saved


def extract_state(messages):
    states = []
    for message in messages:
        if message.type != "human":
            continue

        try:
            data = json.loads(str(message.content))
        except ValueError:
            continue

        if (
            isinstance(data, dict)
            and data.get("context_type") == "current_learning_state"
        ):
            states.append(data)

    # 临时状态每轮只注入一次，不能混入多个旧版本。
    assert len(states) == 1
    return states[0]


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
@pytest.mark.parametrize("restoring", [False, True])
def test_initial_state_is_used_without_reset(
    monkeypatch, tmp_path, chat_spy, strategy, restoring,
):
    requests, saved = chat_spy
    inputs = iter(["只说明当前记录，不修改。", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    state = make_restored_state()
    optional = {"initial_state": state} if restoring else {}

    result = main_module.run_interactive_session(
        analysis="本测试提供的分析，不代表恢复了历史证据。",
        plan="入口默认计划",
        material_context="",
        learning_goal="入口默认目标",
        strategy=strategy,
        material_dir=str(tmp_path),
        **optional,
    )

    data = extract_state(requests[0])
    prompt = str(requests[0][0].content)
    expected_goal = (
        state.learning_goal if restoring else "入口默认目标"
    )
    expected_plan = (
        state.plan_markdown if restoring else "入口默认计划"
    )

    assert result == (expected_plan, expected_goal)
    assert f"当前学习目标：{expected_goal}" in prompt
    assert f"当前学习计划：\n{expected_plan}" in prompt
    assert data["context_revision"] == (4 if restoring else 1)
    assert data["tasks_need_review"] is restoring
    assert data["tasks"] == (
        [task.model_dump() for task in state.tasks]
        if restoring else []
    )
    # 仅问答不改目标/任务，但新增对话背景需要保存。
    assert len(saved) == 1
    assert saved[0].context_revision == (4 if restoring else 1)
    assert len(saved[0].conversation_memory.messages) == 2


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
def test_feedback_updates_restored_task_without_reset(
    monkeypatch, tmp_path, chat_spy, strategy,
):
    requests, saved = chat_spy
    original = make_restored_state()
    inputs = iter([
        '/feedback {"task_id":"task-001","text":"现在理解了。",'
        '"understanding_status":"self_reported_clear"}',
        "现在记录的理解情况是什么？",
        "exit",
    ])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    main_module.run_interactive_session(
        analysis="",
        plan="入口默认计划",
        material_context="",
        learning_goal="入口默认目标",
        strategy=strategy,
        material_dir=str(tmp_path),
        initial_state=original,
    )

    data = extract_state(requests[0])
    assert (
        data["tasks"][0]["understanding_status"]
        == "self_reported_clear"
    )
    assert data["tasks"][0]["activity_status"] == "in_progress"
    assert len(saved) == 2

    updated = saved[0]
    assert updated.context_revision == 4
    assert updated.tasks_need_review is True
    assert (
        updated.initial_learning_goal
        == original.initial_learning_goal
    )
    assert (
        updated.initial_plan_markdown
        == original.initial_plan_markdown
    )
    assert updated.plan_markdown == original.plan_markdown

    # 新反馈产生新状态，原始恢复对象不能被原地污染。
    assert (
        original.tasks[0].understanding_status
        == "self_reported_unclear"
    )
