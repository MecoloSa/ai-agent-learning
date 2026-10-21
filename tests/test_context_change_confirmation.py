"""聊天目标/计划变更的确认门控测试。

使用假模型和假输入，验证取消、确认、非法及无变化路径。
不调用在线服务，不写入真实学习快照。
"""
import builtins
import json

import pytest

import ai_agent.main as main_module
import ai_agent.persistence.memory as memory_module
from ai_agent.learning.learning_state import LearningState, LearningTask


@pytest.mark.parametrize("answer", ["y", "n", ""])
def test_review_preserves_old_state_and_tasks(monkeypatch, answer):
    original = LearningState(
        learning_goal="旧目标",
        plan_markdown="旧计划",
        tasks=(LearningTask(
            task_id="task-001",
            title="阅读文章",
            activity_status="completed",
            understanding_status="self_reported_unclear",
            latest_feedback="读完但还不理解。",
            feedback_source="user_report",
        ),),
    )
    monkeypatch.setattr(
        builtins, "input", lambda _: answer
    )

    candidate, decision = main_module._review_context_change(
        original,
        learning_goal="新目标",
        plan_markdown="新计划",
    )

    assert original.context_revision == 1
    assert original.plan_markdown == "旧计划"

    if answer == "y":
        assert decision == "confirmed"
        assert candidate.context_revision == 2
        assert candidate.initial_plan_markdown == "旧计划"
        assert candidate.tasks == original.tasks
        assert candidate.tasks_need_review is True
    else:
        assert decision == "cancelled"
        assert candidate is original


@pytest.mark.parametrize("text", [
    "===NEW_PLAN_START===计划",
    "===NEW_PLAN_END===计划===NEW_PLAN_START===",
    (
        "===NEW_PLAN_START===甲===NEW_PLAN_END==="
        "===NEW_PLAN_START===乙===NEW_PLAN_END==="
    ),
    (
        "===NEW_PLAN_START==="
        "===NEW_GOAL_START===目标===NEW_GOAL_END==="
        "===NEW_PLAN_END==="
    ),
])
def test_ambiguous_plan_markers_are_rejected(text):
    with pytest.raises(ValueError):
        main_module._extract_updated_plan(text)


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
@pytest.mark.parametrize(
    "answer, goal, plan, status",
    [
        ("y", "新目标", "新计划", "confirmed"),
        ("n", "新目标", "新计划", "cancelled"),
        ("", "新目标", "新计划", "cancelled"),
        (None, "", "新计划", "invalid"),
        (None, "旧目标", "旧计划", "unchanged"),
    ],
)
def test_chat_confirmation_controls_state_memory_and_save(
    monkeypatch, tmp_path, strategy,
    answer, goal, plan, status,
):
    inputs = ["请修改目标和计划"]
    if answer is not None:
        inputs.append(answer)
    inputs.extend(["解释当前计划", "exit"])
    answers = iter(inputs)

    requests = []
    saved = []
    compression_observations = []

    def fake_turn(**kwargs):
        requests.append(list(kwargs["request_messages"]))
        if len(requests) == 1:
            return (
                f"===NEW_GOAL_START==={goal}===NEW_GOAL_END==="
                f"===NEW_PLAN_START==={plan}===NEW_PLAN_END==="
            )
        return "这是对当前生效计划的解释。"

    def save(state):
        saved.append(state)
        return tmp_path / "unused.json"

    def observe_compression(self):
        # 这里只观察压缩的调用时机，不触发真实摘要模型。
        compression_observations.append((
            self.system_prompt,
            str(self.messages[-1].content),
        ))

    monkeypatch.setattr(
        builtins, "input", lambda _: next(answers)
    )
    monkeypatch.setattr(
        main_module, "needs_rag", lambda _: False
    )
    monkeypatch.setattr(
        main_module, "create_llm", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        main_module,
        "create_llm_with_tools",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        main_module, "_invoke_interactive_turn", fake_turn
    )
    monkeypatch.setattr(
        main_module, "save_current_learning_state", lambda state, _path: (save)(state)
    )
    monkeypatch.setattr(
        memory_module.ConversationMemory,
        "_maybe_compress",
        observe_compression,
    )

    result = main_module.run_interactive_session(
        analysis="资料分析",
        plan="旧计划",
        material_context="测试资料",
        learning_goal="旧目标",
        strategy=strategy,
        material_dir=str(tmp_path),
    )

    confirmed = status == "confirmed"
    expected_goal = "新目标" if confirmed else "旧目标"
    expected_plan = "新计划" if confirmed else "旧计划"
    expected_revision = 2 if confirmed else 1

    assert result == (expected_plan, expected_goal)
    # 两轮完整问答都会保存；取消只禁止采用候选，不禁止保存讨论记录。
    assert len(saved) == 2
    assert all(item.learning_goal == expected_goal for item in saved)
    assert all(item.plan_markdown == expected_plan for item in saved)
    assert all(item.context_revision == expected_revision for item in saved)
    assert f"程序处理结果：{status}" in saved[0].conversation_memory.messages[-1].content
    assert len(requests) == 2

    second_prompt = str(requests[1][0].content)
    assert f"当前学习目标：{expected_goal}" in second_prompt
    assert f"当前学习计划：\n{expected_plan}" in second_prompt

    history = [
        str(message.content)
        for message in requests[1]
        if message.type == "ai"
    ]
    assert any(
        f"程序处理结果：{status}" in text
        for text in history
    )

    state_messages = [
        json.loads(str(message.content))
        for message in requests[1]
        if message.type == "human"
        and str(message.content).startswith("{")
    ]
    state_messages = [
        item for item in state_messages
        if item.get("context_type") == "current_learning_state"
    ]
    assert len(state_messages) == 1
    assert state_messages[-1]["context_revision"] == expected_revision

    if strategy == "summary":
        # 首次AI记录就必须包含最终处理结果，且system已同步。
        prompt, text = compression_observations[0]
        assert f"程序处理结果：{status}" in text
        assert f"当前学习目标：{expected_goal}" in prompt
