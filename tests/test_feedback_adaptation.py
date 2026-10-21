"""反馈调整的规则接线与确认边界测试。

假模型预先返回候选，只证明程序如何处理它，不证明真实模型建议合理。
固定业务状态与演示资料可用于后续少量在线人工验收。
"""

import builtins
import json
from pathlib import Path

import pytest

import ai_agent.main as main
from ai_agent.persistence.memory import ConversationMemory
from ai_agent.persistence.state_store import load_learning_state


FIXTURE = (
    Path(__file__).parent / "fixtures" / "evaluation"
    / "feedback_adaptation" / "state.json"
)


def test_business_fixture_separates_reading_and_understanding():
    state = load_learning_state(FIXTURE)
    task = state.tasks[0]
    assert task.activity_status == "completed"
    assert task.understanding_status == "self_reported_unclear"
    assert task.feedback_source == "user_report"
    assert state.context_revision == 1
    assert state.tasks_need_review is False
    assert state.material_binding is None
    assert state.conversation_memory is None


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
@pytest.mark.parametrize("decision", ["y", "n"])
def test_candidate_confirmation_keeps_feedback_and_effective_context(
    monkeypatch, tmp_path, strategy, decision,
):
    state = load_learning_state(FIXTURE)
    source_bytes = FIXTURE.read_bytes()
    current = tmp_path / "current.json"
    retained = state.plan_markdown.splitlines()[2]
    candidate = (
        "# 调整后的学习计划\n\n" + retained + "\n"
        "任务2：对比两个停止例子，10分钟。完成标志：说明停止决定来自哪里。\n"
        "任务3：口头复述区别并自检，10分钟。完成标志：不看资料区分两类情况。\n"
        "后续安排合计20分钟，不计已经完成的阅读。"
    )
    inputs = iter(["请调整任务2、3，后续总共20分钟", decision, "解释当前安排", "exit"])
    requests = []

    def fake_input(_):
        value = next(inputs)
        if value == decision:
            assert not current.exists(), "确认决定前不能保存候选"
        return value

    def fake_turn(**kwargs):
        requests.append(kwargs["request_messages"])
        if len(requests) == 1:
            return f"===NEW_PLAN_START==={candidate}===NEW_PLAN_END==="
        return "解释当前安排，不修改状态。"

    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr(main, "new_current_state_path", lambda: current)
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(main, "_invoke_interactive_turn", fake_turn)
    monkeypatch.setattr(ConversationMemory, "_maybe_compress", lambda self: None)

    main.run_interactive_session(
        analysis="", plan=state.plan_markdown, material_context="",
        learning_goal=state.learning_goal, strategy=strategy,
        material_dir=str(tmp_path), initial_state=state,
    )
    saved = load_learning_state(current)
    confirmed = decision == "y"
    assert saved.plan_markdown == (candidate if confirmed else state.plan_markdown)
    assert saved.learning_goal == state.learning_goal
    assert saved.context_revision == (2 if confirmed else 1)
    assert saved.tasks == state.tasks
    assert saved.tasks_need_review is confirmed
    assert saved.initial_plan_markdown == state.plan_markdown
    assert FIXTURE.read_bytes() == source_bytes
    assert len(requests) == 2
    assert saved.plan_markdown in requests[1][0].content

    data = [
        json.loads(message.content)
        for message in requests[0]
        if message.type == "human" and str(message.content).startswith("{")
    ]
    task = next(
        item for item in data if item.get("context_type") == "current_learning_state"
    )["tasks"][0]
    assert task["activity_status"] == "completed"
    assert task["understanding_status"] == "self_reported_unclear"
