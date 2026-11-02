"""时间规则跨入口接线与新案例契约；不以提示词存在证明估时可靠。"""
from pathlib import Path

import pytest


def test_interactive_entry_receives_time_policy_without_model_call(monkeypatch):
    import ai_agent.main as app
    probe = "<<learning-time-policy-probe>>"
    monkeypatch.setattr(app, "LEARNING_TIME_RULES", probe)
    prompt = app._build_interactive_system_prompt("分析", "旧计划", "", "学习目标",
        retrieval_available=False, on_demand_eligible=False)
    assert prompt.count(probe) == 1
    assert "旧计划" in prompt and "学习目标" in prompt


@pytest.mark.parametrize("mode", ["pipeline", "agent"])
def test_initial_plan_entry_receives_same_time_policy(monkeypatch, mode):
    import ai_agent.runtime.agents as agents
    probe = "<<learning-time-policy-probe>>"
    monkeypatch.setattr(agents, "LEARNING_TIME_RULES", probe)
    monkeypatch.setattr(agents, "needs_rag", lambda _: False)
    monkeypatch.setattr(agents, "create_llm", lambda: object())
    captured = []
    def invoke(llm, messages, *, stage):
        captured.extend(messages)
        return "第1天：阅读概念并口头比较，能说明区别后进入下一阶段。"
    monkeypatch.setattr(agents, "_invoke_llm", invoke)
    agents.create_study_plan("分析", "[source=note.md]原文", "理解概念", 3,
                             material_dir="materials", mode=mode)
    assert captured[0].content.count(probe) == 1
    assert "计划天数：3" in captured[1].content


def test_time_cases_use_explicit_user_constraints_and_keep_existing_cases():
    from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases
    root = Path(__file__).parent / "fixtures/evaluation"
    original = root / "feedback_plan_cases.jsonl"
    before = original.read_bytes()
    cases = load_feedback_plan_cases(root / "feedback_plan_time_cases.jsonl")
    assert [c.case_id for c in cases] == ["feedback-plan-dev-005", "feedback-plan-dev-006"]
    assert all(c.split == "dev" and c.inputs[-1] == "exit" for c in cases)
    assert all([check.category for check in c.checks] == ["control", "quality"] for c in cases)
    assert all(c.checks[0].rule == "state_unchanged" for c in cases)
    assert len(load_feedback_plan_cases(original)) == 3
    assert original.read_bytes() == before
