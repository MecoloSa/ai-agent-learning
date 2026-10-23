"""计划修订的结构提醒：诊断不修改候选，也不代替用户确认。

只模拟输入，不调用模型或保存真实存档；接线未应用时明确跳过。
"""
import builtins

import pytest

import ai_agent.main as main
from ai_agent.learning.learning_state import LearningState


pytestmark = pytest.mark.skipif(
    not hasattr(main, "inspect_plan_format"),
    reason="修订结构提醒尚未接线，不代表该功能已通过",
)


@pytest.mark.parametrize("answer", ["y", "n"])
def test_warning_preserves_candidate_and_confirmation(monkeypatch, capsys, answer):
    original = LearningState(learning_goal="理解文章", plan_markdown="旧计划")
    plan = "任务1：阅读\n学习内容：文章"
    seen = []

    def inspect(text):
        seen.append(text)
        return {"status": "warnings", "task_count": 1, "warnings": ["测试缺项提醒"]}

    monkeypatch.setattr(main, "inspect_plan_format", inspect)
    monkeypatch.setattr(builtins, "input", lambda _: answer)
    updated, decision = main._review_context_change(original, plan_markdown=plan)
    assert seen == [plan]
    assert "测试缺项提醒" in capsys.readouterr().out
    assert decision == ("confirmed" if answer == "y" else "cancelled")
    assert updated.plan_markdown == (plan if answer == "y" else "旧计划")
    assert original.plan_markdown == "旧计划"


def test_goal_only_and_unchanged_do_not_inspect_plan(monkeypatch):
    original = LearningState(learning_goal="旧目标", plan_markdown="旧计划")

    def forbidden(_):
        pytest.fail("未改变计划时不应重复检查")

    monkeypatch.setattr(main, "inspect_plan_format", forbidden)
    monkeypatch.setattr(builtins, "input", lambda _: "y")
    unchanged, decision = main._review_context_change(original, plan_markdown="旧计划")
    assert unchanged is original
    assert decision == "unchanged"
    changed, decision = main._review_context_change(original, learning_goal="新目标")
    assert changed.plan_markdown == "旧计划"
    assert decision == "confirmed"


def test_unrecognized_layout_can_still_be_confirmed(monkeypatch, capsys):
    original = LearningState(learning_goal="理解文章", plan_markdown="旧计划")
    plan = "| 活动 | 行动 |\n| --- | --- |\n| 阅读 | 比较论据 |"
    monkeypatch.setattr(builtins, "input", lambda _: "y")
    updated, decision = main._review_context_change(original, plan_markdown=plan)
    assert "布局" in capsys.readouterr().out
    assert decision == "confirmed"
    assert updated.plan_markdown == plan
