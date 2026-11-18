"""提示词策略：余额只在临近边界出现，计数和权限不由提示承担。"""
import pytest
from ai_agent.runtime.prompt_rules import budget_notice


@pytest.mark.parametrize("remaining", [3, 8, 12])
def test_normal_budget_is_not_announced(remaining):
    assert budget_notice(remaining) == ""


@pytest.mark.parametrize("remaining", [0, 1, 2])
def test_near_exhaustion_has_current_balance(remaining):
    notice = budget_notice(remaining)
    assert f"还可执行{remaining}次" in notice
    assert "非资料证据" in notice
    assert "remaining_by_tool" not in notice


def test_last_model_decision_is_announced_even_with_tool_budget_left():
    assert "最后一次模型决策" in budget_notice(8, 1)
    assert budget_notice(8, 2) == ""
