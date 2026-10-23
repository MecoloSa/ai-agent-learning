"""计划确认的按需全文预览。

只模拟终端输入，不请求模型、不写真实存档。
方案尚未应用时跳过；应用后检查预览不授权、最终输入原因和默认展示。
"""
import builtins
import inspect

import pytest

import ai_agent.main as main
from ai_agent.learning.confirmation import read_confirmation
from ai_agent.learning.learning_state import LearningState


pytestmark = pytest.mark.skipif(
    "on_preview" not in inspect.signature(read_confirmation).parameters,
    reason="按需预览方案尚未应用，不代表新行为已验收",
)


@pytest.mark.parametrize("answers, expected, reason, views", [
    (["y"], True, "user_confirmed", 0),
    (["v", "y"], True, "user_confirmed", 1),
    (["v", "v", "n"], False, "user_declined", 2),
    (["v", ""], False, "user_not_confirmed", 1),
    (["v", None], False, "input_ended", 1),
])
def test_preview_does_not_replace_final_confirmation(monkeypatch, answers, expected, reason, views):
    queue = iter(answers)
    seen = []
    trace = {}

    def read(_):
        answer = next(queue)
        if answer is None:
            raise EOFError
        return answer

    monkeypatch.setattr(builtins, "input", read)
    result = read_confirmation("确认", trace=trace, on_preview=lambda: seen.append("view"))
    assert result is expected
    assert trace == {"reason": reason}
    assert len(seen) == views


def test_v_without_preview_keeps_original_semantics(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda _: "v")
    trace = {}
    assert read_confirmation("确认", trace=trace) is False
    assert trace == {"reason": "user_not_confirmed"}


@pytest.mark.parametrize("answer", ["y", "n"])
def test_plan_preview_shows_exact_candidate_only_on_request(monkeypatch, capsys, answer):
    original = LearningState(learning_goal="理解文章", plan_markdown="旧计划")
    candidate_text = "# 候选全文\n先概括论点，再核对证据。"
    queue = iter(["v", answer])
    calls = 0

    def read(_):
        nonlocal calls
        output = capsys.readouterr().out
        if calls == 0:
            assert candidate_text not in output
            assert "计划差异" not in output
        else:
            assert candidate_text in output
            assert original.plan_markdown == "旧计划"
            assert original.context_revision == 1
        calls += 1
        return next(queue)

    monkeypatch.setattr(builtins, "input", read)
    trace = {}
    updated, decision = main._review_context_change(
        original, plan_markdown=candidate_text, confirmation_trace=trace,
    )
    assert decision == ("confirmed" if answer == "y" else "cancelled")
    assert updated.plan_markdown == (candidate_text if answer == "y" else "旧计划")
    assert updated.tasks == original.tasks
    assert updated.feedback_records == original.feedback_records
    assert original.context_revision == 1
