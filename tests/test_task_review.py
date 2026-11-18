from types import SimpleNamespace

import pytest

from ai_agent.runtime.task_review import ReviewRequest, review_task


@pytest.mark.parametrize(
    ("phase", "reply", "decision", "approved"),
    [
        ("submission", '{"decision":"pass","feedback":""}', "pass", True),
        ("blocked", '{"decision":"pass","feedback":""}', "pass", False),
        ("submission", '{"decision":"revise","feedback":"补读第85页"}', "revise", False),
        ("submission", '{"decision":"revise","feedback":""}', "unverified", False),
        ("submission", "", "unverified", False),
    ],
)
def test_review_verdict_belongs_to_the_submitted_revision(phase, reply, decision, approved):
    request = ReviewRequest("notes", 2, phase, "核对第85页", ("取得正文",), "候选",
                            ({"source": "notes.pdf", "body_pages": [84]},))
    model = SimpleNamespace(invoke=lambda _: SimpleNamespace(content=reply))
    result = review_task(request, model=model)

    assert result.decision == decision
    assert result.approves("notes", 2) is approved
    assert not result.approves("notes", 3)
    assert not result.approves("other", 2)
