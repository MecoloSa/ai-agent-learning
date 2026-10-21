"""待确认反馈契约测试；不调用模型，不证明语义识别准确。

业务模块由用户按指导新增后运行。重点验证来源边界、任务定位和无写入。
"""

import pytest
from pydantic import ValidationError

from ai_agent.learning.feedback_proposals import FeedbackProposalArgs, prepare_feedback_proposal
from ai_agent.learning.learning_state import LearningState, LearningTask


@pytest.fixture
def state():
    return LearningState(
        learning_goal="理解ReAct", plan_markdown="阅读并解释",
        tasks=(LearningTask(task_id="t1", title="阅读文章"),),
    )


def args(**changes):
    return FeedbackProposalArgs.model_validate({
        "scope": "task", "task_id": "t1",
        "activity_status": "in_progress", "understanding_status": None,
        **changes,
    })


def test_proposal_is_pending_and_preserves_program_source(state):
    before = state.model_dump_json()
    proposal = prepare_feedback_proposal(
        state, args(), user_message="文章只读了一半，还需要继续。",
    )
    assert proposal.status == "pending"
    assert proposal.user_message == "文章只读了一半，还需要继续。"
    assert proposal.context_revision == state.context_revision
    assert proposal.request.activity_status == "in_progress"
    assert state.model_dump_json() == before


def test_time_only_feedback_does_not_require_task_or_category(state):
    proposal = prepare_feedback_proposal(
        state,
        args(scope="current_context", task_id=None,
             activity_status=None),
        user_message="今天只剩20分钟，先不要改计划。",
    )
    assert proposal.request.task_id is None
    assert proposal.request.activity_status is None
    assert proposal.request.understanding_status is None
    assert state.context_revision == 1


@pytest.mark.parametrize("changes", [
    {"task_id": None},
    {"scope": "current_context"},
    {"scope": "current_context", "task_id": None},
    {"quote": " "},
    {"learning_goal": "模型越权修改目标"},
    {"activity_status": "mastered"},
])
def test_invalid_proposal_shape_is_rejected(changes):
    with pytest.raises(ValidationError):
        args(**changes)


def test_unknown_task_is_not_created(state):
    with pytest.raises(ValueError, match="任务不存在"):
        prepare_feedback_proposal(state, args(task_id="missing"), user_message="只读了一半")
    assert len(state.tasks) == 1


def test_model_cannot_supply_quoted_user_text():
    with pytest.raises(ValidationError):
        args(quote="模型伪造的原文")


def test_empty_user_message_is_rejected(state):
    with pytest.raises(ValueError):
        prepare_feedback_proposal(state, args(), user_message=" ")


def test_quoted_hypothesis_still_requires_semantic_review(state):
    # 来源真实也不能证明是假设还是用户事实，候选仍须语义核对与确认。
    proposal = prepare_feedback_proposal(
        state, args(), user_message="假设我只读了一半，你会怎么建议？",
    )
    assert proposal.status == "pending"
    assert state.tasks[0].activity_status == "not_started"
