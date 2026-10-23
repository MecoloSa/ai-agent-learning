"""更正正文与来源原话分离的回归，不调用在线模型。

测试确认、取消、持久化和上下文接线；不会判断任意自然语言是否保留了所有条件。
模型生成的完整正文仍须由用户核对，schema合法不代表语义正确。
"""
import builtins
import json

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from ai_agent.learning.feedback_proposals import FeedbackProposalArgs, prepare_feedback_proposal
from ai_agent.learning.feedback_runtime import LearningActionInbox, commit_feedback_proposal, review_feedback_proposal
from ai_agent.learning.feedback_history import FeedbackHistoryArgs, read_feedback_history
from ai_agent.learning.learning_context import build_learning_state_message
from ai_agent.learning.learning_state import FeedbackRecord, LearningState, learning_today
from ai_agent.persistence.state_store import load_learning_state, save_learning_state


RAW = "把刚才记录的可用时间改为40分钟，其他要求不变，先不要修改计划。"
CONTENT = "今天有40分钟，只读不写代码。"


def original_state():
    return LearningState(
        learning_goal="区分两类停止",
        plan_markdown="比较10分钟，复述10分钟。",
        feedback_records=(FeedbackRecord(
            record_id="old", scope="current_context", task_id=None,
            quote="今天只有20分钟，只读不写代码。",
            text="今天只有20分钟，只读不写代码。",
            context_revision=1, recorded_on=learning_today(),
            activity_status=None, understanding_status=None,
        ),),
    )


def request(**changes):
    return FeedbackProposalArgs.model_validate({
        "scope": "current_context", "task_id": None,
        "activity_status": None, "understanding_status": None,
        "supersedes_record_id": "old", "replacement_text": CONTENT,
        **changes,
    })


@pytest.mark.parametrize("changes", [
    {"replacement_text": None}, {"replacement_text": ""},
    {"replacement_text": " \n "}, {"replacement_text": "字" * 4001},
    {"replacement_text": 40}, {"supersedes_record_id": None},
    {"scope": "task", "task_id": "read-001"},
])
def test_invalid_or_unpaired_replacement_content_is_rejected(changes):
    with pytest.raises(ValidationError):
        request(**changes)


def test_normal_feedback_still_uses_original_message():
    old = original_state()
    args = request(supersedes_record_id=None, replacement_text=None)
    pending = prepare_feedback_proposal(old, args, user_message="今天先比较例子。")
    updated = commit_feedback_proposal(old, pending)
    assert updated.feedback_records[-1].text == updated.feedback_records[-1].quote == "今天先比较例子。"


def test_revision_keeps_raw_source_and_adopts_confirmed_complete_content(tmp_path):
    old = original_state()
    before = old.model_dump()
    pending = prepare_feedback_proposal(old, request(), user_message=RAW)
    updated = commit_feedback_proposal(old, pending)
    new = updated.feedback_records[-1]
    assert old.model_dump() == before
    assert new.quote == RAW and new.text == CONTENT
    assert new.supersedes_record_id == "old"
    assert updated.feedback_records[0] == old.feedback_records[0]
    assert updated.plan_markdown == old.plan_markdown
    assert updated.tasks == old.tasks
    assert updated.context_revision == old.context_revision

    # 复用现有v7的text/quote和更正关系，不在磁盘引入模型临时候选参数。
    path = save_learning_state(updated, tmp_path / "state.json")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["schema_version"] == "learning-state-v7"
    assert "replacement_text" not in envelope["state"]["feedback_records"][-1]
    restored = load_learning_state(path)
    assert restored == updated
    data = json.loads(build_learning_state_message(restored).content)
    record = data["current_context_feedback"][0]
    assert record["text"] == CONTENT and record["quote"] == RAW
    assert "确认采用的反馈正文" in data["feedback_usage_note"]
    history = json.loads(read_feedback_history(restored, FeedbackHistoryArgs(record_id=new.record_id)))["data"]
    assert history["records"][0]["quote"] == RAW
    assert history["records"][0]["text"] == CONTENT
    assert "确认采用的反馈正文" in history["note"]


@pytest.mark.parametrize("answer,decision", [("y", "confirmed"), ("n", "cancelled"), ("", "cancelled")])
def test_confirmation_displays_old_raw_and_candidate_before_decision(monkeypatch, capsys, answer, decision):
    old = original_state()
    pending = prepare_feedback_proposal(old, request(), user_message=RAW)

    def confirm(_):
        output = capsys.readouterr().out
        assert old.feedback_records[0].text in output
        assert RAW in output and CONTENT in output
        assert "更正后完整正文" in output
        assert "其余条件也将从当前读取视图退出" in output
        return answer

    monkeypatch.setattr(builtins, "input", confirm)
    updated, actual = review_feedback_proposal(old, pending)
    assert actual == decision
    if decision != "confirmed":
        assert updated is old
    else:
        assert updated.feedback_records[-1].text == CONTENT


def test_followup_source_is_program_bound_not_model_written():
    old = original_state()
    inbox = LearningActionInbox()
    inbox.start_turn(old, "请记录刚才的更正。", history_messages=[HumanMessage(content=RAW)])
    args = request(source_id="previous_user_1")
    result = json.loads(inbox.tool.invoke(args.model_dump()))
    assert result["data"]["status"] == "pending"
    assert inbox.pending.user_message == RAW
    assert inbox.pending.request.replacement_text == CONTENT
    assert old.feedback_records[-1].text != CONTENT
    schema = inbox.tool.args_schema.model_json_schema()
    assert "replacement_text" in schema["properties"]


def test_unknown_source_still_rejected_without_creating_pending():
    inbox = LearningActionInbox()
    inbox.start_turn(original_state(), RAW)
    with pytest.raises(ValueError, match="来源ID"):
        inbox.tool.invoke(request(source_id="assistant_1").model_dump())
    assert inbox.pending is None
