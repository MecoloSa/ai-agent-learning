"""明确更正当前安排反馈：来源、确认、历史关系、读取和恢复回归。

不按文本相似度猜测覆盖，不调用在线模型。假模型只证明工具/状态接线，
不能证明模型能正确选择更正对象，仍需在线查看新旧原话的确认预览。
"""
import builtins
import json
from dataclasses import replace
from datetime import timedelta

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

import ai_agent.main as main
from ai_agent.learning.feedback_proposals import FeedbackProposalArgs, prepare_feedback_proposal
from ai_agent.learning.feedback_runtime import commit_feedback_proposal, review_feedback_proposal
from ai_agent.learning.learning_context import build_learning_state_message
from ai_agent.learning.learning_state import FeedbackRecord, LearningState, LearningTask, learning_today
from ai_agent.persistence.memory import ConversationMemory
from ai_agent.persistence.state_store import load_learning_state, save_learning_state


def record(record_id="limit-20", target=None, **changes):
    return FeedbackRecord.model_validate({
        "record_id": record_id, "scope": "current_context", "task_id": None,
        "quote": "今天20分钟", "text": "今天20分钟", "context_revision": 1,
        "recorded_on": learning_today(), "activity_status": None,
        "understanding_status": None, "supersedes_record_id": target, **changes,
    })


def state(records=None):
    return LearningState(
        learning_goal="区分两类停止", plan_markdown="比较10分钟，复述10分钟。",
        context_revision=2, tasks_need_review=True,
        tasks=(LearningTask(task_id="read-001", title="阅读定义", activity_status="completed"),),
        feedback_records=(record(),) if records is None else records,
    )


def args(target="limit-20", **changes):
    return FeedbackProposalArgs.model_validate({
        "scope": "current_context", "task_id": None,
        "activity_status": None, "understanding_status": None,
        "supersedes_record_id": target,
        "replacement_text": "今天有40分钟。" if target is not None else None,
        **changes,
    })


def pending(original, target="limit-20", text="刚才说错了，今天其实有40分钟，请更正，不改计划。"):
    return prepare_feedback_proposal(
        original,
        args(target, replacement_text=text if target is not None else None),
        user_message=text,
    )


def view(value):
    return json.loads(build_learning_state_message(value).content)


def test_commit_keeps_original_and_links_new_record_without_changing_plan_or_tasks():
    original = state()
    before = original.model_dump()
    proposal = pending(original)
    assert original.model_dump() == before
    updated = commit_feedback_proposal(original, proposal)
    assert updated.feedback_records[0] == original.feedback_records[0]
    assert updated.feedback_records[-1].supersedes_record_id == "limit-20"
    assert updated.feedback_records[-1].context_revision == 2
    assert updated.feedback_records[-1].text == proposal.user_message
    assert updated.context_revision == 2
    assert updated.plan_markdown == original.plan_markdown
    assert updated.tasks == original.tasks
    assert updated.tasks_need_review is True
    assert view(updated)["previous_context_feedback"] == []
    assert [r["record_id"] for r in view(updated)["current_context_feedback"]] == [proposal.proposal_id]
    assert view(updated)["feedback_replacements"] == [
        {"old_record_id": "limit-20", "new_record_id": proposal.proposal_id},
    ]


@pytest.mark.parametrize("answer,decision", [("y", "confirmed"), ("n", "cancelled"), ("", "cancelled")])
def test_confirmation_shows_original_before_deciding(monkeypatch, capsys, answer, decision):
    original = state()
    monkeypatch.setattr(builtins, "input", lambda _: answer)
    after, actual = review_feedback_proposal(original, pending(original))
    output = capsys.readouterr().out
    assert "[反馈更正确认]" in output and "今天20分钟" in output and "40分钟" in output
    assert actual == decision
    if decision == "cancelled":
        assert after is original
        assert view(after)["feedback_replacements"] == []


def test_ordinary_addition_never_implicitly_replaces_prior_record():
    updated = commit_feedback_proposal(state(), pending(state(), target=None, text="今天希望优先比较例子。"))
    assert len(updated.feedback_records) == 2
    assert view(updated)["feedback_replacements"] == []
    assert len(view(updated)["previous_context_feedback"]) == 1


def test_chain_cannot_branch_or_replay_and_does_not_resurrect_old_records():
    original = state()
    first = pending(original)
    revised = commit_feedback_proposal(original, first)
    with pytest.raises(ValueError):
        commit_feedback_proposal(revised, first)
    with pytest.raises(ValueError, match="已被更正"):
        prepare_feedback_proposal(revised, args(), user_message="改成60分钟。")
    latest = commit_feedback_proposal(revised, pending(revised, first.proposal_id, "改成60分钟。"))
    assert len(latest.feedback_records) == 3
    visible = view(latest)
    assert visible["previous_context_feedback"] == []
    assert len(visible["current_context_feedback"]) == 1
    assert visible["current_context_feedback"][0]["text"] == "改成60分钟。"
    assert len(visible["feedback_replacements"]) == 2


@pytest.mark.parametrize("target", ["missing", ""])
def test_unknown_or_empty_target_rejected(target):
    with pytest.raises(ValueError):
        prepare_feedback_proposal(state(), args(target), user_message="改成40分钟。")


def test_task_feedback_cannot_use_replacement_link():
    with pytest.raises(ValidationError):
        args(scope="task", task_id="read-001")
    original = state((record(scope="task", task_id="read-001"),))
    with pytest.raises(ValueError):
        prepare_feedback_proposal(original, args(), user_message="更正任务。")


def test_outside_visible_day_target_is_not_silently_corrected():
    old = record(recorded_on=learning_today() - timedelta(days=1))
    with pytest.raises(ValueError, match="当天"):
        prepare_feedback_proposal(state((old,)), args(), user_message="更正为40分钟。")


@pytest.mark.parametrize("records", [
    lambda: (record(), record()),
    lambda: (record("new", "missing"),),
    lambda: (record("self", "self"),),
    lambda: (record("new", "old"), record("old")),
    lambda: (record(), record("a", "limit-20"), record("b", "limit-20")),
    lambda: (record(scope="task", task_id="read-001"), record("new", "limit-20")),
])
def test_invalid_saved_history_is_rejected(records):
    with pytest.raises(ValidationError):
        state(records())


def test_pending_target_rechecked_even_if_plan_revision_did_not_change():
    original = state()
    a = pending(original)
    b = pending(original)
    adopted = commit_feedback_proposal(original, a)
    assert adopted.context_revision == original.context_revision
    with pytest.raises(ValueError, match="已被更正"):
        commit_feedback_proposal(adopted, b)


def test_revision_guard_not_bypassed():
    original = state()
    proposal = pending(original)
    altered = LearningState.model_validate({**original.model_dump(), "context_revision": 3})
    with pytest.raises(ValueError, match="版本"):
        commit_feedback_proposal(altered, proposal)


def test_v6_read_defaults_to_no_link_and_v7_roundtrip_preserves_link(tmp_path):
    raw = state().model_dump(mode="json")
    del raw["feedback_records"][0]["supersedes_record_id"]
    old = tmp_path / "v6.json"
    old.write_text(json.dumps({"schema_version": "learning-state-v6", "state": raw}), encoding="utf-8")
    restored = load_learning_state(old)
    assert restored.feedback_records[0].supersedes_record_id is None
    revised = commit_feedback_proposal(restored, pending(restored))
    path = save_learning_state(revised, tmp_path / "v7.json")
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "learning-state-v7"
    assert load_learning_state(path) == revised
    assert len(view(load_learning_state(path))["current_context_feedback"]) == 1


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
@pytest.mark.parametrize("decision", ["y", "n"])
def test_tool_confirmation_next_turn_and_restore(monkeypatch, tmp_path, strategy, decision):
    original = state()
    inputs = iter(["刚才20分钟说错了，改成40分钟，请记录，不改计划。", decision, "只查当前未被更正的反馈", "exit"])
    responses = iter([
        AIMessage(content="", tool_calls=[{"name": "propose_learning_feedback", "args": args().model_dump(), "id": "correction-1"}]),
        AIMessage(content="已提出更正建议，等待确认。"), AIMessage(content="测试替身，不代表语义验收。"),
    ])
    requests = []

    def invoke(llm, messages, *, stage):
        requests.append(messages)
        return next(responses)

    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(main, "invoke_llm_response", invoke)
    monkeypatch.setattr(ConversationMemory, "_maybe_compress", lambda self: None)
    path = tmp_path / "current.json"
    main.run_interactive_session("", original.plan_markdown, "", original.learning_goal, strategy,
                                 str(tmp_path), initial_state=original, current_state_path=path)
    restored = load_learning_state(path)
    assert restored.plan_markdown == original.plan_markdown
    assert restored.tasks == original.tasks
    adopted = decision == "y"
    assert len(restored.feedback_records) == (2 if adopted else 1)
    data = next(json.loads(m.content) for m in requests[-1] if m.type == "human"
                and str(m.content).startswith('{"context_type": "current_learning_state"'))
    assert len(data["feedback_replacements"]) == (1 if adopted else 0)
    assert len(data["previous_context_feedback"]) == (0 if adopted else 1)
    assert restored.feedback_records[0] == original.feedback_records[0]
    assert view(restored)["feedback_replacements"] == data["feedback_replacements"]
