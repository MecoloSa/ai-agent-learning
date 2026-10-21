"""自然反馈的工具、确认、保存与下一轮上下文回归。

假模型只验证接线和权限边界；语义是否正确仍需少量在线人工检查。
业务代码按本轮指导应用后，可直接执行本文件。
"""

import builtins
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import ai_agent.main as main
from ai_agent.learning.feedback_proposals import FeedbackProposalArgs, prepare_feedback_proposal
from ai_agent.learning.feedback_runtime import FeedbackInbox, commit_feedback_proposal, review_feedback_proposal
from ai_agent.learning.learning_context import build_learning_state_message
from ai_agent.learning.learning_state import LearningState, LearningTask, learning_today, update_learning_context
from ai_agent.persistence.memory import ConversationMemory
from ai_agent.persistence.state_store import load_learning_state, save_learning_state


def make_state():
    return LearningState(
        learning_goal="理解文章", plan_markdown="阅读并解释",
        tasks=(LearningTask(task_id="t1", title="阅读文章"),),
    )


def payload(scope="task", **changes):
    return {
        "scope": scope,
        "task_id": "t1" if scope == "task" else None,
        "activity_status": "in_progress" if scope == "task" else None,
        "understanding_status": None,
        **changes,
    }


def pending(state, scope="task"):
    args = FeedbackProposalArgs.model_validate(payload(scope))
    text = "只读了一半" if scope == "task" else "今天只剩20分钟"
    return prepare_feedback_proposal(state, args, user_message=text)


def test_tool_keeps_only_one_pending_and_resets_between_turns():
    state = make_state()
    inbox = FeedbackInbox()
    inbox.start_turn(state, "只读了一半")
    result = json.loads(inbox.tool.invoke(payload()))
    first = inbox.pending
    assert result["data"]["status"] == "pending"
    assert not state.feedback_records
    assert state.tasks[0].activity_status == "not_started"
    repeated = json.loads(inbox.tool.invoke(payload(understanding_status="unknown")))
    assert repeated["error"]["code"] == "pending_exists"
    assert inbox.pending is first
    inbox.start_turn(state, "下一轮普通问题")
    assert inbox.pending is None


def test_tool_rejects_fabricated_source():
    inbox = FeedbackInbox()
    inbox.start_turn(make_state(), "解释一下这段")
    with pytest.raises(ValueError):
        inbox.tool.invoke(payload(source_id="assistant_1"))
    assert inbox.pending is None


def test_commit_rejects_replay_and_changed_revision():
    state = make_state()
    proposal = pending(state)
    confirmed = commit_feedback_proposal(state, proposal)
    with pytest.raises(ValueError, match="重复"):
        commit_feedback_proposal(confirmed, proposal)
    changed = update_learning_context(state, plan_markdown="新安排")
    with pytest.raises(ValueError, match="版本"):
        commit_feedback_proposal(changed, proposal)
    assert state.tasks[0].activity_status == "not_started"


def test_eof_is_not_confirmation(monkeypatch):
    state = make_state()
    monkeypatch.setattr(builtins, "input", lambda _: (_ for _ in ()).throw(EOFError()))
    result, decision = review_feedback_proposal(state, pending(state))
    assert result is state
    assert decision == "cancelled"


def test_context_feedback_is_preserved_but_not_permanently_active(tmp_path):
    state = make_state()
    confirmed = commit_feedback_proposal(state, pending(state, "current_context"))
    path = tmp_path / "feedback.json"
    save_learning_state(confirmed, path)
    restored = load_learning_state(path)
    assert len(json.loads(build_learning_state_message(restored).content)["current_context_feedback"]) == 1
    expired = LearningState.model_validate({
        **restored.model_dump(),
        "feedback_records": [{
            **restored.feedback_records[0].model_dump(),
            "recorded_on": learning_today() - timedelta(days=1),
        }],
    })
    assert json.loads(build_learning_state_message(expired).content)["current_context_feedback"] == []
    assert len(expired.feedback_records) == 1
    changed = update_learning_context(restored, plan_markdown="另一版安排")
    assert json.loads(build_learning_state_message(changed).content)["current_context_feedback"] == []
    assert changed.feedback_records == restored.feedback_records


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
@pytest.mark.parametrize("decision", ["y", "n"])
@pytest.mark.parametrize("scope", ["task", "current_context"])
def test_chat_confirmation_save_and_next_turn(monkeypatch, tmp_path, strategy, decision, scope):
    state = make_state()
    current = tmp_path / "current.json"
    message = "只读了一半" if scope == "task" else "今天只剩20分钟"
    inputs = iter([message, decision, "只复述当前记录，不修改", "exit"])
    captured = []
    responses = iter([
        AIMessage(content="", tool_calls=[{
            "name": "propose_learning_feedback", "args": payload(scope), "id": "feedback-1",
        }]),
        AIMessage(content="已提出反馈建议，等待程序确认。"),
        AIMessage(content="根据当前已确认记录回答。"),
    ])

    def fake_input(_):
        value = next(inputs)
        if value == decision:
            assert not current.exists(), "确认前不得保存正式候选"
        return value

    def fake_invoke(llm, messages, *, stage):
        captured.append(list(messages))
        return next(responses)

    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda tools: object())
    monkeypatch.setattr(main, "invoke_llm_response", fake_invoke)
    monkeypatch.setattr(ConversationMemory, "_maybe_compress", lambda self: None)

    main.run_interactive_session(
        analysis="", plan=state.plan_markdown, material_context="", learning_goal=state.learning_goal,
        strategy=strategy, material_dir=str(tmp_path), initial_state=state, current_state_path=current,
    )
    saved = load_learning_state(current)
    assert len(saved.feedback_records) == (1 if decision == "y" else 0)
    assert saved.tasks[0].activity_status == (
        "in_progress" if decision == "y" and scope == "task" else "not_started"
    )
    assert saved.plan_markdown == state.plan_markdown
    assert saved.context_revision == state.context_revision
    assert len(saved.conversation_memory.messages) == 4
    history = "\n".join(m.content for m in saved.conversation_memory.messages)
    assert ("confirmed" if decision == "y" else "cancelled") in history
    data = [
        json.loads(m.content) for m in captured[-1]
        if m.type == "human" and str(m.content).startswith('{"context_type"')
    ][0]
    assert len(data["current_context_feedback"]) == (
        1 if decision == "y" and scope == "current_context" else 0
    )


def test_same_turn_plan_and_feedback_are_both_withheld(monkeypatch, tmp_path):
    state = make_state()
    current = tmp_path / "current.json"
    inputs = iter(["只读了一半", "exit"])

    def fake_turn(**kwargs):
        tool = next(t for t in kwargs["extra_tools"] if t.name == "propose_learning_feedback")
        tool.invoke(payload())
        return "===NEW_PLAN_START===不应生效===NEW_PLAN_END==="

    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda tools: object())
    monkeypatch.setattr(main, "_invoke_interactive_turn", fake_turn)
    main.run_interactive_session(
        "", state.plan_markdown, "", state.learning_goal, "window", str(tmp_path),
        initial_state=state, current_state_path=current,
    )
    saved = load_learning_state(current)
    assert not saved.feedback_records
    assert saved.plan_markdown == state.plan_markdown


def test_explicit_tool_check_means_proposal_only():
    from ai_agent.runtime.execution_guard import tool_result_succeeded
    assert tool_result_succeeded("propose_learning_feedback", '{"ok":true,"data":{"status":"pending"}}')
    assert not tool_result_succeeded("propose_learning_feedback", '{"ok":false,"error":{}}')


def test_proposal_calls_share_existing_budget(monkeypatch):
    inbox = FeedbackInbox()
    inbox.start_turn(make_state(), "只读了一半")
    responses = iter([
        AIMessage(content="", tool_calls=[
            {"name": inbox.tool.name, "args": payload(), "id": "a"},
            {"name": inbox.tool.name, "args": payload(activity_status=None), "id": "b"},
            {"name": inbox.tool.name, "args": payload(understanding_status="unknown"), "id": "c"},
        ]),
        AIMessage(content="等待程序确认。"),
    ])
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(interactive_max_retrieval_calls=2))
    monkeypatch.setattr(main, "invoke_llm_response", lambda *a, **kw: next(responses))
    audit = {}
    main._invoke_interactive_turn(
        request_messages=[], user_input="只读了一半", plain_llm=object(), tool_llm=object(),
        retrieval_tool=None, extra_tools=(inbox.tool,), audit_trace=audit,
    )
    results = [json.loads(item["result"]) for item in audit["tool_results"]]
    assert results[0]["data"]["status"] == "pending"
    assert results[1]["error"]["code"] == "pending_exists"
    assert results[2]["error"]["code"] == "budget_exceeded"
    assert [r["executed"] for r in audit["tool_results"]] == [True, True, False]


def test_save_failure_is_not_reported_as_durable_success(monkeypatch, tmp_path, capsys):
    before = make_state()
    after = commit_feedback_proposal(before, pending(before))

    def fail_save(*_):
        raise OSError("模拟磁盘写入失败")

    monkeypatch.setattr(main, "save_current_learning_state", fail_save)
    result = main._autosave_learning_change(before, after, current_path=tmp_path / "current.json")
    output = capsys.readouterr().out
    assert result is None
    assert "自动保存失败" in output
    assert "自动保存成功" not in output
    assert len(after.feedback_records) == 1
    assert not before.feedback_records


def test_legacy_v4_loads_with_empty_feedback_records():
    fixture = Path(__file__).parent / "fixtures/evaluation/feedback_adaptation/state.json"
    assert json.loads(fixture.read_text(encoding="utf-8"))["schema_version"] == "learning-state-v4"
    assert load_learning_state(fixture).feedback_records == ()
