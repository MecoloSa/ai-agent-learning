"""确认原因与实际操作回执的回归，不调用在线模型。

只记录终端确认事实，不推断用户心理原因；拒绝候选不写入正式反馈。
覆盖三类学习操作，以及下一轮请求、记忆保存和恢复中的真实接线。
"""
import builtins
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import ai_agent.main as main
import ai_agent.persistence.memory as memory_module
from ai_agent.learning.confirmation import read_confirmation
from ai_agent.learning.feedback_proposals import FeedbackProposalArgs, prepare_feedback_proposal
from ai_agent.learning.feedback_runtime import build_learning_action_history, review_learning_action
from ai_agent.learning.learning_state import LearningState, LearningTask
from ai_agent.learning.task_creation import TaskCreationArgs, prepare_task_creation
from ai_agent.learning.task_cancellation import TaskCancellationArgs, prepare_task_cancellation
from ai_agent.persistence.state_store import load_learning_state


def original():
    return LearningState(learning_goal="区分两类停止", plan_markdown="比较后复述。",
                         tasks=(LearningTask(task_id="t1", title="阅读定义"),))


def receipt(text):
    return json.loads(text.split("\n【程序学习操作结果数据】\n", 1)[1])


@pytest.mark.parametrize("answer,accepted,reason", [
    (" y ", True, "user_confirmed"), ("N", False, "user_declined"),
    ("", False, "user_not_confirmed"), ("其他输入", False, "user_not_confirmed"),
    (None, False, "input_ended"),
])
def test_confirmation_distinguishes_decline_from_input_ending(monkeypatch, answer, accepted, reason):
    def read(_):
        if answer is None:
            raise EOFError
        return answer
    monkeypatch.setattr(builtins, "input", read)
    trace = {}
    assert read_confirmation("确认？", trace=trace) is accepted
    assert trace == {"reason": reason}


@pytest.mark.parametrize("kind", ["feedback", "creation", "cancellation"])
@pytest.mark.parametrize("answer,reason", [("y", "user_confirmed"), ("n", "user_declined"), (None, "input_ended")])
def test_all_learning_reviews_pass_real_confirmation_reason(monkeypatch, kind, answer, reason):
    state = original()
    if kind == "feedback":
        pending = prepare_feedback_proposal(state, FeedbackProposalArgs(scope="current_context", task_id=None,
            activity_status=None, understanding_status=None), user_message="今天60分钟。")
    elif kind == "creation":
        pending = prepare_task_creation(state, TaskCreationArgs(title="复述区别"), user_message="新增复述任务")
    else:
        pending = prepare_task_cancellation(state, TaskCancellationArgs(task_id="t1"), user_message="取消阅读任务")

    def read(_):
        if answer is None:
            raise EOFError
        return answer
    monkeypatch.setattr(builtins, "input", read)
    trace = {}
    updated, decision = review_learning_action(state, pending, confirmation_trace=trace)
    data = receipt(build_learning_action_history("候选尚未生效", updated, pending, decision,
                                                confirmation_trace=trace))
    assert data["confirmation_reason"] == reason
    assert data["decision"] == ("confirmed" if answer == "y" else "cancelled")
    if reason == "user_declined":
        assert "用户在确认环节输入n" in data["decision_note"]
    elif reason == "input_ended":
        assert "不能说成用户明确否决" in data["decision_note"]
    if answer != "y":
        assert updated is state
        assert not updated.feedback_records


def test_receipt_without_confirmation_evidence_does_not_invent_user_decline():
    state = original()
    pending = prepare_feedback_proposal(state, FeedbackProposalArgs(scope="current_context", task_id=None,
        activity_status=None, understanding_status=None), user_message="今天60分钟。")
    data = receipt(build_learning_action_history("提议", state, pending, "cancelled"))
    assert data["confirmation_reason"] == "not_recorded"
    assert "不能推断未采用原因" in data["decision_note"]


def test_inconsistent_reason_is_not_promoted_to_fact():
    state = original()
    pending = prepare_feedback_proposal(state, FeedbackProposalArgs(scope="current_context", task_id=None,
        activity_status=None, understanding_status=None), user_message="今天60分钟。")
    data = receipt(build_learning_action_history("提议", state, pending, "confirmed",
                                                confirmation_trace={"reason": "user_declined"}))
    assert data["confirmation_reason"] == "not_recorded"


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
def test_declined_revision_reaches_next_request_and_saved_memory(monkeypatch, tmp_path, strategy):
    state = original()
    inputs = iter(["今天60分钟，请记录，先不改计划", "n", "为什么刚才没有记录60分钟？", "exit"])
    requests = []

    def invoke(**kwargs):
        requests.append(kwargs["request_messages"])
        if len(requests) == 1:
            kwargs["feedback_inbox"].tool.invoke({"scope": "current_context", "task_id": None,
                "activity_status": None, "understanding_status": None})
            return "已提出60分钟候选，等待确认。"
        return "测试替身，仅验证回执数据流。"

    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(main, "_invoke_interactive_turn", invoke)
    monkeypatch.setattr(memory_module.ConversationMemory, "_maybe_compress", lambda self: None)
    path = tmp_path / "current.json"
    main.run_interactive_session("", state.plan_markdown, "", state.learning_goal, strategy, str(tmp_path),
                                initial_state=state, current_state_path=path)
    message = next(m.content for m in requests[1] if m.type == "ai" and "程序学习操作结果数据" in m.content)
    data = receipt(message)
    assert data["confirmation_reason"] == "user_declined"
    assert data["feedback_proposal"]["text"] == "今天60分钟，请记录，先不改计划"
    restored = load_learning_state(path)
    assert restored.feedback_records == state.feedback_records
    assert restored.plan_markdown == state.plan_markdown
    assert receipt(restored.conversation_memory.messages[1].content)["confirmation_reason"] == "user_declined"
    memory = memory_module.ConversationMemory.from_snapshot("当前规则", restored.conversation_memory)
    assert "user_declined" in memory.messages[2].content


@pytest.mark.parametrize("answer,reason", [("n", "user_declined"), (None, "input_ended")])
def test_plan_confirmation_uses_same_reason_contract(monkeypatch, answer, reason):
    state = original()

    def read(_):
        if answer is None:
            raise EOFError
        return answer
    monkeypatch.setattr(builtins, "input", read)
    trace = {}
    updated, decision = main._review_context_change(state, plan_markdown="新计划", confirmation_trace=trace)
    assert updated is state and decision == "cancelled"
    text = main._context_change_history("候选新计划", decision, state.context_revision, confirmation_trace=trace)
    data = json.loads(text.split("\n【程序确认结果数据】\n", 1)[1])
    assert data["confirmation_reason"] == reason


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
@pytest.mark.parametrize("answer", ["y", "n"])
def test_plan_review_reason_is_wired_through_real_chat_loop(monkeypatch, tmp_path, strategy, answer):
    state = original()
    inputs = iter(["请把计划改为比较后举例", answer, "刚才计划变更是否采用，原因是什么？", "exit"])
    requests = []

    def invoke(**kwargs):
        requests.append(kwargs["request_messages"])
        if len(requests) == 1:
            return "===NEW_PLAN_START===比较后举例。===NEW_PLAN_END==="
        return "只核对程序记录。"
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(main, "_invoke_interactive_turn", invoke)
    monkeypatch.setattr(memory_module.ConversationMemory, "_maybe_compress", lambda self: None)
    path = tmp_path / "plan.json"
    main.run_interactive_session("", state.plan_markdown, "", state.learning_goal, strategy, str(tmp_path),
                                initial_state=state, current_state_path=path)
    text = next(m.content for m in requests[1] if m.type == "ai" and "程序确认结果数据" in m.content)
    data = json.loads(text.split("\n【程序确认结果数据】\n", 1)[1])
    assert data["confirmation_reason"] == ("user_confirmed" if answer == "y" else "user_declined")
    saved = load_learning_state(path)
    assert saved.plan_markdown == ("比较后举例。" if answer == "y" else state.plan_markdown)
    assert saved.tasks == state.tasks and saved.feedback_records == state.feedback_records


def test_summary_sees_actual_reason_and_new_retention_rule(monkeypatch):
    calls = []

    def summarize(messages):
        calls.append(messages)
        return AIMessage(content="用户输入n否决60分钟候选，未新增正式记录。")
    monkeypatch.setattr(memory_module, "create_llm", lambda: SimpleNamespace(invoke=summarize))
    state = original()
    pending = prepare_feedback_proposal(state, FeedbackProposalArgs(scope="current_context", task_id=None,
        activity_status=None, understanding_status=None), user_message="今天60分钟。")
    text = build_learning_action_history("候选60分钟", state, pending, "cancelled",
                                        confirmation_trace={"reason": "user_declined"})
    memory = memory_module.ConversationMemory(system_prompt="当前规则", strategy="summary", max_chars=1)
    memory.add_user("今天60分钟")
    memory.add_ai(text)
    memory.add_user("继续讨论")
    memory.add_ai("不修改状态")
    assert "confirmation_reason" in calls[0][0].content
    assert "不编造用户拒绝的动机" in calls[0][0].content
    assert '"confirmation_reason": "user_declined"' in calls[0][1].content
