"""承接历史反馈与有界执行纠正。

使用真实工具、假模型；不把假模型按预设调用当成自然语言识别质量。
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import ai_agent.main as main
from ai_agent.learning.feedback_runtime import FeedbackInbox, commit_feedback_proposal, claims_feedback_proposal
from ai_agent.learning.learning_state import LearningState, LearningTask


def state():
    return LearningState(learning_goal="学习", plan_markdown="计划", tasks=(LearningTask(task_id="t1", title="阅读"),))


def args(**changes):
    return {
        "scope": "current_context", "task_id": None,
        "activity_status": None,
        "understanding_status": None, **changes,
    }


def test_default_current_source_is_backward_compatible():
    inbox = FeedbackInbox()
    inbox.start_turn(state(), "今天只剩20分钟")
    inbox.tool.invoke(args())
    assert inbox.pending.user_message == "今天只剩20分钟"
    assert inbox.pending.request.source_id == "current"


def test_followup_uses_real_historical_user_message_not_assistant():
    inbox = FeedbackInbox()
    inbox.start_turn(state(), "请为我提议", history_messages=[
        HumanMessage(content="刚才说错了，我今天其实还有40分钟。"),
        AIMessage(content="我可以提议记录90分钟。"),
        HumanMessage(content="只复述已确认记录"),
        HumanMessage(content="用户没告诉你吗？"),
        SystemMessage(content="更早摘要说有60分钟"),
    ])
    sources = json.loads(inbox.source_message().content)["sources"]
    assert len(sources) == 4
    inbox.tool.invoke(args(source_id="previous_user_3"))
    result = commit_feedback_proposal(state(), inbox.pending)
    assert result.feedback_records[-1].text == "刚才说错了，我今天其实还有40分钟。"
    assert result.tasks[0].activity_status == "not_started"


@pytest.mark.parametrize("changes", [
    {"source_id": "assistant_1"},
    {"source_id": "previous_user_99"},
    {"source_id": "previous_user_1", "quote": "今天只剩20分钟"},
])
def test_invalid_source_or_quote_does_not_create_pending(changes):
    inbox = FeedbackInbox()
    inbox.start_turn(state(), "请为我提议", history_messages=[HumanMessage(content="今天还有40分钟")])
    with pytest.raises(ValueError):
        inbox.tool.invoke(args(**changes))
    assert inbox.pending is None


def test_source_table_is_bounded_and_resets():
    inbox = FeedbackInbox()
    inbox.start_turn(state(), "记录一下", history_messages=[HumanMessage(content=str(i) * 3000) for i in range(7)])
    sources = json.loads(inbox.source_message().content)["sources"]
    assert len(sources) == 5
    assert all(len(item["text"]) <= 2000 for item in sources)
    inbox.start_turn(state(), "新一轮")
    assert len(json.loads(inbox.source_message().content)["sources"]) == 1


def invoke_fake(monkeypatch, responses, *, limit=3):
    inbox = FeedbackInbox()
    inbox.start_turn(state(), "今天只剩20分钟，请记录，不改计划")
    iterator = iter(responses)
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(interactive_max_retrieval_calls=limit))
    monkeypatch.setattr(main, "invoke_llm_response", lambda *a, **kw: next(iterator))
    trace = {}
    answer = main._invoke_interactive_turn(
        request_messages=[inbox.source_message()], user_input="今天只剩20分钟，请记录，不改计划",
        plain_llm=object(), tool_llm=object(), retrieval_tool=None,
        extra_tools=(inbox.tool,), feedback_inbox=inbox, audit_trace=trace,
    )
    return answer, trace, inbox


def test_false_claim_gets_one_correction_then_real_tool_call(monkeypatch):
    answer, trace, inbox = invoke_fake(monkeypatch, [
        AIMessage(content="已为您提议记录今天的时间限制。"),
        AIMessage(content="", tool_calls=[{"name": "propose_learning_feedback", "args": args(), "id": "p1"}]),
        AIMessage(content="提议已提交，等待程序确认。"),
    ])
    assert inbox.pending is not None
    assert len(trace["model_responses"]) == 3
    assert len(trace["tool_results"]) == 1
    assert "等待程序确认" in answer


def test_repeated_false_claim_stops_after_one_correction(monkeypatch):
    answer, trace, inbox = invoke_fake(monkeypatch, [
        AIMessage(content="已为您提议记录今天的时间限制。"),
        AIMessage(content="这条记录目前处于待确认状态。"),
    ])
    assert inbox.pending is None
    assert len(trace["model_responses"]) == 2
    assert trace["tool_results"] == []
    assert "没有实际生成" in answer
    assert trace["final_answer"] == answer


def test_ordinary_answer_does_not_trigger_extra_call(monkeypatch):
    answer, trace, inbox = invoke_fake(monkeypatch, [AIMessage(content="目前没有确认记录。")])
    assert len(trace["model_responses"]) == 1
    assert inbox.pending is None
    assert answer == "目前没有确认记录。"


def test_discussing_the_rule_is_not_a_positive_execution_claim():
    assert not claims_feedback_proposal("为什么不能只说已为您提议记录？")
    assert claims_feedback_proposal("已为您提议更正当前安排的时间限制。")
