"""真实工具循环的交付边界；不调用在线模型。"""
import json

from langchain_core.messages import AIMessage
import pytest

import ai_agent.main as app
from ai_agent.learning.feedback_runtime import LearningActionInbox
from ai_agent.learning.learning_state import LearningState
from ai_agent.learning.plan_proposals import PlanProposalInbox
from ai_agent.resources.tools import build_read_material_lines_tool
from ai_agent.runtime.source_delivery import TurnSourceLedger


@pytest.mark.parametrize("repair", [False, True])
def test_failed_feedback_cannot_claim_recorded_but_may_repair(monkeypatch, repair):
    state = LearningState(learning_goal="学习", plan_markdown="原计划")
    inbox = LearningActionInbox()
    question = "请记录：今天最多20分钟，只阅读和口头说明。不要修改计划或任务。"
    inbox.start_turn(state, question)
    bad = dict(scope="current_context", task_id="missing", activity_status=None,
               understanding_status=None)
    good = {**bad, "task_id": None}
    responses = [AIMessage(content="操作开始。", tool_calls=[
        {"name": inbox.tool.name, "args": bad, "id": "bad"}])]
    if repair:
        responses.append(AIMessage(content="", tool_calls=[
            {"name": inbox.tool.name, "args": good, "id": "good"}]))
    responses.append(AIMessage(content="您的反馈已记录。"))
    stream = iter(responses)
    monkeypatch.setattr(app, "invoke_llm_response", lambda *a, **k: next(stream))
    audit = {}
    answer = app._invoke_interactive_turn(
        request_messages=[], user_input=question, plain_llm=object(), tool_llm=object(),
        retrieval_tool=None, extra_tools=(inbox.tool,), feedback_inbox=inbox,
        audit_trace=audit,
    )
    assert "您的反馈已记录" not in answer
    assert ("尚未生效" if repair else "没有成功生成") in answer
    assert (inbox.pending is not None) is repair
    assert not state.feedback_records
    assert state.plan_markdown == "原计划"
    assert json.loads(audit["tool_results"][0]["result"])["ok"] is False


def test_footer_and_plan_body_use_separate_channels(monkeypatch, tmp_path):
    (tmp_path / "notes.md").write_text("阅读与口头说明。", encoding="utf-8")
    state = LearningState(learning_goal="学习", plan_markdown="原计划")
    plans = PlanProposalInbox(app.PLAN_START_TAG, app.PLAN_END_TAG,
                             (app.GOAL_START_TAG, app.GOAL_END_TAG))
    plans.start_turn(state)
    reader = build_read_material_lines_tool(str(tmp_path))
    stream = iter([
        AIMessage(content="", tool_calls=[{"name": reader.name, "id": "read",
            "args": {"relative_path": "notes.md", "start_line": 1, "end_line": 1}}]),
        AIMessage(content="", tool_calls=[{"name": plans.tool.name, "id": "plan",
            "args": {"plan_markdown": "阅读5分钟，口头说明15分钟。"}}]),
    ])
    monkeypatch.setattr(app, "invoke_llm_response", lambda *a, **k: next(stream))
    answer = app._invoke_interactive_turn(
        request_messages=[], user_input="请调整计划。", plain_llm=object(), tool_llm=object(),
        retrieval_tool=None, read_tool=reader, extra_tools=(plans.tool,), plan_inbox=plans,
    )
    assert "程序整理的本轮读取来源" in answer
    assert plans.confirmation_plan == "阅读5分钟，口头说明15分钟。"


def test_action_delivery_keeps_legacy_goal_conflict_visible(monkeypatch):
    state = LearningState(learning_goal="学习", plan_markdown="原计划")
    inbox = LearningActionInbox()
    inbox.start_turn(state, "请记录今天20分钟。")
    plans = PlanProposalInbox(app.PLAN_START_TAG, app.PLAN_END_TAG,
                             (app.GOAL_START_TAG, app.GOAL_END_TAG))
    plans.start_turn(state)
    stream = iter([
        AIMessage(content="", tool_calls=[{"name": inbox.tool.name, "id": "feedback",
            "args": {"scope": "current_context", "task_id": None,
                     "activity_status": None, "understanding_status": None}}]),
        AIMessage(content=app.GOAL_START_TAG + "新目标" + app.GOAL_END_TAG),
    ])
    monkeypatch.setattr(app, "invoke_llm_response", lambda *a, **k: next(stream))
    answer = app._invoke_interactive_turn(
        request_messages=[], user_input="请记录今天20分钟。", plain_llm=object(),
        tool_llm=object(), retrieval_tool=None, extra_tools=(inbox.tool, plans.tool),
        feedback_inbox=inbox, plan_inbox=plans,
    )
    # 外层会话仍须看到旧入口标记，才能按原流程拒绝混合操作。
    assert inbox.pending is not None and app.GOAL_START_TAG in answer
    assert not state.feedback_records and state.learning_goal == "学习"


def test_ledger_deduplicates_and_preserves_limits_without_inventing_sources():
    ledger = TurnSourceLedger()
    citation = "[source=notes.md, lines=1-2]"
    result = json.dumps({"ok": True, "data": {"citation": citation,
        "truncated": True, "lines": [{"partial": True}]}})
    for _ in range(2):
        ledger.observe("read_material_lines", result, executed=True)
    ledger.observe("read_material_lines", '{"ok":false}', executed=True)
    ledger.observe("read_material_lines", result.replace("notes.md", "unread.md"), executed=False)
    ledger.observe("retrieve_learning_context", result.replace("notes.md", "retrieved.md"), executed=True)
    ledger.observe("read_pdf_pages", json.dumps({"ok": True, "data": {"pages": [
        {"citation": "[source=a.pdf, page=1]", "text_status": "no_text"}]}}), executed=True)
    text = ledger.render()
    assert text.count(citation) == 1
    assert "截断" in text and "未完整返回" in text and "不代表页面空白" in text
    assert "unread.md" not in text and "retrieved.md" not in text
    assert not TurnSourceLedger().render()
