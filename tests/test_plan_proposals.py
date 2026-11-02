"""计划候选工具契约；候选未应用时应明确失败，不跳过。"""
from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

import ai_agent.main as app
from ai_agent.learning.learning_state import LearningState


@pytest.fixture
def state():
    return LearningState(learning_goal="理解概念", plan_markdown="原计划")


@pytest.fixture
def inbox(state):
    module = importlib.import_module("ai_agent.learning.plan_proposals")
    value = module.PlanProposalInbox(app.PLAN_START_TAG, app.PLAN_END_TAG,
                                    (app.GOAL_START_TAG, app.GOAL_END_TAG))
    value.start_turn(state)
    return value


def test_success_only_registers_pending_and_preserves_state(inbox, state):
    before = state.model_dump()
    result = json.loads(inbox.tool.invoke({"plan_markdown": "阅读5分钟，口述15分钟。"}))
    assert result == {"ok": True, "data": {"status": "pending", "action": "plan_change"}}
    assert inbox.pending.plan_markdown == "阅读5分钟，口述15分钟。"
    assert state.model_dump() == before


@pytest.mark.parametrize("payload", [
    {"plan_markdown": " "}, {"plan_markdown": 123},
    {"plan_markdown": "新计划", "confirmed": True},
    {"plan_markdown": "新计划", "learning_goal": "更换目标"},
    {"plan_markdown": "新计划", "context_revision": 5},
])
def test_schema_rejects_blank_or_state_injection(inbox, payload):
    with pytest.raises(ValidationError):
        inbox.tool.invoke(payload)
    assert inbox.pending is None


@pytest.mark.parametrize("marker", [app.PLAN_START_TAG, app.PLAN_END_TAG, app.GOAL_START_TAG, app.GOAL_END_TAG])
def test_reserved_markers_are_rejected_in_plan_body(inbox, marker):
    result = json.loads(inbox.tool.invoke({"plan_markdown": "新计划" + marker}))
    assert result["ok"] is False and result["error"]["code"] == "reserved_marker"
    assert inbox.pending is None


def test_same_plan_and_second_submission_do_not_replace_pending(inbox, state):
    result = json.loads(inbox.tool.invoke({"plan_markdown": state.plan_markdown}))
    assert result["error"]["code"] == "unchanged_plan"
    inbox.tool.invoke({"plan_markdown": "第一份"})
    result = json.loads(inbox.tool.invoke({"plan_markdown": "第二份"}))
    assert result["error"]["code"] == "pending_exists"
    assert inbox.pending.plan_markdown == "第一份"
    inbox.start_turn(state)
    assert inbox.pending is None
    assert inbox.prepare_delivery("普通回答") is None
    assert inbox.confirmation_plan is None and inbox.conflicted is False


@pytest.mark.parametrize("marker", [app.PLAN_START_TAG, app.GOAL_START_TAG])
def test_tool_and_legacy_marker_conflict_does_not_produce_an_update(inbox, marker):
    inbox.tool.invoke({"plan_markdown": "工具候选"})
    assert inbox.prepare_delivery("另一入口" + marker) is None
    assert inbox.conflicted is True and inbox.confirmation_plan is None


def test_feedback_conflict_is_explicit_data_not_a_text_marker(inbox):
    inbox.tool.invoke({"plan_markdown": "工具候选"})
    assert inbox.prepare_delivery("", has_learning_action=True) is None
    assert inbox.conflicted is True and inbox.confirmation_plan is None


@pytest.mark.parametrize("require_read", [False, True])
def test_batch_results_finish_before_return_and_required_execution_is_not_bypassed(inbox, monkeypatch, require_read):
    settings = replace(app.get_settings(), interactive_max_retrieval_calls=8)
    monkeypatch.setattr(app, "get_settings", lambda: settings)
    calls = []
    read = SimpleNamespace(name="read_material_lines", invoke=lambda args: '{"ok":true,"data":{}}')
    other = SimpleNamespace(name="auxiliary", invoke=lambda args: calls.append("auxiliary") or "done")
    def model(*args, **kwargs):
        calls.append("model")
        assert calls.count("model") == 1
        return AIMessage(content="", tool_calls=[
            {"name": "propose_plan_change", "args": {"plan_markdown": "新计划"}, "id": "p", "type": "tool_call"},
            {"name": "auxiliary", "args": {}, "id": "a", "type": "tool_call"},
        ])
    monkeypatch.setattr(app, "invoke_llm_response", model)
    trace = {}
    text = app._invoke_interactive_turn(request_messages=[],
        user_input="请实际调用 read_material_lines" if require_read else "请调整计划。",
        plain_llm=object(), tool_llm=object(), retrieval_tool=None,
        extra_tools=(inbox.tool, read, other), plan_inbox=inbox, audit_trace=trace)
    assert calls == ["model", "auxiliary"]
    assert len(trace["tool_results"]) == 2
    if require_read:
        assert app.PLAN_START_TAG not in text
        assert "未实际执行指定工具" in text
        assert inbox.confirmation_plan is None
    else:
        assert inbox.confirmation_plan == "新计划"
        assert "新计划" in text
        assert app.PLAN_START_TAG not in text and app.PLAN_END_TAG not in text


@pytest.mark.parametrize("body", ["先阅读，再口述。", "# 安排\n\n- 比较例子\n- 口头说明", "| 活动 |\n|---|\n| 口述 |"])
def test_candidate_body_format_is_not_forced_to_a_display_template(inbox, body):
    inbox.tool.invoke({"plan_markdown": body})
    selected = inbox.prepare_delivery("模型解释不决定候选正文。")
    assert selected.plan_markdown == body
    assert inbox.confirmation_plan is None  # 需要出口完成来源规范化才交给确认层。


def test_confirmation_channel_and_conflict_are_reset_for_next_turn(inbox, state):
    inbox.tool.invoke({"plan_markdown": "旧候选"})
    inbox.confirmation_plan = "已规范化的旧候选"
    inbox.conflicted = True
    inbox.start_turn(state)
    assert inbox.pending is None
    assert inbox.confirmation_plan is None and inbox.conflicted is False
