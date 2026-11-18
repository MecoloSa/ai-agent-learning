import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from tests.helpers.responsibility_lab import history_view, phase_view, snapshot, PlanOutcome
from pydantic import ValidationError
import pytest
from tests.helpers.history_evidence_panel import render_history_evidence


def test_history_view_hides_ids_without_mutating_raw_or_losing_evidence():
    data = {"evidence_kind": "archived_conversation", "archive_id": "private-id", "session_id": "private-session",
        "citation": "long", "citation_ref": "[history-ref=1:1]", "user_text": "原话\r\n未改", "truncated": True,
        "source_index": 1, "turn_number": 1}
    raw = ToolMessage(content=json.dumps({"ok": True, "data": data}), tool_call_id="r")
    viewed = history_view([raw])[0]
    assert json.loads(raw.content)["data"] == data
    new = json.loads(viewed.content)["data"]
    assert "archive_id" not in new and "citation" not in new and "session_id" not in new
    assert new["user_text"] == data["user_text"] and new["truncated"] is True
    assert new["citation_ref"] == data["citation_ref"] and viewed.tool_call_id == "r"


def test_unknown_tool_result_is_not_rewritten():
    raw = ToolMessage(content='{"ok":false,"error":{"code":"scope"}}', tool_call_id="r")
    assert history_view([raw])[0].content == raw.content


def test_phase_view_retains_rules_and_dynamic_state_not_old_chat():
    messages = [SystemMessage(content="规则与当前计划"), HumanMessage(content="旧问题"), AIMessage(content="旧建议"),
        HumanMessage(content='{"context_type":"current_learning_state","tasks":[1]}')]
    new = phase_view(messages)
    assert len(new) == 2 and "current_learning_state" in new[1].content
    assert len(messages) == 4


def test_outcome_does_not_accept_unknown_status():
    with pytest.raises(ValidationError):
        PlanOutcome(status="saved", plan_markdown="新计划", explanation="已保存")


def test_program_evidence_preserves_exact_quote_and_ref_and_deduplicates():
    data = {"evidence_kind": "archived_conversation", "truncated": False,
        "user_text": "用户原话\r\n下一行", "citation": "[history=actual/session, turn=1]"}
    row = {"tool_name": "read_conversation_turn", "executed": True,
        "result": json.dumps({"ok": True, "data": data})}
    panel = render_history_evidence([row, row])
    assert data["user_text"] in panel and panel.count(data["citation"]) == 1
    data["truncated"] = True
    row["result"] = json.dumps({"ok": True, "data": data})
    assert render_history_evidence([row]) == ""
    row["result"] = '{"ok":false,"error":{"code":"out_of_scope"}}'
    assert render_history_evidence([row]) == ""
