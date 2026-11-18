"""不完整回执原样保留，完整输出等价，并检查规划预算联动。"""
import json
from unittest.mock import patch

import pytest

import ai_agent.runtime.agents as agents
from tests.helpers.selective_receipts import candidate_formatter
from tests.helpers.receipt_view_compare import build_history, CASES
from tests.test_read_receipt_view_candidate import receipt
from tests.helpers.scaffold_lab import stage_evidence as previous_formatter


@pytest.mark.parametrize("truncated, partial", [(True, False), (False, True), (True, True)])
def test_incomplete_receipt_is_unchanged(truncated, partial):
    payload = json.loads(receipt())
    payload["data"]["truncated"] = truncated
    payload["data"]["lines"][0]["partial"] = partial
    raw = json.dumps(payload, ensure_ascii=False)
    assert candidate_formatter()(raw) == raw


def test_complete_receipt_matches_current_format():
    assert candidate_formatter()(receipt()) == previous_formatter(receipt())
    assert agents.stage_evidence(receipt()) == previous_formatter(receipt())


def test_first_model_request_after_truncation_matches_raw_view(tmp_path):
    case = json.loads(CASES.read_text(encoding="utf-8"))[1]
    history = build_history(case, tmp_path)
    raw = agents.prepare_model_messages(history)
    selective = agents.prepare_model_messages(history, tool_result_formatter=candidate_formatter())
    assert [m.model_dump() for m in raw] == [m.model_dump() for m in selective]
    assert json.loads(selective[-1].content)["data"]["truncated"] is True


def test_planning_budget_reports_larger_incomplete_receipt():
    payload = json.loads(receipt())
    payload["data"]["truncated"] = True
    raw = json.dumps(payload, ensure_ascii=False)
    trace = {}
    formatter = candidate_formatter()
    with patch.object(agents, "stage_evidence", formatter):
        result = agents._limit_evidence_context(raw, len(raw), selection_trace=trace)
        assert result == raw
        assert trace["omitted_blocks"] == 0
        result = agents._limit_evidence_context(raw, len(raw) - 1, selection_trace=trace)
        assert result == ""
        assert trace["omitted_blocks"] == trace["oversized_blocks"] == 1
