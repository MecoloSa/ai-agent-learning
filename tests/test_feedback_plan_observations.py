"""记录实际执行的候选；拒绝、非法候选和未消费回复不能混为一谈。"""
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def replay():
    runner = importlib.import_module("ai_agent.evaluation.feedback_plan_runner")
    from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases
    root = Path(__file__).parent / "fixtures/evaluation"
    cases = load_feedback_plan_cases(root / "feedback_plan_cases.jsonl")
    replies = json.loads((root / "feedback_plan/offline_responses.json").read_text(encoding="utf-8"))
    return runner, cases, replies


def test_declined_candidate_is_retained_separately_from_saved_plan(replay, tmp_path):
    runner, cases, replies = replay
    case = cases[1]
    result = runner.run_case(case, replies[case.case_id], tmp_path)
    change = result["plan_changes"][0]
    assert result["status"] == "completed"
    assert change["turn"] == 1
    assert change["decision"] == "cancelled"
    assert change["confirmation_reason"] == "user_declined"
    assert "总计60分钟" in change["proposed_plan"]
    from ai_agent.persistence.state_store import load_learning_state
    saved = load_learning_state(Path(result["artifacts"]) / "current.json")
    assert saved.plan_markdown == change["base_plan"]
    assert change["proposed_plan"] != change["base_plan"]
    assert result["turns"] == [{"turn": 1, "user_input": case.inputs[0], "response": replies[case.case_id][0]}]
    assert all(c["value"] is None for c in result["checks"] if c["category"] != "control")


def test_discussion_has_response_but_no_plan_change(replay, tmp_path):
    runner, cases, replies = replay
    result = runner.run_case(cases[0], replies[cases[0].case_id], tmp_path)
    assert len(result["turns"]) == 1
    assert result["plan_changes"] == []


def test_unused_response_is_not_recorded_as_executed(replay, tmp_path):
    runner, cases, _ = replay
    result = runner.run_case(cases[0], ["实际回复", "未消费回复"], tmp_path)
    assert result["status"] == "script_mismatch"
    assert [t["response"] for t in result["turns"]] == ["实际回复"]


def test_missing_response_produces_no_turn_observation(replay, tmp_path):
    runner, cases, _ = replay
    result = runner.run_case(cases[0], [], tmp_path)
    assert result["status"] == "script_mismatch"
    assert result["turns"] == []


def test_accepted_candidate_records_real_decision(replay, tmp_path):
    runner, cases, _ = replay
    case = cases[1].model_copy(update={"inputs": ["给候选", "v", "y", "exit"]})
    result = runner.run_case(case, ["===NEW_PLAN_START===\n新计划。\n===NEW_PLAN_END==="], tmp_path)
    assert result["plan_changes"][0]["decision"] == "confirmed"
    assert result["business_state_unchanged"] is False


def test_invalid_candidate_is_not_reported_as_confirmed(replay, tmp_path):
    runner, cases, _ = replay
    result = runner.run_case(cases[0], ["===NEW_PLAN_START===\n\n===NEW_PLAN_END==="], tmp_path)
    assert result["status"] == "completed"
    change = result["plan_changes"][0]
    assert change["decision"] == "raised"
    assert change["error_type"] == "ValidationError"
    assert change["confirmation_reason"] is None
    assert result["plan_confirmations"] == []


def test_initial_context_and_rubric_are_snapshots(replay, tmp_path):
    runner, cases, _ = replay
    case = cases[0].model_copy(deep=True)
    result = runner.run_case(case, ["仅讨论"], tmp_path)
    assert result["initial_context"]["plan_markdown"]
    assert "conversation_memory" not in result["initial_context"]
    assert "material_binding" not in result["initial_context"]
    criterion = result["case_spec"]["checks"][0]["criterion"]
    case.checks[0].criterion = "后续编辑"
    assert result["case_spec"]["checks"][0]["criterion"] == criterion
    json.dumps(result, ensure_ascii=False)
