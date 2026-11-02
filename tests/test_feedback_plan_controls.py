"""显式控制规则评分；不按案例编号、关键词或脚本中的n伪造拒绝。"""
import importlib
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def api():
    return importlib.import_module("ai_agent.evaluation.feedback_plan_runner")


@pytest.fixture
def case_factory():
    from ai_agent.evaluation.feedback_plan_cases import FeedbackPlanCase
    def make(rule="plan_declined", category="control"):
        return FeedbackPlanCase.model_validate({
            "case_id": "arbitrary-case", "split": "dev", "material_dir": "materials",
            "initial_state_file": "state.json", "inputs": ["给候选", "v", "n", "exit"],
            "checks": [{"check_id": "arbitrary-check", "category": category,
                        "criterion": "不能通过文字猜测规则", "basis": "测试依据", "rule": rule}],
        })
    return make


def observation(**changes):
    return {
        "status": "completed", "business_state_unchanged": True,
        "plan_confirmations": [{"accepted": False, "reason": "user_declined"}],
        **changes,
    }


def test_explicit_decline_and_unchanged_pass(api, case_factory):
    result = api.evaluate_checks(case_factory(), observation())[0]
    assert result["value"] is True
    assert result["check_id"] == "arbitrary-check"


@pytest.mark.parametrize("changes", [
    {"business_state_unchanged": False},
    {"plan_confirmations": []},
    {"plan_confirmations": [{"accepted": True, "reason": "user_confirmed"}]},
    {"plan_confirmations": [{"accepted": False, "reason": "user_not_confirmed"}]},
    {"plan_confirmations": [{"accepted": False, "reason": "input_ended"}]},
])
def test_known_control_violation_is_false(api, case_factory, changes):
    assert api.evaluate_checks(case_factory(), observation(**changes))[0]["value"] is False


@pytest.mark.parametrize("changes", [
    {"status": "script_mismatch"}, {"status": "failed"},
    {"business_state_unchanged": None}, {"business_state_unchanged": 1},
    {"plan_confirmations": None},
    {"plan_confirmations": [{"accepted": False}]},
    {"plan_confirmations": [{"accepted": 0, "reason": "user_declined"}]},
])
def test_missing_or_incomplete_evidence_is_unknown(api, case_factory, changes):
    assert api.evaluate_checks(case_factory(), observation(**changes))[0]["value"] is None


@pytest.mark.parametrize("unchanged", [True, False])
def test_state_rule_does_not_require_confirmation(api, case_factory, unchanged):
    assert api.evaluate_checks(case_factory("state_unchanged"), observation(
        business_state_unchanged=unchanged, plan_confirmations=[],
    ))[0]["value"] is unchanged


@pytest.mark.parametrize("category", ["control", "quality", "evidence"])
def test_no_rule_remains_unknown(api, case_factory, category):
    assert api.evaluate_checks(case_factory(None, category), observation())[0]["value"] is None


@pytest.mark.parametrize("rule,category", [
    ("made_up", "control"), ("state_unchanged", "quality"), ("plan_declined", "evidence"),
])
def test_schema_rejects_unknown_or_misplaced_rules(case_factory, rule, category):
    with pytest.raises(ValueError):
        case_factory(rule, category)


def test_real_decline_capture_and_seed_grades(api, tmp_path):
    from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases
    root = Path(__file__).parent / "fixtures/evaluation"
    cases = load_feedback_plan_cases(root / "feedback_plan_cases.jsonl")
    responses = json.loads((root / "feedback_plan/offline_responses.json").read_text(encoding="utf-8"))
    for index, case in enumerate(cases):
        result = api.run_case(case, responses[case.case_id], tmp_path)
        assert result["status"] == "completed"
        assert result["plan_confirmations"] == ([] if index == 0 else [
            {"accepted": False, "reason": "user_declined"},
        ])
        assert all(c["value"] is True for c in result["checks"] if c["category"] == "control")
        assert all(c["value"] is None for c in result["checks"] if c["category"] != "control")
        assert result["assessment"] == "not_evaluated"


def test_absent_legacy_confirmation_trace_is_not_inferred_from_n(api, case_factory):
    old_record = {"status": "completed", "business_state_unchanged": True}
    assert "n" in case_factory().inputs
    assert api.evaluate_checks(case_factory(), old_record)[0]["value"] is None


def test_cli_nonzero_on_control_false_even_when_replay_completed(api, tmp_path, monkeypatch):
    root = Path(__file__).parent / "fixtures/evaluation"
    monkeypatch.setattr(sys, "argv", [
        "runner", "--cases", str(root / "feedback_plan_cases.jsonl"),
        "--responses", str(root / "feedback_plan/offline_responses.json"),
        "--output", str(tmp_path / "result.jsonl"),
    ])
    def fake_run(case, responses, artifacts):
        return {"case_id": case.case_id, "status": "completed", "assessment": "not_evaluated",
                "checks": [{"category": "control", "value": False}]}
    monkeypatch.setattr(api, "run_case", fake_run)
    with pytest.raises(SystemExit) as exc:
        api.main()
    assert exc.value.code == 1
    rows = (tmp_path / "result.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
