"""采用后允许计划变化，但任务/反馈等不得被重置；真实确认保存，固定假回复。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases
from ai_agent.evaluation import feedback_plan_runner as runner


CASE_FILE = Path(__file__).parent / "fixtures/evaluation/feedback_plan_adoption_cases.jsonl"
REPLY = "===NEW_PLAN_START===\n针对易混淆处比较例子10分钟，口述区别10分钟；总计20分钟。\n===NEW_PLAN_END==="


@pytest.fixture
def adopted(tmp_path):
    case = load_feedback_plan_cases(CASE_FILE)[0]
    source = case.initial_state_file.read_bytes()
    result = runner.run_case(case, [REPLY], tmp_path)
    assert case.initial_state_file.read_bytes() == source
    return case, result


def test_actual_adoption_persists_candidate_without_erasing_records(adopted):
    case, record = adopted
    assert record["status"] == "completed"
    assert record["business_state_unchanged"] is False
    assert record["source_unchanged"] is True
    before, after = record["initial_context"], record["final_context"]
    assert len(before["tasks"]) == 3 and len(before["feedback_records"]) == 1
    assert after["tasks"] == before["tasks"] and after["feedback_records"] == before["feedback_records"]
    assert after["context_revision"] == 2 and after["tasks_need_review"] is True
    assert after["plan_markdown"] == record["plan_changes"][0]["proposed_plan"]
    assert record["checks"][0]["value"] is True
    assert record["checks"][1]["value"] is None


@pytest.mark.parametrize("damage", ["erase_task", "clear_feedback", "reset_understanding", "restore_cancelled", "overwrite_initial", "wrong_plan", "wrong_goal", "extra_revision", "no_review_flag", "wrong_confirmation", "unchanged_source_false", "no_change_event"])
def test_known_adoption_corruption_is_false(adopted, damage):
    case, original = adopted
    record = deepcopy(original)
    final = record["final_context"]
    if damage == "erase_task": final["tasks"].pop()
    elif damage == "clear_feedback": final["feedback_records"] = []
    elif damage == "reset_understanding": final["tasks"][0]["understanding_status"] = "self_reported_clear"
    elif damage == "restore_cancelled": final["tasks"][2].update(is_cancelled=False, cancellation_reason="")
    elif damage == "overwrite_initial": final["initial_plan_markdown"] = final["plan_markdown"]
    elif damage == "wrong_plan": final["plan_markdown"] = "没有采用该候选"
    elif damage == "wrong_goal": final["learning_goal"] = "擅自改变目标"
    elif damage == "extra_revision": final["context_revision"] = 3
    elif damage == "no_review_flag": final["tasks_need_review"] = False
    elif damage == "wrong_confirmation": record["plan_confirmations"] = [{"accepted": False, "reason": "user_declined"}]
    elif damage == "unchanged_source_false": record["source_unchanged"] = False
    else: record["plan_changes"] = []
    assert runner.evaluate_checks(case, record)[0]["value"] is False


@pytest.mark.parametrize("missing", ["final_context", "initial_context", "plan_confirmations", "plan_changes", "source_unchanged"])
def test_missing_observations_are_unknown_not_inferred_from_y(adopted, missing):
    case, record = adopted
    record.pop(missing)
    assert runner.evaluate_checks(case, record)[0]["value"] is None


def test_decline_cannot_pass_adoption_rule(tmp_path):
    case = load_feedback_plan_cases(CASE_FILE)[0].model_copy(update={"inputs": ["给候选", "v", "n", "exit"]})
    record = runner.run_case(case, [REPLY], tmp_path)
    assert record["status"] == "completed" and record["business_state_unchanged"] is True
    assert record["checks"][0]["value"] is False


def test_completed_flag_without_snapshots_is_not_enough():
    case = load_feedback_plan_cases(CASE_FILE)[0]
    assert runner.evaluate_checks(case, {"status": "completed"})[0]["value"] is None


def test_review_packet_exposes_actual_final_state(adopted, tmp_path):
    from ai_agent.evaluation.feedback_plan_review import prepare_packets, _resolve_ref
    _, record = adopted
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    packet = prepare_packets(path)[0]
    assert packet["final_context"] == record["final_context"]
    assert _resolve_ref(packet, "/final_context/tasks/0/latest_feedback") == "阅读已经完成，但还是分不清两类停止。"
