"""人工评审辅助只检验来源、标注结构和引用定位，不代替语义判断。"""
import hashlib
import json
from pathlib import Path
import sys

import pytest

from ai_agent.evaluation import feedback_plan_review as review


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def batch(tmp_path):
    from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases
    from ai_agent.evaluation.feedback_plan_runner import run_case
    root = Path(__file__).parent / "fixtures/evaluation"
    cases = load_feedback_plan_cases(root / "feedback_plan_cases.jsonl")
    replies = json.loads((root / "feedback_plan/offline_responses.json").read_text(encoding="utf-8"))
    rows = [run_case(case, replies[case.case_id], tmp_path / "artifacts") for case in cases]
    results, reviews = tmp_path / "results.jsonl", tmp_path / "reviews.jsonl"
    write_rows(results, rows)
    packets = review.prepare_packets(results)
    write_rows(reviews, packets)
    return results, reviews, packets, rows


def test_prepare_keeps_bad_declined_candidate_and_leaves_quality_pending(batch):
    results, _, packets, _ = batch
    assert len(packets) == 3
    assert [len(p["checks"]) for p in packets] == [1, 1, 2]
    assert packets[1]["plan_changes"][0]["decision"] == "cancelled"
    assert "总计60分钟" in packets[1]["plan_changes"][0]["proposed_plan"]
    assert all(c["status"] == "pending" and c["value"] is None for p in packets for c in p["checks"])
    assert all(p["source_evidence_status"] == "not_collected" for p in packets)
    assert packets[0]["source_sha256"] == hashlib.sha256(results.read_bytes()).hexdigest()


def test_report_pending_is_not_unknown_or_success(batch):
    results, reviews, _, _ = batch
    before = results.read_bytes()
    report = review.summarize_reviews(results, reviews)
    assert report["manual_checks"]["quality"] == {"pending": 3, "true": 0, "false": 0, "unknown": 0}
    assert report["manual_checks"]["evidence"]["pending"] == 1
    assert report["annotation_status"] == "pending"
    assert report["model_quality_evaluated"] is False
    assert results.read_bytes() == before


def test_explicit_false_and_reviewed_unknown_are_counted_separately(batch):
    results, reviews, packets, _ = batch
    packets[1]["reviewer"] = "测试用评审人，并非真实人工标注"
    packets[1]["checks"][0].update(
        status="reviewed", value=False, reason="30+30=60，超过用户30分钟上限。",
        evidence_refs=["/turns/0/user_input", "/plan_changes/0/proposed_plan"],
    )
    packets[2]["reviewer"] = "测试用评审人"
    packets[2]["checks"][1].update(status="reviewed", reason="离线未采集资料证据，无法确认。")
    write_rows(reviews, packets)
    report = review.summarize_reviews(results, reviews)
    assert report["manual_checks"]["quality"]["false"] == 1
    assert report["manual_checks"]["evidence"]["unknown"] == 1
    assert report["model_quality_evaluated"] is False


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "changed"), ("case_id", "other"), ("case_spec", {}),
    ("initial_context", {}), ("turns", []), ("plan_changes", []), ("control_checks", []),
    ("execution_mode", "online"),
])
def test_changed_observations_or_identity_are_rejected(batch, field, value):
    results, reviews, packets, _ = batch
    packets[1][field] = value
    write_rows(reviews, packets)
    with pytest.raises(ValueError):
        review.summarize_reviews(results, reviews)


@pytest.mark.parametrize("updates", [
    {"value": True}, {"reason": "尚未评审但有理由"}, {"value": 1, "status": "reviewed"},
    {"value": "false", "status": "reviewed"}, {"criterion": "换个判据"},
    {"status": "passed"}, {"evidence_refs": ["/turns/99/response"]},
    {"evidence_refs": ["/checks/0/criterion"]},
])
def test_malformed_or_contradictory_annotations_are_rejected(batch, updates):
    results, reviews, packets, _ = batch
    packets[1]["checks"][0].update(updates)
    write_rows(reviews, packets)
    with pytest.raises(ValueError):
        review.summarize_reviews(results, reviews)


@pytest.mark.parametrize("missing", ["reviewer", "reason", "evidence_refs"])
def test_boolean_verdict_requires_reviewer_reason_and_observation(batch, missing):
    results, reviews, packets, _ = batch
    packets[1]["reviewer"] = "reviewer"
    packets[1]["checks"][0].update(status="reviewed", value=False, reason="超过时间上限",
                                  evidence_refs=["/plan_changes/0/proposed_plan"])
    if missing == "reviewer":
        packets[1]["reviewer"] = " "
    else:
        packets[1]["checks"][0][missing] = [] if missing == "evidence_refs" else " "
    write_rows(reviews, packets)
    with pytest.raises(ValueError):
        review.summarize_reviews(results, reviews)


def test_modified_result_bytes_invalidate_old_review_link(batch):
    results, reviews, _, _ = batch
    results.write_bytes(results.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        review.summarize_reviews(results, reviews)


@pytest.mark.parametrize("mode", ["duplicate", "missing", "extra"])
def test_review_run_coverage_must_match(batch, mode):
    results, reviews, packets, _ = batch
    if mode == "duplicate":
        packets[1] = packets[0]
    elif mode == "missing":
        packets.pop()
    else:
        packets.append(packets[0])
    write_rows(reviews, packets)
    with pytest.raises(ValueError):
        review.summarize_reviews(results, reviews)


def test_old_results_are_not_reconstructed_from_mutable_files(batch):
    results, _, _, rows = batch
    rows[0].pop("plan_changes")
    write_rows(results, rows)
    with pytest.raises(ValueError, match="重新回放"):
        review.prepare_packets(results)


def test_failed_run_can_be_exported_but_not_graded(batch):
    results, reviews, _, rows = batch
    rows[0].update(status="failed", initial_context=None)
    write_rows(results, rows)
    packets = review.prepare_packets(results)
    packets[0]["reviewer"] = "reviewer"
    packets[0]["checks"][0].update(status="reviewed", reason="不应判定")
    write_rows(reviews, packets)
    with pytest.raises(ValueError, match="未完成"):
        review.summarize_reviews(results, reviews)


def test_cli_prepare_report_and_no_overwrite(batch, tmp_path, monkeypatch):
    results, _, _, _ = batch
    output = tmp_path / "new/reviews.jsonl"
    argv = ["review", "prepare", "--results", str(results), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    review.main()
    before = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        review.main()
    assert exc.value.code == 2
    assert output.read_bytes() == before
    report_file = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["review", "report", "--results", str(results),
                                      "--reviews", str(output), "--output", str(report_file)])
    review.main()
    assert json.loads(report_file.read_text(encoding="utf-8"))["annotation_status"] == "pending"


def test_invalid_input_does_not_create_output(tmp_path, monkeypatch):
    results, output = tmp_path / "bad.jsonl", tmp_path / "output.jsonl"
    results.write_text("不是JSON", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["review", "prepare", "--results", str(results), "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        review.main()
    assert exc.value.code == 2
    assert not output.exists()


def test_pointer_rejects_negative_index_and_resolves_nested_fields():
    packet = {"turns": [{"response": "观察"}]}
    assert review._resolve_ref(packet, "/turns/0/response") == "观察"
    with pytest.raises(ValueError):
        review._resolve_ref(packet, "/turns/-1/response")
