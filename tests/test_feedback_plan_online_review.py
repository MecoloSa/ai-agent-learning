"""在线记录的人工评审契约；使用合成记录，不请求模型或提供实验标签。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ai_agent.evaluation import feedback_plan_review as review


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def online(tmp_path):
    from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases
    from ai_agent.evaluation.feedback_plan_runner import run_case
    root = Path(__file__).parent / "fixtures/evaluation"
    case = load_feedback_plan_cases(root / "feedback_plan_cases.jsonl")[1]
    row = run_case(case, ["===NEW_PLAN_START===\n阅读15分钟，口述15分钟。\n===NEW_PLAN_END==="], tmp_path / "artifacts")
    # 合成在线记录仅检验评审文件格式，不能作为真实模型结果。
    row["execution_mode"] = "online_model"
    manifest = {"schema_version": "feedback-plan-online-v1"}
    row["experiment"] = {"before": manifest, "after": deepcopy(manifest),
                         "conditions_unchanged": True, "condition_error_type": None}
    results, reviews = tmp_path / "results.jsonl", tmp_path / "reviews.jsonl"
    write_rows(results, [row])
    return results, reviews, row


def test_online_packet_keeps_experiment_and_pending_is_not_evaluated(online):
    results, reviews, row = online
    packets = review.prepare_packets(results)
    assert packets[0]["experiment"] == row["experiment"]
    assert packets[0]["execution_mode"] == "online_model"
    assert packets[0]["checks"][0]["value"] is None
    assert packets[0]["source_evidence_status"] == "not_collected"
    write_rows(reviews, packets)
    report = review.summarize_reviews(results, reviews)
    assert report["execution_mode"] == "online_model"
    assert report["model_quality_evaluated"] is False


def test_manual_boolean_label_is_reported_without_rewriting_source(online):
    results, reviews, _ = online
    before = results.read_bytes()
    packets = review.prepare_packets(results)
    packets[0]["reviewer"] = "测试标签，不是实验人工标注"
    packets[0]["checks"][0].update(status="reviewed", value=True, reason="分项15+15=30，满足上限。",
        evidence_refs=["/turns/0/user_input", "/plan_changes/0/proposed_plan"])
    write_rows(reviews, packets)
    report = review.summarize_reviews(results, reviews)
    assert report["manual_checks"]["quality"]["true"] == 1
    assert report["model_quality_evaluated"] is True
    assert report["annotation_status"] == "complete"
    assert report["assessment"] == "annotation_summary_only"
    assert results.read_bytes() == before


@pytest.mark.parametrize("unchanged", [False, None])
def test_changed_or_unknown_conditions_reject_boolean_label(online, unchanged):
    results, reviews, row = online
    row["experiment"]["conditions_unchanged"] = unchanged
    write_rows(results, [row])
    packets = review.prepare_packets(results)
    packets[0]["reviewer"] = "测试评审人"
    packets[0]["checks"][0].update(status="reviewed", value=True, reason="不能判定此运行条件",
        evidence_refs=["/plan_changes/0/proposed_plan"])
    write_rows(reviews, packets)
    with pytest.raises(ValueError, match="运行条件"):
        review.summarize_reviews(results, reviews)


def test_unknown_conditions_allow_explained_unknown(online):
    results, reviews, row = online
    row["experiment"]["conditions_unchanged"] = None
    write_rows(results, [row])
    packets = review.prepare_packets(results)
    packets[0]["reviewer"] = "测试评审人"
    packets[0]["checks"][0].update(status="reviewed", reason="前后条件未核实，暂不下布尔结论。")
    write_rows(reviews, packets)
    report = review.summarize_reviews(results, reviews)
    assert report["manual_checks"]["quality"]["unknown"] == 1
    assert report["model_quality_evaluated"] is False


@pytest.mark.parametrize("experiment", [None, {}, {"before": {}, "conditions_unchanged": "true"}])
def test_online_record_requires_valid_condition_metadata(online, experiment):
    results, _, row = online
    row["experiment"] = experiment
    write_rows(results, [row])
    with pytest.raises(ValueError):
        review.prepare_packets(results)


def test_editing_experiment_in_review_is_rejected(online):
    results, reviews, _ = online
    packets = review.prepare_packets(results)
    packets[0]["experiment"]["conditions_unchanged"] = False
    write_rows(reviews, packets)
    with pytest.raises(ValueError):
        review.summarize_reviews(results, reviews)


def test_offline_online_results_cannot_be_aggregated_together(online):
    results, _, row = online
    offline = deepcopy(row)
    offline.update(run_id="separate-offline-run", execution_mode="offline_stub")
    offline.pop("experiment")
    write_rows(results, [row, offline])
    with pytest.raises(ValueError, match="混合"):
        review.prepare_packets(results)


def test_offline_packet_format_remains_compatible(online):
    results, reviews, row = online
    row["execution_mode"] = "offline_stub"
    row.pop("experiment")
    write_rows(results, [row])
    packets = review.prepare_packets(results)
    assert "experiment" not in packets[0]
    write_rows(reviews, packets)
    report = review.summarize_reviews(results, reviews)
    assert report["execution_mode"] == "offline_stub"
    assert report["model_quality_evaluated"] is False
