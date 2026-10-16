import importlib.util
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

spec = importlib.util.spec_from_file_location("chat_rerank_experiment", Path(__file__).with_name("run.py"))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_valid_order_preserves_permutation():
    assert experiment.validate_order('{"order":[2,0,1]}', 3) == [2, 0, 1]


@pytest.mark.parametrize("models,prefix", [(["unknown"], ""), ([], ""), (["qwen3.6-flash"], "../outside")])
def test_followup_rejects_unknown_model_and_unsafe_output_before_any_request(models, prefix):
    with pytest.raises(ValueError):
        experiment.run(2, False, 0, models=models, output_prefix=prefix)


def test_partial_order_only_appends_missing_candidates():
    assert experiment.complete_partial_order('{"order":[5,0,3,2,1]}', 7) == ([5, 0, 3, 2, 1, 4, 6], 2)


@pytest.mark.parametrize("order", [[0, 1, 2, 3, 3], [0, 1, 2, 3, 8], [0, 1], [False, 1, 2, 3, 4]])
def test_unsafe_partial_order_still_rejected(order):
    with pytest.raises(ValueError):
        experiment.complete_partial_order(json.dumps({"order": order}), 7)


@pytest.mark.parametrize("order", [[0, 0], [0], [0, 2], [False, 1], [0.0, 1], "01"])
def test_invalid_order_cannot_silently_corrupt_evidence(order):
    with pytest.raises(ValueError):
        experiment.validate_order(json.dumps({"order": order}), 2)


def test_page_metrics_deduplicate_chunks_before_cutoff():
    docs = [Document(page_content="", metadata={"source": "book", "page": page})
            for page in [1, 1, 1, 1, 1, 2]]
    result = experiment.quality(docs, {"gold_ids": ["book#page=2"], "acceptable_ids": ["book#page=1", "book#page=2"]})
    assert result["recall5"] == 1
    assert result["mrr5"] == .5
    assert result["acceptable_hit1"] is True
    assert result["hit1"] is False


def test_failures_remain_in_quality_and_latency_denominators(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "HERE", tmp_path)
    rows = [{"model": "fake", "split": "holdout", "status": status, "latency_ms": latency,
             "hit1": hit, "acceptable_hit1": hit, "mrr5": hit, "recall5": hit, "recall8": hit}
            for status, latency, hit in [("ok", 100, 1), ("fallback", 45000, 0)]]
    summary = experiment.summarize(rows)["fake/all"]
    assert summary["requests"] == 2
    assert summary["successes"] == 1
    assert summary["hit1"] == .5
    assert summary["p95_ms"] == 45000
