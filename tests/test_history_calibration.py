import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def helper():
    path = Path(__file__).parent / "helpers/history_calibration.py"
    spec = importlib.util.spec_from_file_location("history_calibration_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_prioritizes_negative_constraint_then_recall(helper):
    results = [dict(threshold=0.3, positive_hits=3, negative_with_candidates=1),
               dict(threshold=0.5, positive_hits=2, negative_with_candidates=0),
               dict(threshold=0.6, positive_hits=2, negative_with_candidates=0)]
    assert helper.choose(results) == 0.5


def test_holdout_never_participates_in_threshold_selection(helper, monkeypatch):
    fixture = {"sources": [{"through_turn": 1, "turns": ["合成原话"]}],
               "dev": [{"id": "dev", "query": "开发问题", "target": [1, 1]}],
               "holdout": [{"id": "holdout", "query": "验收问题", "target": None}]}
    frozen = {"fixture": fixture, "document_vectors": [[1.0, 0.0]], "query_vectors": {}}
    calls = []
    def score(index, cases, threshold):
        group = cases[0]["id"]
        calls.append((group, threshold))
        return dict(threshold=threshold, positive_hits=1, negative_with_candidates=int(threshold < 0.5), rows=[])
    monkeypatch.setattr(helper, "score", score)
    result = helper.evaluate(frozen)
    assert result["selected_threshold"] == 0.5
    assert [threshold for group, threshold in calls if group == "holdout"] == [0.5, 0.4]


def test_preflight_creates_no_client_or_output(helper, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(helper, "create_embeddings", lambda: pytest.fail("预检不能创建客户端"))
    output = tmp_path / "result.json"
    assert helper.main(["--output", str(output)]) == 0
    assert not output.exists()
    assert "preflight_only" in capsys.readouterr().out
