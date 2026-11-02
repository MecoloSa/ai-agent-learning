import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def helper():
    path = Path(__file__).parent / "helpers/history_semantic_probe.py"
    spec = importlib.util.spec_from_file_location("history_semantic_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_does_not_create_client_or_output(helper, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(helper, "create_embeddings", lambda: pytest.fail("预检不能创建客户端"))
    output = tmp_path / "result.json"
    assert helper.main(["--output", str(output)]) == 0
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["scoped_user_turns"] == 5


def test_scope_and_negative_nearest_neighbour_are_visible(helper):
    fixture = {"sources": [{"through_turn": 1, "user_turns": ["范围内", "范围外"]}],
               "cases": [{"case_id": "negative", "group": "negative", "query": "无关", "targets": []}]}
    class Embedding:
        def embed_documents(self, texts):
            assert texts == ["范围内"]
            return [[1.0, 0.0]]
        def embed_query(self, text):
            return [0.0, 1.0]
    result = helper.probe(fixture, Embedding())
    row = result["rows"][0]
    assert row["bm25_top3"] == []
    assert row["dense_top3"][0]["cosine"] == 0.0
    assert row["dense_hit_at_1"] is None


@pytest.mark.parametrize("left,right", [([], []), ([0.0], [1.0]), ([1.0], [1.0, 2.0]), ([float('nan')], [1.0])])
def test_invalid_vectors_are_rejected(helper, left, right):
    with pytest.raises(ValueError):
        helper.cosine(left, right)
