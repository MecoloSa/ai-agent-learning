import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def helper():
    path = Path(__file__).parent / "helpers/history_strategy_comparison.py"
    spec = importlib.util.spec_from_file_location("strategy_comparison_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nonempty_lexical_can_block_correct_dense_candidate_and_rrf_can_promote_wrong_match(helper):
    fixture = {"sources": [{"through_turn": 2, "user_turns": ["可以写代码", "通勤听讲解"]}]}
    docs = helper.documents_from_fixture(fixture)
    result = helper.compare(docs, [[0.0, 1.0], [1.0, 0.0]], [1.0, 0.0], "可以听课")
    p = result["policies"]
    assert p["bm25"][0]["turn_number"] == 1
    assert p["dense"][0]["turn_number"] == 2
    assert p["empty_fallback"][0]["turn_number"] == 1
    assert p["hybrid_rrf"][0]["turn_number"] == 1
    assert result["query_embedding_needed"]["empty_fallback"] is False


def test_negative_dense_neighbour_requires_filtering(helper):
    docs = helper.documents_from_fixture({"sources": [{"through_turn": 1, "user_turns": ["学习"]}]})
    raw = helper.compare(docs, [[1.0, 0.0]], [0.0, 1.0], "火山")
    filtered = helper.compare(docs, [[1.0, 0.0]], [0.0, 1.0], "火山", dense_min=0.3)
    assert raw["policies"]["dense"]
    assert all(not hits for hits in filtered["policies"].values())


def test_preflight_creates_no_model_or_files(helper, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(helper, "create_embeddings", lambda: pytest.fail("预检不能调用模型"))
    output = tmp_path / "result.json"
    assert helper.main(["--output", str(output)]) == 0
    assert not output.exists()
    assert "preflight_only" in capsys.readouterr().out


def test_frozen_replay_reranks_without_regenerating_embeddings(helper, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import ai_agent.retrieval.reranking as reranking
    docs = helper.documents_from_fixture({"sources": [{"through_turn": 1, "user_turns": ["原话"]}]})
    frozen = {"embedding_model": "stub", "document_embedding_ms": 1,
              "documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
              "document_vectors": [[1.0, 0.0]], "queries": [
                  {"case": {"case_id": ident, "query": "查询", "targets": []}, "fixture": "stub", "query_embedding_ms": 1, "vector": [1.0, 0.0]}
                  for ident in ("coverage-010", "semantic-followup-001")]}
    source = tmp_path / "frozen.json"
    source.write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(helper, "create_embeddings", lambda: pytest.fail("重放不应生成向量"))
    monkeypatch.setattr(helper, "get_settings", lambda: object())
    monkeypatch.setattr(reranking, "configured_rerank_options", lambda *a, **kw: {"model": "stub"})
    calls = []
    def rerank(query, candidates, **kwargs):
        calls.append(query)
        return SimpleNamespace(documents=candidates, total_tokens=42, backend="chat")
    monkeypatch.setattr(reranking, "rerank_candidates", rerank)
    output = tmp_path / "results.json"
    assert helper.main(["--replay", str(source), "--rerank", "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(calls) == 2 and result["query_embedding_client_calls_this_run"] == 0
    assert all(r["status"] == "completed" for r in result["chat_rerank_probe"])
