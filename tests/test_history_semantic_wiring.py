"""历史语义接线候选：全部使用人工向量和固定模型，不请求网络。"""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import ai_agent.config as config
import ai_agent.main as app
from ai_agent.persistence.history_readback import HistoryReadAccess
from ai_agent.persistence.history_session import ArchiveCapture
from ai_agent.persistence.memory_state import HistoryArchiveState
from ai_agent.retrieval.history_embedding_cache import HistoryEmbeddingCache


class FakeEmbedding:
    def __init__(self):
        self.documents = []
        self.queries = []
    def embed_documents(self, texts):
        self.documents.append(list(texts))
        return [[1.0, 0.0] if "通勤" in text else [0.0, 1.0] for text in texts]
    def embed_query(self, query):
        self.queries.append(query)
        return [1.0, 0.0]


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch):
    monkeypatch.setenv("HISTORY_RETRIEVAL_STRATEGY", "bm25")
    monkeypatch.delenv("HISTORY_SEMANTIC_MIN_SIMILARITY", raising=False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def prepare(tmp_path):
    capture = ArchiveCapture(tmp_path, None, True)
    capture.capture("通勤时听讲解，不方便看屏幕。", "已了解。")
    embedding = FakeEmbedding()
    factories = []
    def factory():
        factories.append(True)
        return embedding
    access = HistoryReadAccess(tmp_path, strategy="semantic", min_similarity=0.5, embedding_factory=factory)
    access.start_turn(capture.state)
    tools = {t.name: t for t in access.tools}
    return capture, embedding, factories, access, tools


def search(tools, query="路上用耳朵学东西"):
    return json.loads(tools["search_conversation_turns"].invoke({"query": query}))


def test_lazy_creation_and_cross_turn_incremental_embeddings(tmp_path):
    capture, embedding, factories, access, tools = prepare(tmp_path)
    tools["list_conversation_sources"].invoke({})
    tools["read_conversation_turn"].invoke({"source_index": 1, "turn_number": 1})
    assert factories == []
    first = search(tools)["data"]
    assert first["retrieval_strategy"] == "semantic" and first["fallback_reason"] is None
    assert first["matches"][0]["turn_number"] == 1
    assert first["embedding_activity"]["document_texts_encoded"] == 1
    access.start_turn(capture.state)
    second = search(tools)["data"]
    assert second["embedding_activity"]["document_texts_encoded"] == 0
    assert second["embedding_activity"]["document_cache_hits"] == 1
    capture.capture("新增阅读安排。", "已了解。")
    access.start_turn(capture.state)
    third = search(tools)["data"]
    assert third["embedding_activity"]["document_texts_encoded"] == 1
    assert embedding.documents == [["通勤时听讲解，不方便看屏幕。"], ["新增阅读安排。"]]
    assert len(factories) == 1 and len(embedding.queries) == 3


def test_cached_vectors_never_expand_scope_or_keep_old_source_number(tmp_path):
    capture, embedding, _, access, tools = prepare(tmp_path)
    other = ArchiveCapture(tmp_path, None, True)
    other.capture("阅读安排。", "已了解。")
    original = capture.state.sources[0]
    secondary = other.state.sources[0]
    search(tools)
    access.start_turn(HistoryArchiveState(sources=(secondary, original)))
    data = search(tools)["data"]
    assert data["matches"][0]["source_index"] == 2
    access.start_turn(HistoryArchiveState(sources=(secondary,)))
    assert search(tools)["data"]["matches"] == []
    assert sum("通勤" in text for batch in embedding.documents for text in batch) == 1


def test_semantic_empty_does_not_silently_switch_to_lexical(tmp_path):
    _, embedding, _, _, tools = prepare(tmp_path)
    embedding.embed_query = lambda _: [-1.0, 0.0]
    data = search(tools, "通勤")["data"]
    assert data["matches"] == [] and data["retrieval_strategy"] == "semantic"
    assert data["fallback_reason"] is None


def test_embedding_error_falls_back_with_safe_reason(tmp_path, caplog):
    _, embedding, _, _, tools = prepare(tmp_path)
    def fail(texts):
        raise RuntimeError("PRIVATE_ENDPOINT_AND_KEY")
    embedding.embed_documents = fail
    result = search(tools, "通勤")
    assert result["ok"] is True
    assert result["data"]["retrieval_strategy"] == "char_bigram_bm25"
    assert result["data"]["requested_strategy"] == "semantic"
    assert result["data"]["fallback_reason"] == "embedding_unavailable"
    assert result["data"]["matches"]
    assert "PRIVATE" not in json.dumps(result) + caplog.text


def test_missing_archive_is_error_before_embedding_not_fallback(tmp_path):
    capture, _, factories, _, tools = prepare(tmp_path)
    (tmp_path / (capture.state.sources[0].archive_id + ".sqlite3")).unlink()
    result = search(tools)
    assert result["ok"] is False and factories == []


def test_cache_deduplicates_and_rejects_invalid_whole_batch():
    embedding = FakeEmbedding()
    cache = HistoryEmbeddingCache(lambda: embedding)
    cache.embed_documents(["通勤", "通勤"])
    assert embedding.documents == [["通勤"]]
    embedding.embed_documents = lambda _: [[1.0, 0.0], [0.0, 0.0]]
    with pytest.raises(ValueError):
        cache.embed_documents(["甲", "乙"])
    assert len(cache._cache) == 1


def test_cache_eviction_is_bounded_and_never_uses_positions():
    cache = HistoryEmbeddingCache(FakeEmbedding)
    cache.embed_documents([str(i) for i in range(200)])
    cache.embed_documents(["新增"])
    assert len(cache._cache) == 200 and cache._chars <= 100000


@pytest.mark.parametrize("mode,threshold", [("unknown", "0.4"), ("semantic", None), ("semantic", "nan"), ("semantic", "inf"), ("semantic", "1.1")])
def test_invalid_settings_rejected(monkeypatch, mode, threshold):
    monkeypatch.setenv("HISTORY_RETRIEVAL_STRATEGY", mode)
    if threshold is not None:
        monkeypatch.setenv("HISTORY_SEMANTIC_MIN_SIMILARITY", threshold)
    with pytest.raises(RuntimeError):
        config.get_settings()


def test_real_interactive_entry_uses_selected_semantic_strategy(tmp_path, monkeypatch):
    path = Path(__file__).parent / "helpers/history_readback_online.py"
    spec = importlib.util.spec_from_file_location("semantic_wiring_online_helper", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    fixture, case = helper.load_case(helper.ROOT / "tests/fixtures/evaluation/history_paraphrase_online_case.json", "history-paraphrase-online-011")
    embedding = FakeEmbedding()
    settings = replace(config.get_settings(), history_retrieval_strategy="semantic", history_semantic_min_similarity=0.5)
    monkeypatch.setattr(app, "get_settings", lambda: settings)
    monkeypatch.setattr(app, "create_embeddings", lambda: embedding)
    monkeypatch.setattr(app, "create_llm", lambda **kw: object())
    monkeypatch.setattr(app, "create_llm_with_tools", lambda *a, **kw: object())
    def model(llm, messages, *, stage):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            name, args = "search_conversation_turns", {"query": "路上用耳朵学东西"}
        elif len(results) == 1:
            data = json.loads(results[-1].content)["data"]
            assert data["retrieval_strategy"] == "semantic"
            match = data["matches"][0]
            name, args = "read_conversation_turn", {"source_index": match["source_index"], "turn_number": match["turn_number"]}
        else:
            data = json.loads(results[-1].content)["data"]
            return AIMessage(content=data["user_text"] + data["citation"])
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(len(results)), "type": "tool_call"}])
    monkeypatch.setattr(app, "invoke_llm_response", model)
    row = helper.run_case(fixture, case, tmp_path / "results.jsonl", helper.manifest(helper.DEFAULT_CASES))
    assert row["business_state_unchanged"] and row["source_unchanged"] and row["inherited_scope_unchanged"]
    assert row["observations"]["exact_target_quote_in_answer"] is True
