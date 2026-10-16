"""聊天重排适配、配置路由和异常边界；全部使用假客户端。"""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from ai_agent.config import get_settings
from ai_agent.reranking import configured_rerank_options, parse_chat_rerank_order, rerank_candidates


def documents(n=6):
    return [Document(page_content=f"证据{i}", metadata={
        "source": "book.pdf", "page": i + 1, "chunk_index": i, "rerank_score": .8,
    }) for i in range(n)]


def fake_client(content, finish_reason="stop"):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id="fake", usage=SimpleNamespace(total_tokens=20), choices=[
            SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), captured


def test_chat_reranks_without_fabricated_scores_or_mutating_evidence():
    docs = documents()
    client, request = fake_client('{"order":[5,1,0,2,3]}')
    outcome = rerank_candidates("问题", docs, backend="chat", model="deepseek-v4.1-flash",
        api_key="chat-key", base_address="https://example.com/v1", request_timeout=5,
        force_ipv4=False, client=client)
    assert [d.metadata["chunk_index"] for d in outcome.documents] == [5, 1, 0, 2, 3, 4]
    assert [d.metadata["rerank_rank"] for d in outcome.documents] == list(range(1, 7))
    assert all("rerank_score" not in d.metadata for d in outcome.documents)
    assert docs[0].metadata["rerank_score"] == .8
    assert outcome.backend == "chat" and outcome.completed_missing_count == 1
    assert outcome.total_tokens == 20 and outcome.request_id == "fake"
    assert request["model"] == "deepseek-v4.1-flash"
    assert request["extra_body"] == {"enable_thinking": False}
    assert request["temperature"] == 0 and request["max_tokens"] == 160
    assert request["response_format"] == {"type": "json_object"}
    assert set(json.loads(request["messages"][1]["content"])) == {"question", "candidates"}


@pytest.mark.parametrize("content", [
    '{"order":[0,1,2,3,3]}', '{"order":[0,1,2,3,6]}', '{"order":[0,1]}',
    '{"order":[false,1,2,3,4]}', '{"order":[0.0,1,2,3,4]}', '[]', '{}', 'not-json',
])
def test_invalid_chat_order_is_rejected_for_runtime_fallback(content):
    with pytest.raises(ValueError):
        parse_chat_rerank_order(content, 6)


def test_small_pool_still_requires_all_candidates():
    assert parse_chat_rerank_order('{"order":[1,0]}', 2) == ([1, 0], 0)
    with pytest.raises(ValueError):
        parse_chat_rerank_order('{"order":[1]}', 2)


def test_truncated_response_rejected_even_if_json_looks_valid():
    client, _ = fake_client('{"order":[1,0]}', "length")
    with pytest.raises(ValueError, match="未正常完成"):
        rerank_candidates("问题", documents(2), backend="chat", model="deepseek-v4.1-flash",
            api_key="chat-key", base_address="https://example.com/v1", request_timeout=5,
            force_ipv4=False, client=client)


def test_backend_routes_only_selected_credentials_and_url(monkeypatch):
    settings = replace(get_settings(), chat_api_key="chat-key", chat_base_url="https://chat.example/v1",
                       embedding_api_key="normal-key", rag_rerank_backend="chat")
    monkeypatch.delenv("NORMAL_DASHSCOPE_URL", raising=False)
    chat = configured_rerank_options(settings)
    assert chat["model"] == settings.rag_rerank_chat_model
    assert chat["api_key"] == "chat-key" and chat["base_address"] == "https://chat.example/v1"
    monkeypatch.setenv("NORMAL_DASHSCOPE_URL", "https://normal.example/compatible-mode/v1")
    dedicated = configured_rerank_options(settings, backend="dashscope")
    assert dedicated["model"] == settings.rag_rerank_model
    assert dedicated["api_key"] == "normal-key"
    assert dedicated["base_address"] == "https://normal.example/api/v1"
    assert configured_rerank_options(settings, model="qwen3.6-flash")["model"] == "qwen3.6-flash"


def test_deepseek_default_and_invalid_backend_configuration(monkeypatch):
    for name in ["RAG_RERANK_ENABLED", "RAG_RERANK_BACKEND", "RAG_RERANK_CHAT_MODEL"]:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.rag_rerank_enabled is True
        assert settings.rag_rerank_backend == "chat"
        assert settings.rag_rerank_chat_model == "deepseek-v4.1-flash"
        monkeypatch.setenv("RAG_RERANK_BACKEND", "typo")
        get_settings.cache_clear()
        with pytest.raises(RuntimeError, match="RAG_RERANK_BACKEND"):
            get_settings()
    finally:
        get_settings.cache_clear()
