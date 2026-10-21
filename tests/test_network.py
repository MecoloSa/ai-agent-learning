"""连接策略测试：验证开关、代理、SDK 参数流向，不请求远端服务。"""

from types import SimpleNamespace

import pytest

from ai_agent import config
from ai_agent.infrastructure import network
from ai_agent.retrieval.embeddings import BatchedDashScopeEmbeddings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.mark.parametrize("value,expected", [(None, False), ("false", False), ("true", True)])
def test_ipv4_setting_defaults_off_and_can_be_enabled(monkeypatch, value, expected):
    monkeypatch.delenv("API_FORCE_IPV4", raising=False)
    if value is not None:
        monkeypatch.setenv("API_FORCE_IPV4", value)
    assert config.get_settings().api_force_ipv4 is expected


def test_invalid_ipv4_setting_fails_before_request(monkeypatch):
    monkeypatch.setenv("API_FORCE_IPV4", "maybe")
    with pytest.raises(RuntimeError, match="API_FORCE_IPV4"):
        config.get_settings()


@pytest.mark.parametrize("enabled", [False, True])
def test_factories_apply_ipv4_to_both_api_clients(monkeypatch, enabled):
    monkeypatch.setenv("API_FORCE_IPV4", str(enabled))
    monkeypatch.setattr(config, "_required_env", lambda name: "https://example.test" if name == "CHAT_BASE_URL" else "test-key")
    sentinel = object()
    monkeypatch.setattr(config, "get_ipv4_chat_client", lambda url: sentinel)
    monkeypatch.setattr(config, "ChatOpenAI", lambda **kw: kw)
    monkeypatch.setattr(config, "BatchedDashScopeEmbeddings", lambda **kw: kw)
    chat = config.create_llm()
    embedding = config.create_embeddings()
    assert (chat.get("http_client") is sentinel) is enabled
    assert embedding["force_ipv4"] is enabled
    assert chat["max_retries"] == 0


@pytest.mark.parametrize("bypass,expected_proxy", [(False, "http://proxy.test:8080"), (True, None)])
def test_chat_transport_preserves_proxy_and_no_proxy(monkeypatch, bypass, expected_proxy):
    transport_args = {}
    fake_client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(network, "getproxies", lambda: {"https": "http://proxy.test:8080"})
    monkeypatch.setattr(network, "proxy_bypass", lambda host: bypass)
    monkeypatch.setattr(network.httpx, "HTTPTransport", lambda **kw: transport_args.update(kw) or object())
    monkeypatch.setattr(network.httpx, "Client", lambda **kw: fake_client)
    # 绕过缓存防止测试替身进入后续真实客户端缓存。
    assert network.get_ipv4_chat_client.__wrapped__("https://example.test/api") is fake_client
    assert transport_args == {"local_address": "0.0.0.0", "proxy": expected_proxy}


def test_embedding_adapter_binds_ipv4_for_direct_and_proxy_connections():
    adapter = network.IPv4Adapter()
    try:
        assert adapter.poolmanager.connection_pool_kw["source_address"] == ("0.0.0.0", 0)
        proxy = adapter.proxy_manager_for("http://proxy.test:8080")
        assert proxy.connection_pool_kw["source_address"] == ("0.0.0.0", 0)
    finally:
        adapter.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_embedding_keeps_query_document_semantics_and_batching(monkeypatch, enabled):
    import dashscope
    calls = []
    session = object()
    monkeypatch.setattr("ai_agent.retrieval.embeddings.get_ipv4_embedding_session", lambda: session)
    def fake_call(**kw):
        calls.append(kw)
        texts = kw["input"]
        count = len(texts) if isinstance(texts, list) else 1
        return SimpleNamespace(status_code=200, output={"embeddings": [{"embedding": [1.0, 0.0]} for _ in range(count)]})
    monkeypatch.setattr(dashscope.TextEmbedding, "call", fake_call)
    embeddings = BatchedDashScopeEmbeddings(model="test-model", api_key="test-key", batch_size=2, max_retries=1, force_ipv4=enabled)
    assert embeddings.embed_query("question") == [1.0, 0.0]
    assert len(embeddings.embed_documents(["a", "b", "c"])) == 3
    assert [kw["text_type"] for kw in calls] == ["query", "document", "document"]
    assert [len(kw["input"]) for kw in calls[1:]] == [2, 1]
    assert all((kw.get("session") is session) is enabled for kw in calls)


def test_session_client_propagates_failure_without_extra_retry():
    calls = []
    def fail(**kw):
        calls.append(kw)
        raise TimeoutError("test timeout")
    client = network.SessionEmbeddingClient(SimpleNamespace(call=fail), object())
    with pytest.raises(TimeoutError):
        client.call(model="test", input="question", text_type="query")
    assert len(calls) == 1
