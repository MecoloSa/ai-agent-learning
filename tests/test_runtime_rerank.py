"""业务重排接入测试。

使用假向量候选和假重排接口，验证同页片段选择、失败回退及关闭开关。
不连接 Chroma、Embedding 或真实模型。
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import ai_agent.rag as rag_module


@pytest.fixture
def runtime_rag(monkeypatch):
    settings = replace(
        rag_module.get_settings(),
        rag_rerank_candidate_limit=2,
        rag_rerank_backend="dashscope",
        embedding_api_key="offline-test",
    )
    monkeypatch.setattr(rag_module, "get_settings", lambda: settings)
    monkeypatch.setenv(
        "NORMAL_DASHSCOPE_URL",
        "https://example.com/compatible-mode/v1",
    )

    candidates = [
        (Document(
            page_content="同页较泛的内容",
            metadata={"source": "book.pdf", "page": 1, "chunk_index": 0},
        ), 0.9),
        (Document(
            page_content="同页直接回答问题的内容",
            metadata={"source": "book.pdf", "page": 1, "chunk_index": 1},
        ), 0.8),
    ]
    rag = object.__new__(rag_module.LearningRAG)
    rag.store = SimpleNamespace(
        similarity_search_with_relevance_scores=lambda **_: candidates
    )
    rag.sync = lambda: None
    return rag


def retrieve(rag, *, rerank):
    return rag.retrieve(
        "问题",
        k=1,
        min_relevance=0.3,
        relative_margin=1,
        min_hits=1,
        rerank=rerank,
    )


def test_rerank_runs_before_pdf_page_dedup(runtime_rag, monkeypatch):
    def fake_rerank(query, documents, **kwargs):
        assert len(documents) == 2
        return SimpleNamespace(
            documents=[
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "rerank_score": 0.9},
                )
                for doc in reversed(documents)
            ],
            total_tokens=12,
        )

    monkeypatch.setattr(rag_module, "rerank_candidates", fake_rerank)

    hits = retrieve(runtime_rag, rerank=True)
    assert hits[0].metadata["chunk_index"] == 1
    assert hits[0].metadata["relevance_score"] == 0.8


@pytest.mark.parametrize("error", [TimeoutError(), ValueError()])
def test_rerank_failure_restores_dense_order(
    runtime_rag, monkeypatch, error
):
    def failing_rerank(*args, **kwargs):
        raise error

    monkeypatch.setattr(rag_module, "rerank_candidates", failing_rerank)

    hits = retrieve(runtime_rag, rerank=True)
    assert hits[0].metadata["chunk_index"] == 0


def test_disabled_rerank_does_not_call_service(runtime_rag, monkeypatch):
    calls = []

    def unexpected_call(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("不应调用")

    monkeypatch.setattr(rag_module, "rerank_candidates", unexpected_call)

    assert retrieve(runtime_rag, rerank=False)[0].metadata["chunk_index"] == 0
    assert calls == []


def test_dense_chat_rerank_preserves_vector_scores_without_rerank_scores(runtime_rag, monkeypatch):
    settings = replace(rag_module.get_settings(), rag_rerank_backend="chat")
    monkeypatch.setattr(rag_module, "get_settings", lambda: settings)
    monkeypatch.delenv("NORMAL_DASHSCOPE_URL", raising=False)

    def chat_order(query, docs, **options):
        assert options["backend"] == "chat"
        assert options["model"] == "deepseek-v4.1-flash"
        return SimpleNamespace(documents=[Document(page_content=d.page_content,
            metadata={**d.metadata, "rerank_rank": i, "rerank_backend": "chat"})
            for i, d in enumerate(reversed(docs), 1)], total_tokens=20)

    monkeypatch.setattr(rag_module, "rerank_candidates", chat_order)
    hits = retrieve(runtime_rag, rerank=True)
    assert hits[0].metadata["chunk_index"] == 1
    assert hits[0].metadata["relevance_score"] == .8
    assert hits[0].metadata["rerank_rank"] == 1
    assert "rerank_score" not in hits[0].metadata
