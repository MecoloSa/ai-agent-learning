"""混合策略业务接入测试。

使用假候选和假重排，验证筛选、去重、回退与邻居元数据。
不连接向量库、Embedding 或真实重排模型。
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import ai_agent.rag as rag_module


@pytest.fixture
def hybrid_rag(monkeypatch):
    settings = replace(
        rag_module.get_settings(),
        rag_retrieval_strategy="hybrid",
        rag_rerank_backend="dashscope",
        rag_hybrid_min_rerank_score=0.5,
        embedding_api_key="offline-test",
    )
    monkeypatch.setattr(
        rag_module, "get_settings", lambda: settings
    )
    monkeypatch.setenv(
        "NORMAL_DASHSCOPE_URL",
        "https://example.com/compatible-mode/v1",
    )

    candidates = [
        Document(
            page_content="同页概述",
            metadata={
                "source": "book.pdf", "page": 1,
                "chunk_index": 0, "rrf_score": 0.03,
                "dense_score": 0.9,
            },
        ),
        Document(
            page_content="同页关键正文",
            metadata={
                "source": "book.pdf", "page": 1,
                "chunk_index": 1, "rrf_score": 0.02,
                "dense_score": 0.8,
            },
        ),
        Document(
            page_content="仅关键词召回的补充",
            metadata={
                "source": "notes.md",
                "chunk_index": 0, "rrf_score": 0.01,
            },
        ),
    ]

    rag = object.__new__(rag_module.LearningRAG)
    rag.retrieve_hybrid_candidates = (
        lambda *args, **kwargs: candidates
    )
    dense_calls = []

    def fake_dense(**kwargs):
        dense_calls.append(kwargs)
        return [Document(
            page_content="向量回退结果",
            metadata={
                "source": "fallback.md",
                "chunk_index": 0,
                "relevance_score": 0.8,
            },
        )]

    rag.retrieve = fake_dense

    def fake_rerank(query, documents, **kwargs):
        return SimpleNamespace(
            documents=[
                Document(
                    page_content=candidates[index].page_content,
                    metadata={
                        **candidates[index].metadata,
                        "rerank_score": score,
                    },
                )
                for index, score in [(1, 0.95), (0, 0.8), (2, 0.7)]
            ],
            total_tokens=10,
        )

    monkeypatch.setattr(
        rag_module, "rerank_candidates", fake_rerank
    )
    return rag, candidates, dense_calls


def test_hybrid_reranks_before_page_dedup(hybrid_rag):
    rag, _, dense_calls = hybrid_rag
    hits = rag.retrieve_with_strategy("问题", k=2, rerank=True)

    assert [
        doc.metadata["chunk_index"] for doc in hits
    ] == [1, 0]
    assert hits[0].metadata["source"] == "book.pdf"
    assert hits[1].metadata["source"] == "notes.md"
    assert hits[0].metadata["relevance_score"] == 0.8
    assert "relevance_score" not in hits[1].metadata
    assert dense_calls == []


def test_low_rerank_scores_return_empty_without_fallback(
    hybrid_rag, monkeypatch
):
    rag, candidates, dense_calls = hybrid_rag

    def low_scores(*args, **kwargs):
        return SimpleNamespace(
            documents=[
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "rerank_score": 0.1},
                )
                for doc in candidates
            ],
            total_tokens=10,
        )

    monkeypatch.setattr(
        rag_module, "rerank_candidates", low_scores
    )

    assert rag.retrieve_with_strategy("问题", rerank=True) == []
    assert dense_calls == []


@pytest.mark.parametrize("stage", ["index", "rerank"])
def test_failure_falls_back_without_retrying_rerank(
    hybrid_rag, monkeypatch, stage
):
    rag, _, dense_calls = hybrid_rag

    def fail(*args, **kwargs):
        raise TimeoutError("模拟失败")

    if stage == "index":
        rag.retrieve_hybrid_candidates = fail
    else:
        monkeypatch.setattr(
            rag_module, "rerank_candidates", fail
        )

    hits = rag.retrieve_with_strategy("问题", rerank=True)

    assert len(dense_calls) == 1
    assert dense_calls[0]["rerank"] is False
    assert hits[0].metadata["retrieval_strategy"] == "dense_fallback"


def test_disabled_rerank_uses_dense_fallback(hybrid_rag):
    rag, _, dense_calls = hybrid_rag
    hits = rag.retrieve_with_strategy("问题", rerank=False)

    assert dense_calls[0]["rerank"] is False
    assert hits[0].metadata["retrieval_fallback_reason"] == (
        "rerank_disabled"
    )


def test_explicit_dense_strategy_preserves_original_route(hybrid_rag):
    rag, _, dense_calls = hybrid_rag
    rag.retrieve_with_strategy(
        "问题", strategy="dense", rerank=True
    )
    assert dense_calls[0]["rerank"] is True


def test_chat_hybrid_rank_only_does_not_require_score_or_apply_dedicated_threshold(hybrid_rag, monkeypatch):
    rag, candidates, dense_calls = hybrid_rag
    settings = replace(rag_module.get_settings(), rag_rerank_backend="chat",
                       rag_hybrid_min_rerank_score=1)
    monkeypatch.setattr(rag_module, "get_settings", lambda: settings)
    monkeypatch.delenv("NORMAL_DASHSCOPE_URL", raising=False)

    def chat_order(query, docs, **options):
        assert options["backend"] == "chat"
        return SimpleNamespace(documents=[Document(page_content=candidates[index].page_content,
            metadata={**candidates[index].metadata, "rerank_rank": rank, "rerank_backend": "chat"})
            for rank, index in enumerate([1, 0, 2], 1)], total_tokens=20)

    monkeypatch.setattr(rag_module, "rerank_candidates", chat_order)
    hits = rag.retrieve_with_strategy("问题", k=2, rerank=True)
    assert [d.metadata["source"] for d in hits] == ["book.pdf", "notes.md"]
    assert all("rerank_score" not in d.metadata for d in hits)
    assert hits[0].metadata["rerank_rank"] == 1
    assert "relevance_score" not in hits[1].metadata
    assert dense_calls == []


def test_neighbor_expansion_preserves_hit_scores_only(hybrid_rag):
    rag, candidates, _ = hybrid_rag
    hit = rag.retrieve_with_strategy("问题", k=1, rerank=True)[0]

    rag.store = SimpleNamespace(
        get=lambda **kwargs: {
            "documents": [
                doc.page_content for doc in candidates[:2]
            ],
            "metadatas": [
                {
                    "source": "book.pdf",
                    "page": 1,
                    "chunk_index": index,
                }
                for index in range(2)
            ],
        }
    )

    expanded = rag.expand_neighbors([hit], window=1)
    actual_hit = next(
        doc for doc in expanded
        if doc.metadata["retrieval_role"] == "hit"
    )
    neighbor = next(
        doc for doc in expanded
        if doc.metadata["retrieval_role"] == "neighbor"
    )

    assert actual_hit.metadata["rerank_score"] == 0.95
    assert actual_hit.metadata["retrieval_strategy"] == "hybrid"
    assert "rerank_score" not in neighbor.metadata

    text = rag.format_documents(
        [actual_hit], max_context_chars=2000
    )
    assert "rrf_score=" in text
    assert "rerank_score=0.9500" in text
    assert "strategy=hybrid" in text
