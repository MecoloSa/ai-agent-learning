"""RRF 与混合候选入口的离线测试。

验证排名、无页码资料、同页不同块、分数隔离与版本冲突。
使用假向量库，不连接 Embedding、重排或聊天模型。
"""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from ai_agent.lexical_retrieval import CharBigramBM25
from ai_agent.rag import LearningRAG, RAGError
from ai_agent.retrieval_fusion import fuse_rrf_candidates


def make_doc(source, text, chunk=0, **metadata):
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "chunk_index": chunk,
            **metadata,
        },
    )


def test_rrf_combines_rankings_without_requiring_pages():
    a = make_doc("a.md", "A")
    b = make_doc("b.md", "B")
    c = make_doc("c.txt", "C")

    hits = fuse_rrf_candidates(
        [(a, 0.9), (b, 0.8)],
        [b, c],
    )

    assert [
        doc.metadata["source"] for doc in hits
    ] == ["b.md", "a.md", "c.txt"]

    assert hits[0].metadata["rrf_score"] == pytest.approx(
        1 / 62 + 1 / 61
    )
    assert hits[0].metadata["dense_score"] == 0.8
    assert hits[0].metadata["dense_rank"] == 2
    assert hits[0].metadata["lexical_rank"] == 1

    # 仅关键词召回的块没有向量分数，不能伪造。
    assert "dense_score" not in hits[2].metadata
    assert all(
        "relevance_score" not in doc.metadata
        for doc in hits
    )
    assert all(
        "rrf_score" not in doc.metadata
        for doc in [a, b, c]
    )


def test_rrf_keeps_distinct_chunks_on_same_pdf_page():
    a = make_doc("book.pdf", "概述", chunk=0, page=10)
    b = make_doc("book.pdf", "关键正文", chunk=1, page=10)

    hits = fuse_rrf_candidates([(a, 0.9)], [b])

    assert len(hits) == 2
    assert {
        doc.metadata["chunk_index"] for doc in hits
    } == {0, 1}


def test_rrf_rejects_conflicting_text_at_same_position():
    old = make_doc("a.md", "旧正文")
    new = make_doc("a.md", "新正文")

    with pytest.raises(ValueError, match="不一致"):
        fuse_rrf_candidates([(old, 0.9)], [new])


def test_rrf_rejects_duplicate_positions_within_one_branch():
    a = make_doc("a.md", "正文")

    with pytest.raises(ValueError, match="重复"):
        fuse_rrf_candidates([(a, 0.9), (a, 0.8)], [])


def test_rrf_handles_empty_candidates_and_limit():
    assert fuse_rrf_candidates([], []) == []

    with pytest.raises(ValueError, match="limit"):
        fuse_rrf_candidates([], [], limit=0)


def test_hybrid_entry_combines_branches_and_syncs_once():
    a = make_doc("a.md", "ReAct 概述")
    b = make_doc("b.md", "ReAct 终止条件")
    c = make_doc("c.txt", "ReAct 工具结果")

    rag = object.__new__(LearningRAG)
    sync_calls = []
    rag.sync = lambda: sync_calls.append(True)
    rag.store = SimpleNamespace(
        similarity_search_with_relevance_scores=(
            lambda **kwargs: [(a, 0.9), (b, 0.8)]
        )
    )

    index = CharBigramBM25([b, c])
    rag._get_lexical_index = lambda: index

    hits = rag.retrieve_hybrid_candidates("ReAct")

    assert [
        doc.metadata["source"] for doc in hits
    ] == ["b.md", "a.md", "c.txt"]
    assert len(sync_calls) == 1
    assert all(
        "rrf_score" not in doc.metadata
        for doc in index.documents
    )


def test_hybrid_entry_does_not_hide_keyword_index_failure():
    rag = object.__new__(LearningRAG)
    rag.sync = lambda: None
    rag.store = SimpleNamespace(
        similarity_search_with_relevance_scores=lambda **kwargs: []
    )

    def fail_index():
        raise RAGError("模拟关键词索引失败")

    rag._get_lexical_index = fail_index

    with pytest.raises(RAGError, match="模拟关键词索引失败"):
        rag.retrieve_hybrid_candidates("ReAct")