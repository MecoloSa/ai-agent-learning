"""关键词及融合探针的离线测试：不连接 Chroma、Embedding 或聊天模型。"""

import pytest
from langchain_core.documents import Document

from ai_agent.evaluation.lexical_probe import (
    CharBigramBM25,
    rrf_page_ids,
    rrf_candidate_documents,
)

def test_bm25_prefers_exact_chinese_heading():
    documents = [
        Document(
            page_content="第四部分 实战篇 综合案例",
            metadata={"source": "book.pdf", "page": 9, "chunk_index": 1},
        ),
        Document(
            page_content="第四部分 综合案例进阶 实践应用",
            metadata={"source": "book.pdf", "page": 4, "chunk_index": 2},
        ),
    ]
    retriever = CharBigramBM25(documents)

    assert retriever.search(
        "第四部分 综合案例进阶", k=2
    )[0].metadata["page"] == 4
    assert retriever.search("  ", k=2) == []


def test_rrf_can_restore_page_from_sparse_branch():
    dense_audit = [
        {
            "source": "book.pdf", "page": 9,
            "chunk_index": 1, "raw_rank": 1,
        },
        {
            "source": "book.pdf", "page": 4,
            "chunk_index": 2, "raw_rank": 13,
        },
    ]
    sparse_hits = [
        Document(
            page_content="第四部分 综合案例进阶",
            metadata={"source": "book.pdf", "page": 4, "chunk_index": 2},
        ),
    ]

    assert rrf_page_ids(
        dense_audit, sparse_hits, k=2
    )[0] == "book.pdf#page=4"

def test_rrf_candidates_preserve_text_and_existing_page_ranking():
    """准备正文后，页级排名应与旧接口一致，原始文档不应被污染。"""
    documents = [
        Document(
            page_content="overview",
            metadata={"source": "book.pdf", "page": 9, "chunk_index": 1},
        ),
        Document(
            page_content="target",
            metadata={"source": "book.pdf", "page": 4, "chunk_index": 2},
        ),
        Document(
            page_content="target continuation",
            metadata={"source": "book.pdf", "page": 4, "chunk_index": 3},
        ),
    ]
    audit = [
        {"source": "book.pdf", "page": 9, "chunk_index": 1, "raw_rank": 1},
        {"source": "book.pdf", "page": 4, "chunk_index": 2, "raw_rank": 13},
    ]
    sparse_hits = [documents[1], documents[2]]

    candidates = rrf_candidate_documents(
        audit, sparse_hits, documents=documents
    )
    pages = list(dict.fromkeys(
        f"{doc.metadata['source']}#page={doc.metadata['page']}"
        for doc in candidates
    ))

    assert pages == rrf_page_ids(audit, sparse_hits, k=8)
    assert candidates[0].page_content == "target"
    assert len(candidates) == 3 and len(pages) == 2
    assert candidates[0].metadata["rrf_rank"] == 1
    assert all("rrf_rank" not in doc.metadata for doc in documents)


def test_rrf_candidates_reject_missing_text():
    """索引位置找不到正文时，应暴露问题，避免静默改变候选池。"""
    audit = [
        {"source": "book.pdf", "page": 4, "chunk_index": 2, "raw_rank": 1},
    ]
    with pytest.raises(ValueError, match="正文"):
        rrf_candidate_documents(audit, [], documents=[])