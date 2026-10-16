from __future__ import annotations

from langchain_core.documents import Document

from ai_agent.rag import LearningRAG


class FakeStore:
    def __init__(self, candidates):
        self.candidates = candidates

    def similarity_search_with_relevance_scores(self, query, k):
        return self.candidates[:k]


def make_rag(candidates) -> LearningRAG:
    """绕过真实 Chroma 和 Embedding，只测试检索决策逻辑。"""
    rag = object.__new__(LearningRAG)
    rag.store = FakeStore(candidates)
    rag.sync = lambda: None
    return rag


def test_retrieve_applies_absolute_and_softened_relative_threshold():
    rag = make_rag([
        (
            Document(
                page_content="最高分证据",
                metadata={"source": "a.md", "chunk_index": 0},
            ),
            0.90,
        ),
        (
            Document(
                page_content="第二条有效证据",
                metadata={"source": "b.md", "chunk_index": 0},
            ),
            0.75,
        ),
        (
            Document(
                page_content="低于绝对阈值",
                metadata={"source": "c.md", "chunk_index": 0},
            ),
            0.20,
        ),
    ])

    results = rag.retrieve(
        "Agent 工具调用",
        k=3,
        min_relevance=0.35,
        relative_margin=0.12,
        min_hits=2,
    )

    assert [doc.metadata["source"] for doc in results] == [
        "a.md",
        "b.md",
    ]
    assert results[0].metadata["relevance_score"] == 0.90


def test_retrieve_deduplicates_chunks_from_same_pdf_page():
    rag = make_rag([
        (
            Document(
                page_content="同一页的第一个块",
                metadata={
                    "source": "book.pdf",
                    "page": 10,
                    "chunk_index": 20,
                },
            ),
            0.90,
        ),
        (
            Document(
                page_content="同一页的第二个块",
                metadata={
                    "source": "book.pdf",
                    "page": 10,
                    "chunk_index": 21,
                },
            ),
            0.88,
        ),
    ])

    results = rag.retrieve(
        "RAG",
        k=5,
        min_relevance=0.35,
        relative_margin=1.0,
        min_hits=1,
    )

    assert len(results) == 1
    assert results[0].metadata["chunk_index"] == 20


def test_retrieve_many_round_robins_across_queries():
    rag = object.__new__(LearningRAG)

    goal_1 = Document(
        page_content="goal-1",
        metadata={"source": "goal.md", "chunk_index": 0},
    )
    goal_2 = Document(
        page_content="goal-2",
        metadata={"source": "goal.md", "chunk_index": 1},
    )
    implementation_1 = Document(
        page_content="implementation-1",
        metadata={"source": "impl.md", "chunk_index": 0},
    )

    hits = {
        "目标": [goal_1, goal_2],
        "实现": [implementation_1],
    }
    rag.retrieve = lambda *, query, **kwargs: hits[query]

    results = rag.retrieve_many(
        (
            ("goal", "目标"),
            ("implementation", "实现"),
        ),
        k_per_query=2,
        max_hits=3,
        min_relevance=0.35,
        relative_margin=0.12,
        min_hits=1,
    )

    assert [
        (doc.metadata["source"], doc.metadata["chunk_index"])
        for doc in results
    ] == [
        ("goal.md", 0),
        ("impl.md", 0),
        ("goal.md", 1),
    ]


def test_format_documents_stops_at_complete_chunk_boundary():
    rag = object.__new__(LearningRAG)
    documents = [
        Document(
            page_content="第一条完整证据",
            metadata={"source": "a.md", "chunk_index": 0},
        ),
        Document(
            page_content="第二条不应进入预算",
            metadata={"source": "b.md", "chunk_index": 0},
        ),
    ]

    first_only = rag.format_documents(
        documents,
        max_context_chars=50,
    )

    assert "第一条完整证据" in first_only
    assert "第二条不应进入预算" not in first_only

def test_retrieve_audit_explains_candidate_decisions():
    rag = make_rag([
        (Document(page_content="源码", metadata={
            "source": "notes.py", "chunk_index": 0,
        }), 0.95),
        (Document(page_content="目标页", metadata={
            "source": "book.pdf", "page": 4, "chunk_index": 1,
        }), 0.90),
        (Document(page_content="目标页另一块", metadata={
            "source": "book.pdf", "page": 4, "chunk_index": 2,
        }), 0.88),
        (Document(page_content="其他页", metadata={
            "source": "book.pdf", "page": 5, "chunk_index": 3,
        }), 0.80),
        (Document(page_content="低分页", metadata={
            "source": "book.pdf", "page": 6, "chunk_index": 4,
        }), 0.20),
    ])
    audit = []

    hits = rag.retrieve(
        "综合案例进阶",
        k=1,
        min_relevance=0.35,
        relative_margin=1.0,
        min_hits=1,
        exclude_suffixes={".py"},
        candidate_audit=audit,
    )

    assert [doc.metadata["page"] for doc in hits] == [4]
    assert [row["decision"] for row in audit] == [
        "reject_suffix",
        "selected",
        "reject_duplicate",
        "reject_k_limit",
        "reject_absolute",
    ]
    assert all("page_content" not in row for row in audit)