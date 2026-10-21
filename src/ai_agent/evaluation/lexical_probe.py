"""离线关键词与 RRF 融合探针。

BM25 复用业务基础模块；本文件仅保留现有 PDF 评测的融合逻辑。
在线混合召回仍需单独接入，不能直接依赖这里的页级评测假设。
"""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document

# 保留旧导入路径；已有评测入口和测试不必同时迁移。
from ai_agent.retrieval.lexical_retrieval import CharBigramBM25, _terms


def _rank_rrf_chunks(
    dense_audit: list[dict[str, object]],
    sparse_hits: list[Document],
) -> list[tuple[str, int, int, float]]:
    """内部排序结果为 (source, chunk_index, page, rrf_score)。"""
    scores: dict[tuple[str, int], float] = defaultdict(float)
    pages: dict[tuple[str, int], int] = {}

    def add(source: str, page: int, chunk: int, rank: int) -> None:
        key = (source, chunk)
        scores[key] += 1.0 / (60 + rank)
        pages[key] = page

    for row in dense_audit:
        if row["page"] is not None and row["chunk_index"] is not None:
            add(
                str(row["source"]), int(row["page"]),
                int(row["chunk_index"]), int(row["raw_rank"]),
            )

    for rank, document in enumerate(sparse_hits, start=1):
        metadata = document.metadata
        if metadata.get("page") is not None:
            add(
                str(metadata["source"]), int(metadata["page"]),
                int(metadata["chunk_index"]), rank,
            )

    # 保留原有 RRF 公式与平分时的顺序，避免准备正文时改变基线。
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [
        (source, chunk, pages[(source, chunk)], scores[(source, chunk)])
        for source, chunk in ordered
    ]


def rrf_page_ids(
    dense_audit: list[dict[str, object]],
    sparse_hits: list[Document],
    *,
    k: int,
) -> list[str]:
    """兼容已有调用：按 chunk 融合，再按页去重并取前 k 页。"""
    return list(dict.fromkeys(
        f"{source}#page={page}"
        for source, _, page, _ in _rank_rrf_chunks(
            dense_audit, sparse_hits
        )
    ))[:k]


def rrf_candidate_documents(
    dense_audit: list[dict[str, object]],
    sparse_hits: list[Document],
    *,
    documents: list[Document],
) -> list[Document]:
    """给融合候选补回正文；完整候选池将作为重排输入。"""
    by_key = {
        (
            str(doc.metadata["source"]),
            int(doc.metadata["chunk_index"]),
        ): doc
        for doc in documents
        if doc.metadata.get("page") is not None
        and doc.metadata.get("chunk_index") is not None
    }
    candidates: list[Document] = []

    for rank, (source, chunk, page, score) in enumerate(
        _rank_rrf_chunks(dense_audit, sparse_hits),
        start=1,
    ):
        original = by_key.get((source, chunk))
        if original is None or int(original.metadata["page"]) != page:
            # 不能悄悄丢候选，否则会把索引不一致误判成重排效果变化。
            raise ValueError(
                f"融合候选正文缺失或页码不一致："
                f"{source} chunk={chunk}"
            )

        # 新建对象和元数据，避免给共享索引文档写入本次查询的临时排名。
        candidates.append(Document(
            page_content=original.page_content,
            metadata={
                **original.metadata,
                "rrf_rank": rank,
                "rrf_score": score,
            },
        ))

    # 此处不先取前 k，也不按页合并：
    # 重排应有机会检查靠后的候选，以及同一页的不同内容片段。
    return candidates
