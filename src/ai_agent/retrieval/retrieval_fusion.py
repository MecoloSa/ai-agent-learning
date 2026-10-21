"""通用 RRF 候选融合：支持 PDF、MD、TXT 等具有块位置的资料。

只融合两路排名，不把 RRF 分数当作向量相似度或语义支持度。
返回新 Document，保留来源位置，不修改输入或提前按页去重。
"""

from __future__ import annotations

from math import isfinite

from langchain_core.documents import Document


def fuse_rrf_candidates(
    dense: list[tuple[Document, float]],
    lexical: list[Document],
    *,
    limit: int = 16,
) -> list[Document]:
    if limit < 1:
        raise ValueError("limit 必须大于 0")

    originals: dict[tuple[str, int], Document] = {}
    scores: dict[tuple[str, int], float] = {}
    ranks: dict[tuple[str, int], dict[str, int]] = {}
    dense_scores: dict[tuple[str, int], float] = {}

    branches = (
        ("dense", [(doc, float(score)) for doc, score in dense]),
        ("lexical", [(doc, None) for doc in lexical]),
    )

    for branch, items in branches:
        seen: set[tuple[str, int]] = set()

        for rank, (document, vector_score) in enumerate(
            items, start=1
        ):
            source = document.metadata.get("source")
            chunk = document.metadata.get("chunk_index")

            if (
                not isinstance(source, str)
                or not source.strip()
                or type(chunk) is not int
                or chunk < 0
            ):
                raise ValueError(
                    "融合候选缺少有效 source/chunk_index"
                )

            key = (source, chunk)
            if key in seen:
                raise ValueError("同一路召回出现重复块位置")
            seen.add(key)

            old = originals.get(key)
            if old is not None and (
                old.page_content != document.page_content
                or old.metadata.get("page")
                != document.metadata.get("page")
                or (
                    old.metadata.get("file_hash") is not None
                    and document.metadata.get("file_hash") is not None
                    and old.metadata["file_hash"]
                    != document.metadata["file_hash"]
                )
            ):
                # 同位置却是不同正文，可能是缓存或索引版本不一致。
                # 不能随意选一个版本，更不能拼接成虚假的证据。
                raise ValueError(
                    "两路候选的同一位置内容或版本不一致"
                )

            originals.setdefault(key, document)
            scores[key] = (
                scores.get(key, 0.0) + 1.0 / (60 + rank)
            )
            ranks.setdefault(key, {})[
                f"{branch}_rank"
            ] = rank

            if vector_score is not None:
                if not isfinite(vector_score):
                    raise ValueError(
                        "向量候选分数不是有限数值"
                    )
                dense_scores[key] = vector_score

    ordered = sorted(
        scores,
        key=lambda key: (-scores[key], key),
    )[:limit]

    result: list[Document] = []
    for rank, key in enumerate(ordered, start=1):
        metadata = dict(originals[key].metadata)

        # 清理可能遗留的查询评分；缺少向量分数时不能伪造。
        for name in (
            "relevance_score", "rerank_score", "retrieval_rank",
            "dense_score", "dense_rank", "lexical_rank",
            "rrf_score", "rrf_rank",
        ):
            metadata.pop(name, None)

        metadata.update(ranks[key])
        metadata.update(
            rrf_score=scores[key],
            rrf_rank=rank,
        )

        if key in dense_scores:
            metadata["dense_score"] = dense_scores[key]

        result.append(Document(
            page_content=originals[key].page_content,
            metadata=metadata,
        ))

    return result
