"""
检索评测指标工具：提供 recall@k 与 reciprocal rank 两个标准检索质量度量。

1. recall_at_k：前 K 条结果覆盖金标准文档的比例；
2. reciprocal_rank：首个正确结果排名的倒数，未命中返回 0。
"""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(
    ranked_ids: Sequence[str],  # 系统检索到的文档ID列表，按相关性从高到低排序
    gold_ids: set[str],         # 人工标注的"正确答案"——真正相关的文档ID集合
    k: int,                     # 只看前 K 条结果
) -> float:
    """前 K 条结果覆盖了多少人工标注的正确证据。"""
    if not gold_ids:
        raise ValueError("gold_ids 不能为空")
    if k < 1:
        raise ValueError("k 必须大于 0")

    retrieved = set(ranked_ids[:k])
    return len(retrieved & set(gold_ids)) / len(set(gold_ids))


def reciprocal_rank(
    ranked_ids: Sequence[str],
    gold_ids: set[str],
) -> float:
    """第一个正确证据排名的倒数；完全未命中时返回 0。"""
    if not gold_ids:
        raise ValueError("gold_ids 不能为空")

    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in gold_ids:
            return 1.0 / rank

    return 0.0