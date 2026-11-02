"""接收已限定范围的历史消息，建立小规模内存语义索引。"""
from __future__ import annotations

import math

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def _normalize(vector) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("向量为空或包含非有限数值")

    length = math.hypot(*values)
    if not math.isfinite(length) or length == 0:
        raise ValueError("向量长度无效")

    return tuple(value / length for value in values)


class SemanticHistoryIndex:
    """只负责语义排序；来源权限、范围过滤和原文回读由上层保证。"""

    def __init__(self, documents: list[Document], embedding: Embeddings):
        # 复制正文和位置元数据，防止外部修改破坏向量与消息的对应关系。
        self._documents = tuple(
            document.model_copy(deep=True) for document in documents
        )
        if (
            len(self._documents) > 200
            or sum(len(d.page_content) for d in self._documents) > 100000
        ):
            raise ValueError("语义索引超过历史检索预算")

        self._embedding = embedding
        texts = [document.page_content for document in self._documents]
        raw_vectors = embedding.embed_documents(texts) if texts else []

        if len(raw_vectors) != len(texts):
            raise ValueError("向量数量与消息数量不一致")

        self._vectors = tuple(_normalize(vector) for vector in raw_vectors)
        self._dimension = len(self._vectors[0]) if self._vectors else None

        if any(len(vector) != self._dimension for vector in self._vectors):
            raise ValueError("文档向量维数不一致")

    def search(
        self,
        query: str,
        *,
        k: int,
        min_similarity: float,
    ) -> list[tuple[Document, float]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询不能为空")

        if type(k) is not int or not 1 <= k <= 5:
            raise ValueError("k必须为1至5的整数")

        if (
            isinstance(min_similarity, bool)
            or not isinstance(min_similarity, (int, float))
            or not math.isfinite(min_similarity)
            or not -1 <= min_similarity <= 1
        ):
            raise ValueError("相似度阈值必须为-1至1之间的有限数值")

        if not self._documents:
            return []

        query_vector = _normalize(self._embedding.embed_query(query))
        if len(query_vector) != self._dimension:
            raise ValueError("查询与文档向量维数不一致")

        # 向量归一化后，点积等于余弦相似度；分数不是正确概率。
        ranked = []
        for index, vector in enumerate(self._vectors):
            score = sum(a * b for a, b in zip(query_vector, vector))
            score = max(-1.0, min(1.0, score))

            if score >= min_similarity:
                ranked.append((index, score))

        ranked.sort(key=lambda item: (-item[1], item[0]))

        # 返回副本，避免调用方添加临时评分时污染缓存中的原始消息。
        return [
            (self._documents[index].model_copy(deep=True), score)
            for index, score in ranked[:k]
        ]