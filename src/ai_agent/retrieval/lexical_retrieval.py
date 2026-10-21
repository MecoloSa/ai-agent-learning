"""可复用的轻量 BM25 关键词检索，不连接模型或读取磁盘。

从评测探针提取，保留中文双字词、BM25 公式及同分排序规则。
一次构建后复用倒排索引；资料正文变化时，应由调用方重建实例。
这里只返回词面相关的候选，不证明候选支持答案，也不生成引用。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from langchain_core.documents import Document


def _terms(text: str) -> list[str]:
    """中文取相邻双字，英文、数字、下划线保留整词。"""
    result: list[str] = []
    for part in re.findall(
        r"[a-z0-9_]+|[\u3400-\u9fff]+", text.casefold()
    ):
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            result.extend(
                part[index:index + 2]
                for index in range(len(part) - 1)
            )
            if len(part) == 1:
                result.append(part)
        else:
            result.append(part)
    return result


class CharBigramBM25:
    """对传入的资料块建立内存索引；不要求 PDF 页码元数据。"""

    def __init__(self, documents: list[Document]) -> None:
        # 复制列表，避免调用方 append/remove 改变索引的位置对应关系。
        # Document 正文仍视为只读；修改资料后需要重新构建。
        self.documents = list(documents)
        self.term_counts = [
            Counter(_terms(doc.page_content))
            for doc in self.documents
        ]

        self.doc_frequency: Counter[str] = Counter()
        self._postings: dict[str, set[int]] = defaultdict(set)
        self._lengths = [
            sum(counts.values()) for counts in self.term_counts
        ]

        for index, counts in enumerate(self.term_counts):
            self.doc_frequency.update(counts.keys())
            for term in counts:
                self._postings[term].add(index)

        self.average_length = (
            sum(self._lengths) / len(self._lengths)
            if self._lengths else 1.0
        ) or 1.0

    def search(self, query: str, *, k: int) -> list[Document]:
        if k < 1:
            raise ValueError("k 必须大于 0")

        query_terms = set(_terms(query))
        candidate_indices: set[int] = set()

        for term in query_terms:
            candidate_indices.update(
                self._postings.get(term, ())
            )

        # 倒排索引 = 词 -> 包含该词的资料块编号。
        # 零匹配资料块的旧 BM25 分数恒为 0，跳过不改变排名。
        ranked: list[tuple[float, int]] = []
        document_count = len(self.documents)

        for index in sorted(candidate_indices):
            counts = self.term_counts[index]
            length = self._lengths[index]
            score = 0.0

            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue

                frequency_in_docs = self.doc_frequency[term]
                idf = math.log1p(
                    (document_count - frequency_in_docs + 0.5)
                    / (frequency_in_docs + 0.5)
                )

                # 保留旧公式，不在模块迁移时同时调整检索参数。
                score += (
                    idf * frequency * 2.2
                    / (
                        frequency
                        + 1.2 * (
                            0.25
                            + 0.75 * length / self.average_length
                        )
                    )
                )

            if score > 0:
                ranked.append((score, index))

        # 同分保持输入顺序，便于与旧实验结果确定性比较。
        # 动态评估：先验收候选顺序不变；高频词可能命中全库，
        # 因而不能宣称所有查询都会明显加速。
        ranked.sort(key=lambda item: (-item[0], item[1]))

        return [
            self.documents[index]
            for _, index in ranked[:k]
        ]
