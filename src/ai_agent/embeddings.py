"""Embedding 模型适配器。"""

from __future__ import annotations

import logging
import os

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings

logger = logging.getLogger("ai_agent.embeddings")

# DashScope 当前模型要求单次请求不能超过 20 条，不放在 .env 中把用户批大小当成服务端上限
DASHSCOPE_MAX_BATCH_SIZE = 20

class BatchedDashScopeEmbeddings(Embeddings):
    """为 DashScopeEmbeddings 增加批处理能力。

    Chroma 建立索引时可能一次传入数百个 chunks。
    本适配器将它们切成每批最多 20 条，再合并所有向量。
    """
    def __init__(
            self,
            *,
            model: str,
            api_key: str,
            batch_size: int,
            max_retries: int,
    ) -> None:
        if not 1 <= batch_size <= DASHSCOPE_MAX_BATCH_SIZE:
            raise ValueError(
                "DashScope Embedding batch_size 必须在 1 到 20 之间，"
                f"当前值为 {batch_size}。"
            )

        self.batch_size = batch_size

        # 使用组合而不是继承，避免受到 DashScopeEmbeddings
        # 内部 Pydantic 配置的影响。
        self.client = DashScopeEmbeddings(
            model=model,
            dashscope_api_key=api_key,
            max_retries=max_retries,
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """批量生成文档向量，每次最多提交 20 个文本。"""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        total_batches = (
            len(texts) + self.batch_size - 1
        ) // self.batch_size

        logger.info(
            "开始生成 Embedding：chunks=%d batches=%d batch_size=%d",
            len(texts),
            total_batches,
            self.batch_size,
        )
        for batch_number, start in enumerate(
            range(0, len(texts), self.batch_size),
            start=1,
        ):
            batch = texts[start:start + self.batch_size]

            logger.debug(
                "Embedding 批次：batch=%d/%d chunks=%d",
                batch_number,
                total_batches,
                len(batch),
            )

            batch_vectors = self.client.embed_documents(batch)

            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    "DashScope 返回的向量数量与输入文本数量不一致："
                    f"输入 {len(batch)} 条，返回 {len(batch_vectors)} 条。"
                )

            all_vectors.extend(batch_vectors)
        logger.info(
            "Embedding 完成：chunks=%d vectors=%d",
            len(texts),
            len(all_vectors),
        )

        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        """生成检索问题的向量。查询只有一条，无需拆批。"""
        return self.client.embed_query(text)