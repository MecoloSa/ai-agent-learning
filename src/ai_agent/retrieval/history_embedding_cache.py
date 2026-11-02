"""历史消息向量的进程内缓存；不保存来源权限或正文位置。"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from hashlib import sha256

from langchain_core.embeddings import Embeddings
from ai_agent.retrieval.semantic_history import _normalize


class HistoryEmbeddingCache(Embeddings):
    def __init__(self, factory: Callable[[], Embeddings]):
        self._factory = factory
        self._client = None
        # 一个实例绑定一个客户端/模型，不与其他会话或模型共享缓存。
        self._cache = OrderedDict()
        self._chars = 0
        self.stats = {
            "document_requests": 0,
            "document_texts_encoded": 0,
            "document_cache_hits": 0,
            "query_requests": 0,
        }

    def _get_client(self):
        if self._client is None:
            self._client = self._factory()
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [sha256(text.encode("utf-8")).hexdigest() for text in texts]
        found, missing = {}, {}

        for key, text in zip(keys, texts):
            if key in self._cache:
                found[key] = self._cache[key][0]
                self._cache.move_to_end(key)
                self.stats["document_cache_hits"] += 1
            else:
                missing.setdefault(key, text)

        if missing:
            self.stats["document_requests"] += 1
            raw = self._get_client().embed_documents(list(missing.values()))
            if len(raw) != len(missing):
                raise ValueError("向量数量与待编码消息不一致")

            vectors = [_normalize(vector) for vector in raw]
            if len({len(vector) for vector in vectors}) != 1:
                raise ValueError("新消息向量维数不一致")

            # 全批校验成功后才写缓存，避免保存部分无效响应。
            for (key, text), vector in zip(missing.items(), vectors):
                found[key] = vector
                self._cache[key] = (vector, len(text))
                self._chars += len(text)
                self.stats["document_texts_encoded"] += 1

                while len(self._cache) > 200 or self._chars > 100000:
                    _, (_, chars) = self._cache.popitem(last=False)
                    self._chars -= chars

        # 返回独立列表；正文相同可复用向量，位置由本轮索引另行绑定。
        return [list(found[key]) for key in keys]

    def embed_query(self, text: str) -> list[float]:
        self.stats["query_requests"] += 1
        return self._get_client().embed_query(text)