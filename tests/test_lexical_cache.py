"""关键词索引缓存的离线测试。

使用假向量库，覆盖复用、更新、删除、更新失败和结果隔离。
真实临时文件用于验证现有 sync 的文件哈希变化检测，不连接模型。
"""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from ai_agent.retrieval.rag import LearningRAG, RAGError


class FakeStore:
    def __init__(self, documents=None):
        self.documents = list(documents or [])
        self.get_count = 0
        self.fail_add = False
        self.bad_payload = False

    def get(self, *, include):
        self.get_count += 1
        if self.bad_payload:
            return {
                "documents": ["ReAct"],
                "metadatas": [],
            }
        return {
            "documents": [
                doc.page_content for doc in self.documents
            ],
            "metadatas": [
                dict(doc.metadata) for doc in self.documents
            ],
        }

    def delete(self, *, where):
        self.documents = [
            doc
            for doc in self.documents
            if doc.metadata["source"] != where["source"]
        ]

    def add_documents(self, documents, *, ids):
        if self.fail_add:
            raise OSError("模拟向量更新失败")
        self.documents.extend(documents)


def make_rag(tmp_path, documents=None):
    # 绕过真实 Chroma、Embedding 初始化，但保留真实 sync。
    rag = object.__new__(LearningRAG)
    rag.root = tmp_path
    rag.manifest_path = tmp_path / "manifest.json"
    rag.store = FakeStore(documents)
    rag._lexical_index = None
    rag.splitter = SimpleNamespace(
        split_documents=lambda documents: documents
    )
    return rag


def test_cache_reuses_index_and_isolates_returned_metadata(tmp_path):
    (tmp_path / "a.md").write_text(
        "ReAct 工具调用", encoding="utf-8"
    )
    rag = make_rag(tmp_path)

    first = rag.retrieve_lexical("ReAct")
    first[0].metadata["rrf_score"] = 999

    second = rag.retrieve_lexical("ReAct")

    assert rag.store.get_count == 1
    assert "rrf_score" not in second[0].metadata


@pytest.mark.parametrize("action", ["modify", "add", "delete"])
def test_file_change_rebuilds_keyword_cache(tmp_path, action):
    path = tmp_path / "a.md"
    path.write_text("ReAct 旧内容", encoding="utf-8")
    rag = make_rag(tmp_path)

    assert rag.retrieve_lexical("ReAct")
    old_index = rag._lexical_index

    if action == "modify":
        path.write_text("Memory 新内容", encoding="utf-8")
    elif action == "add":
        (tmp_path / "b.md").write_text(
            "Memory 新增资料", encoding="utf-8"
        )
    else:
        path.unlink()

    hits = rag.retrieve_lexical("Memory")

    assert rag.store.get_count == 2
    assert rag._lexical_index is not old_index

    if action == "delete":
        assert hits == []
        assert rag.retrieve_lexical("ReAct") == []
    else:
        assert hits
        assert "Memory" in hits[0].page_content


def test_sync_failure_invalidates_old_cache_and_can_retry(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("ReAct 旧内容", encoding="utf-8")
    rag = make_rag(tmp_path)
    assert rag.retrieve_lexical("ReAct")

    path.write_text("Memory 新内容", encoding="utf-8")
    rag.store.fail_add = True

    with pytest.raises(OSError, match="模拟向量更新失败"):
        rag.retrieve_lexical("Memory")

    # 更新中途失败，旧缓存也必须已经失效。
    assert rag._lexical_index is None

    rag.store.fail_add = False
    assert rag.retrieve_lexical("Memory")
    assert rag.retrieve_lexical("ReAct") == []


def test_invalid_payload_does_not_publish_partial_cache(tmp_path):
    rag = make_rag(tmp_path)
    rag.store.bad_payload = True

    with pytest.raises(RAGError, match="数量不一致"):
        rag.retrieve_lexical("ReAct")

    assert rag._lexical_index is None

    rag.store.bad_payload = False
    assert rag.retrieve_lexical("ReAct") == []
    assert rag._lexical_index is not None


def test_suffix_exclusion_happens_before_top_k(tmp_path):
    documents = [
        Document(
            page_content="ReAct",
            metadata={"source": "a.py", "chunk_index": 0},
        ),
        Document(
            page_content="ReAct",
            metadata={"source": "b.md", "chunk_index": 0},
        ),
    ]
    rag = make_rag(tmp_path, documents)

    # 本例只验证候选过滤，跳过文件同步。
    rag.sync = lambda: None

    hits = rag.retrieve_lexical(
        "ReAct",
        k=1,
        exclude_suffixes={".PY"},
    )

    assert [doc.metadata["source"] for doc in hits] == ["b.md"]
