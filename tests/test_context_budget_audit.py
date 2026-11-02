"""同类预算问题审计：固定文档、本地格式化，无模型或向量请求。"""
from langchain_core.documents import Document
import pytest
import ai_agent.retrieval.rag as module


def document(text, source):
    return Document(page_content=text, metadata={"source": source, "chunk_index": 0})


@pytest.mark.parametrize("budget", [80, 100, 150])
def test_overview_counts_separators_and_preserves_truncation_notice(tmp_path, monkeypatch, budget):
    paths = [tmp_path / "a.md", tmp_path / "b.md"]
    for path in paths:
        path.write_text("甲" * 300, encoding="utf-8")
    monkeypatch.setattr(module, "_load_file", lambda root, path: [document("甲" * 300, path.name)])
    rag = object.__new__(module.LearningRAG)
    rag.root = tmp_path
    text = rag.build_overview_text(max_chars=budget)
    assert len(text) <= budget
    assert "[source=a.md]" in text
    assert "[该文件的overview内容已截断]" in text
    for block in text.split("\n\n---\n\n"):
        assert block.startswith("[source=")


def test_overview_does_not_emit_notice_without_any_original_body(tmp_path, monkeypatch):
    path = tmp_path / "a.md"
    path.write_text("甲" * 300, encoding="utf-8")
    monkeypatch.setattr(module, "_load_file", lambda root, path: [
        document("甲" * 300, path.name), document("乙" * 300, path.name),
    ])
    rag = object.__new__(module.LearningRAG)
    rag.root = tmp_path
    text = rag.build_overview_text(max_chars=60)
    # 预算连“来源头＋至少一个原文字符＋完整截断提示”都放不下，不生成伪证据。
    assert "[source=" not in text
    assert "[该文件的overview内容已截断]" not in text


def test_retrieve_text_uses_canonical_formatter_budget_and_metadata():
    docs = [document("甲", "a.md"), document("乙", "b.md")]
    docs[0].metadata.update({"retrieval_strategy": "hybrid", "rerank_rank": 1})
    rag = object.__new__(module.LearningRAG)
    rag.retrieve_with_strategy = lambda **kwargs: docs
    rag.expand_neighbors = lambda hits, window: docs
    expected = rag.format_documents(docs, max_context_chars=1000)
    actual = rag.retrieve_text("固定测试问题", max_context_chars=1000)
    assert actual == expected


def test_retrieve_text_cannot_exceed_budget_by_separator_length():
    docs = [document("甲", "a.md"), document("乙", "b.md")]
    rag = object.__new__(module.LearningRAG)
    rag.retrieve_with_strategy = lambda **kwargs: docs
    rag.expand_neighbors = lambda hits, window: docs
    expected = rag.format_documents(docs, max_context_chars=1000)
    budget = len(expected) - 1
    actual = rag.retrieve_text("固定测试问题", max_context_chars=budget)
    assert len(actual) <= budget
    assert "source=b.md" not in actual
