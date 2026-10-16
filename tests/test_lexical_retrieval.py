"""可复用关键词检索模块的离线测试。

验证分词、候选匹配、稳定排序、空输入与旧导入兼容。
不连接向量库、Embedding、重排或聊天模型。
"""

import pytest
from langchain_core.documents import Document

from ai_agent.lexical_retrieval import CharBigramBM25, _terms
from ai_agent.evaluation.lexical_probe import (
    CharBigramBM25 as ProbeBM25,
)


def test_old_import_path_reuses_same_class():
    """旧评测入口和未来业务入口必须使用同一份实现。"""
    assert ProbeBM25 is CharBigramBM25


def test_terms_preserve_existing_rules():
    assert _terms("工具调用 ReAct tool_call 123") == [
        "工具",
        "具调",
        "调用",
        "react",
        "tool_call",
        "123",
    ]


def test_keyword_search_supports_non_pdf_and_stable_ties():
    """关键词索引不应依赖 page，也不应让同分结果随机交换。"""
    documents = [
        Document(
            page_content="ReAct 调用工具",
            metadata={"source": "a.md", "chunk_index": 0},
        ),
        Document(
            page_content="天气预报",
            metadata={"source": "b.txt", "chunk_index": 0},
        ),
        Document(
            page_content="ReAct 循环终止",
            metadata={"source": "c.md", "chunk_index": 0},
        ),
    ]
    retriever = CharBigramBM25(documents)

    hits = retriever.search("react", k=8)

    assert [
        doc.metadata["source"] for doc in hits
    ] == ["a.md", "c.md"]

    # 两个匹配块长度和词频相同，同分时保持输入顺序。
    assert hits == retriever.search("REACT", k=8)


@pytest.mark.parametrize("query", ["", "   ", "不存在的词"])
def test_empty_or_unmatched_query_returns_no_hits(query):
    retriever = CharBigramBM25([
        Document(page_content="ReAct Memory"),
    ])
    assert retriever.search(query, k=3) == []


@pytest.mark.parametrize("documents", [
    [],
    [Document(page_content="")],
])
def test_empty_corpus_or_empty_text_is_safe(documents):
    retriever = CharBigramBM25(documents)
    assert retriever.search("ReAct", k=3) == []


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_is_rejected(k):
    retriever = CharBigramBM25([])
    with pytest.raises(ValueError, match="k"):
        retriever.search("ReAct", k=k)


def test_queries_do_not_pollute_metadata_or_rebuild_terms(monkeypatch):
    """多次查询复用索引，且不把临时检索信息写入原始 metadata。"""
    documents = [
        Document(
            page_content="ReAct",
            metadata={"source": "a.md", "chunk_index": 0},
        ),
        Document(
            page_content="Memory",
            metadata={"source": "b.txt", "chunk_index": 0},
        ),
    ]
    original_metadata = [
        dict(doc.metadata) for doc in documents
    ]
    retriever = CharBigramBM25(documents)

    import ai_agent.lexical_retrieval as lexical_module

    original_terms = lexical_module._terms
    tokenized_texts = []

    def record_terms(text):
        tokenized_texts.append(text)
        return original_terms(text)

    # 在构建之后记录分词，后续应该只分词查询，不重新分词资料。
    monkeypatch.setattr(lexical_module, "_terms", record_terms)

    assert retriever.search("react", k=1)[0] is documents[0]
    assert retriever.search("memory", k=1)[0] is documents[1]

    assert tokenized_texts == ["react", "memory"]
    assert [
        doc.metadata for doc in documents
    ] == original_metadata