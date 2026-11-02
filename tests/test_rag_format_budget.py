"""真实RAG格式化函数的预算边界；不构建索引、不调用模型。"""
import pytest
from langchain_core.documents import Document
from ai_agent.retrieval.rag import LearningRAG


def render(documents, budget):
    # format_documents 只使用传入的文档，不依赖实例索引状态。
    return LearningRAG.format_documents(None, documents, max_context_chars=budget)


def doc(text, source):
    return Document(page_content=text, metadata={"source": source, "chunk_index": 0})


def test_oversized_first_chunk_returns_empty_to_preserve_prefix_contract():
    small = doc("短证据", "small.md")
    expected = render([small], 1000)
    assert render([doc("甲" * 500, "large.md"), small], len(expected)) == ""


def test_actual_separators_are_counted_and_exact_fit_is_kept():
    docs = [doc("甲", "a.md"), doc("乙", "b.md")]
    expected = render(docs, 1000)
    assert render(docs, len(expected)) == expected
    assert render(docs, len(expected) - 1) == render(docs[:1], 1000)


def test_all_oversized_returns_empty_without_partial_body():
    assert render([doc("甲" * 100, "a.md")], 10) == ""


@pytest.mark.parametrize("budget", [1, 50, 100, 200])
def test_result_never_exceeds_budget_and_keeps_whole_blocks(budget):
    docs = [doc("甲" * 80, "a.md"), doc("乙", "b.md"), doc("丙", "c.md")]
    result = render(docs, budget)
    assert len(result) <= budget
    blocks = [render([item], 1000) for item in docs]
    chosen = result.split("\n\n---\n\n") if result else []
    assert all(block in blocks for block in chosen)
    assert [blocks.index(block) for block in chosen] == sorted(blocks.index(block) for block in chosen)
