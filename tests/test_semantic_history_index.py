"""语义索引候选契约；不请求网络。业务模块应用前这些测试不能直接运行通过。"""
from langchain_core.documents import Document
import pytest

from ai_agent.retrieval.semantic_history import SemanticHistoryIndex


class Embedding:
    def __init__(self):
        self.document_calls = []
        self.query_calls = []
    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [[3.0, 0.0], [0.0, 4.0]][:len(texts)]
    def embed_query(self, text):
        self.query_calls.append(text)
        return [2.0, 0.0]


def documents():
    return [Document(page_content="原话甲", metadata={"source_index": 2, "turn_number": 1}),
            Document(page_content="原话乙", metadata={"source_index": 1, "turn_number": 3})]


def test_normalization_threshold_cache_and_document_isolation():
    embedding = Embedding()
    docs = documents()
    index = SemanticHistoryIndex(docs, embedding)
    docs[0].page_content = "外部修改"
    hits = index.search("换说法", k=5, min_similarity=0.5)
    assert len(hits) == 1 and hits[0][1] == pytest.approx(1.0)
    assert hits[0][0].metadata == {"source_index": 2, "turn_number": 1}
    assert hits[0][0].page_content == "原话甲"
    hits[0][0].metadata["source_index"] = 99
    again = index.search("另一查询", k=1, min_similarity=0.5)
    assert again[0][0].metadata["source_index"] == 2
    assert embedding.document_calls == [["原话甲", "原话乙"]]
    assert len(embedding.query_calls) == 2


def test_low_similarity_returns_empty():
    embedding = Embedding()
    embedding.embed_query = lambda _: [-1.0, 0.0]
    assert SemanticHistoryIndex(documents(), embedding).search("无关", k=3, min_similarity=0.1) == []


def test_empty_index_makes_no_embedding_requests():
    embedding = Embedding()
    assert SemanticHistoryIndex([], embedding).search("问题", k=1, min_similarity=0.5) == []
    assert embedding.document_calls == [] and embedding.query_calls == []


@pytest.mark.parametrize("vectors", [[[0.0, 0.0], [1.0, 0.0]], [[1.0], [1.0, 0.0]], [[float('nan')], [1.0]], [[1.0, 0.0]]])
def test_invalid_embedding_response_rejected(vectors):
    embedding = Embedding()
    embedding.embed_documents = lambda _: vectors
    with pytest.raises(ValueError):
        SemanticHistoryIndex(documents(), embedding)


def test_query_dimension_mismatch_rejected():
    embedding = Embedding()
    index = SemanticHistoryIndex(documents(), embedding)
    embedding.embed_query = lambda _: [1.0]
    with pytest.raises(ValueError):
        index.search("问题", k=1, min_similarity=0.5)


@pytest.mark.parametrize("kwargs", [{"k": True, "min_similarity": 0.5}, {"k": 6, "min_similarity": 0.5},
                                   {"k": 1, "min_similarity": float('nan')}, {"k": 1, "min_similarity": True}])
def test_invalid_search_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        SemanticHistoryIndex(documents(), Embedding()).search("问题", **kwargs)


def test_oversized_corpus_rejected_before_embedding():
    embedding = Embedding()
    with pytest.raises(ValueError):
        SemanticHistoryIndex([Document(page_content="字" * 100001)], embedding)
    assert embedding.document_calls == []
