"""重排探针测试：验证接口索引能正确映射回原文，异常结果会被拒绝。"""
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from ai_agent.evaluation.rerank_probe import rerank_candidates, rerank_api_base_address


def test_rerank_reorders_documents_without_losing_metadata():
    docs = [
        Document(page_content="干扰内容",
                 metadata={"source": "book.pdf", "page": 9}),
        Document(page_content="ReAct 的结束条件",
                 metadata={"source": "book.pdf", "page": 4}),
    ]
    received = {}

    def fake_call(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(results=[
                SimpleNamespace(index=1, relevance_score=0.9),
                SimpleNamespace(index=0, relevance_score=0.2),
            ]),
            usage=SimpleNamespace(total_tokens=12),
            request_id="offline-test",
        )

    result = rerank_candidates(
        "ReAct 什么时候结束",
        docs,
        model="test-model",
        api_key="test-key",
        request_timeout=5,
        force_ipv4=False,
        client=SimpleNamespace(call=fake_call),
    )

    assert received["top_n"] == 2
    assert result.documents[0].metadata["page"] == 4
    assert result.documents[0].page_content == docs[1].page_content
    assert docs[1].metadata.get("rerank_score") is None
    assert result.total_tokens == 12


def test_rerank_rejects_missing_candidate():
    docs = [
        Document(page_content="甲", metadata={"source": "b.pdf", "page": 1}),
        Document(page_content="乙", metadata={"source": "b.pdf", "page": 2}),
    ]

    def fake_call(**kwargs):
        return SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(results=[
                SimpleNamespace(index=1, relevance_score=0.9),
            ]),
            usage=None,
            request_id="offline-test",
        )

    with pytest.raises(ValueError, match="返回数量"):
        rerank_candidates(
            "问题", docs,
            model="test-model",
            api_key="test-key",
            request_timeout=5,
            force_ipv4=False,
            client=SimpleNamespace(call=fake_call),
        )

def test_normal_dashscope_url_maps_to_rerank_api_root():
    assert rerank_api_base_address(
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ) == "https://dashscope.aliyuncs.com/api/v1"