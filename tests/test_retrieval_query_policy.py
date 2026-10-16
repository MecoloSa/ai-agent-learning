"""按需查询策略与单查询入口测试。

验证真实提示、工具描述、单条/多条 query 参数传递及重排开关。
不依赖在线模型；这些测试不证明模型一定遵守提示。
"""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import ai_agent.main as main_module
import ai_agent.tools as tools_module
import ai_agent.rag as rag_module


def test_interactive_prompt_allows_focused_and_adaptive_queries(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(
            interactive_preload_materials=False,
            interactive_max_retrieval_calls=3,
            interactive_max_queries_per_tool_call=5,
        ),
    )

    prompt = main_module._build_interactive_system_prompt(
        analysis="资料分析",
        plan="学习计划",
        material_context="",
        learning_goal="理解 ReAct",
        retrieval_available=True,
        on_demand_eligible=True,
    )

    assert "聚焦问题默认只提交一个当前必要查询" in prompt
    assert "必须先看结果才能确定的下一问" in prompt
    assert "目录条目" in prompt
    assert "所有工具合计最多调用3次" in prompt


@pytest.mark.parametrize("queries", [
    ["目录中的终止条件条目"],
    ["工具调用的定义", "结果回传的定义"],
])
def test_tool_accepts_single_or_complementary_queries(
    monkeypatch, queries
):
    captured = []

    class FakeRAG:
        def __init__(self, *args, **kwargs):
            pass

        def retrieve_many(self, query_pairs, **kwargs):
            captured.append(query_pairs)
            return [Document(
                page_content="资料证据",
                metadata={"source": "a.md", "chunk_index": 0},
            )]

        def expand_neighbors(self, hits, *, window):
            return hits

        def format_documents(self, documents, *, max_context_chars):
            return "[source=a.md, chunk=0]\n资料证据"

    monkeypatch.setattr(tools_module, "LearningRAG", FakeRAG)

    tool = tools_module.build_retrieve_learning_context_tool(
        "materials",
        max_queries_per_call=2,
        max_context_chars=2000,
    )
    result = tool.invoke({
        "queries": queries,
        "neighbor_window": 0,
    })

    assert captured == [
        tuple(
            (f"tool_query_{index}", query)
            for index, query in enumerate(queries, start=1)
        )
    ]
    assert "资料证据" in result
    assert "聚焦问题默认只提交一个明确查询" in tool.description
    assert "必须先看结果" in tool.description


@pytest.mark.parametrize("enabled", [False, True])
def test_retrieve_text_forwards_rerank_setting(monkeypatch, enabled):
    monkeypatch.setattr(
        rag_module,
        "get_settings",
        lambda: SimpleNamespace(
            rag_min_hits_per_query=3,
            rag_rerank_enabled=enabled,
        ),
    )

    rag = object.__new__(rag_module.LearningRAG)
    captured = {}

    def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return []

    rag.retrieve_with_strategy = fake_retrieve

    rag.retrieve_text("目录中的终止条件条目")

    assert captured["rerank"] is enabled