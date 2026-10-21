"""通用学习适配的离线回归测试。

检查查询不因后缀增加领域假设，以及共用规则是否进入模型输入。
假模型不调用在线服务；通过这些测试不代表真实生成质量已经改善。
"""

import pytest

from langchain_core.messages import AIMessage

import ai_agent.runtime.agents as agents_module


def test_queries_do_not_infer_topic_from_file_suffix(tmp_path):
    """同一目标换文件格式，不应改变检索主题。"""
    goal = "理解文章观点，不要求编程"

    md_queries = agents_module._build_broad_retrieval_queries(
        str(tmp_path / "markdown"), goal,
    )
    pdf_queries = agents_module._build_broad_retrieval_queries(
        str(tmp_path / "pdf"), goal,
    )

    assert md_queries == pdf_queries
    assert len(md_queries) == 4

    for _, query in md_queries:
        assert goal in query
        assert "配置与接口" not in query
        assert "故障排查" not in query


@pytest.mark.parametrize("mode", ["pipeline", "agent"])
def test_analysis_receives_learning_activity_rules(
    monkeypatch, mode,
):
    """两种分析入口都应收到规则，避免只修正一种模式。"""
    captured = []

    monkeypatch.setattr(
        agents_module, "create_llm", lambda: object(),
    )
    monkeypatch.setattr(
        agents_module, "create_llm_with_tools",
        lambda tools: object(),
    )

    if mode == "pipeline":
        monkeypatch.setattr(
            agents_module, "_collect_python_sources",
            # 假函数也要兼容真实接口：源码使用 material_dir= 传参。
            # 返回空字符串，避免本测试读取实际 Python 资料。
            lambda material_dir: "",
        )
        monkeypatch.setattr(
            agents_module, "needs_rag", lambda _: False,
        )
        monkeypatch.setattr(
            agents_module, "collect_material_context",
            lambda _: "[source=reading.md]\n文章观点",
        )

        def invoke(llm, messages, *, stage):
            captured.extend(messages)
            return "资料分析。[source=reading.md]"

        monkeypatch.setattr(
            agents_module, "_invoke_llm", invoke,
        )
        agents_module.analyse_materials(
            "materials", "理解文章观点",
        )
    else:
        monkeypatch.setattr(
            agents_module, "build_agent_tools",
            lambda _: [],
        )

        def invoke_raw(llm, messages, *, stage):
            captured.extend(messages)
            return AIMessage(content="资料分析完成")

        monkeypatch.setattr(
            agents_module, "_invoke_llm_raw", invoke_raw,
        )
        agents_module.analyse_materials_agentic(
            "materials", "理解文章观点",
        )

    assert (
        agents_module.LEARNING_ACTIVITY_RULES
        in captured[0].content
    )


@pytest.mark.parametrize("mode", ["pipeline", "agent"])
def test_plan_receives_goal_matched_completion_rules(
    monkeypatch, mode,
):
    """无 RAG 路径也必须收到规则，不能只依赖检索工具。"""
    captured = []

    monkeypatch.setattr(
        agents_module, "needs_rag", lambda _: False,
    )
    monkeypatch.setattr(
        agents_module, "create_llm", lambda: object(),
    )

    def invoke(llm, messages, *, stage):
        captured.extend(messages)
        return "第1天：阅读并解释主要观点"

    monkeypatch.setattr(
        agents_module, "_invoke_llm", invoke,
    )

    agents_module.create_study_plan(
        analysis="文章包含两种观点",
        material_context="[source=reading.md]\n原文",
        learning_goal="理解文章观点，不要求实践",
        days=1,
        material_dir="materials",
        mode=mode,
    )

    prompt = captured[0].content
    assert agents_module.LEARNING_ACTIVITY_RULES in prompt
    assert "与目标匹配的完成标志" in prompt
    assert "不要编造“必须先学”的关系" in prompt
