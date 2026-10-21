"""验证 LangGraph 与旧流程的关键行为一致。

使用假分析器和假规划器，不访问模型或重排服务。
检查阶段顺序、证据原样传递、模式选择和失败停止。
"""
import sys

import pytest

from ai_agent.runtime import agents as steps


@pytest.mark.parametrize("orchestrator", ["legacy", "langgraph"])
@pytest.mark.parametrize("mode", ["pipeline", "agent"])
def test_orchestration_preserves_flow_and_evidence(
    monkeypatch, orchestrator, mode
):
    events = []
    trace = {}

    # 模拟现有聊天重排生成的证据头；图不应删除或改写它。
    evidence = (
        "[source=book.pdf, page=5, chunk=8, "
        "rerank_rank=1, rerank_backend=chat, "
        "rerank_model=deepseek-v4.1-flash]\n原始证据"
    )

    def validate(material_dir):
        assert material_dir == "materials"
        events.append("validate")
        return None

    def analyse(material_dir, goal):
        assert (material_dir, goal) == ("materials", "理解 ReAct")
        events.append("analyse")
        return "有效分析", evidence

    def unexpected_analyse(*args, **kwargs):
        pytest.fail("调用了错误的分析模式")

    def plan(analysis, context, goal, days, **kwargs):
        events.append("plan")
        assert (analysis, context, goal, days) == (
            "有效分析", evidence, "理解 ReAct", 3
        )
        assert kwargs["material_dir"] == "materials"
        assert kwargs["mode"] == mode
        # 检查对象身份，确保评测审计仍写入调用方持有的对象。
        assert kwargs["plan_evidence_trace"] is trace
        return "学习计划"

    selected = (
        "analyse_materials_agentic"
        if mode == "agent"
        else "analyse_materials"
    )
    other = (
        "analyse_materials"
        if mode == "agent"
        else "analyse_materials_agentic"
    )

    monkeypatch.setattr(steps, "_validate_material_dir", validate)
    monkeypatch.setattr(steps, selected, analyse)
    monkeypatch.setattr(steps, other, unexpected_analyse)
    monkeypatch.setattr(steps, "create_study_plan", plan)

    result = steps.run_learning_assistant(
        "materials",
        "理解 ReAct",
        3,
        mode,
        orchestrator=orchestrator,
        plan_evidence_trace=trace,
    )

    assert result == ("有效分析", "学习计划", evidence)
    assert events == ["validate", "analyse", "plan"]


@pytest.mark.parametrize("orchestrator", ["legacy", "langgraph"])
def test_invalid_materials_skip_analysis_and_plan(
    monkeypatch, orchestrator
):
    monkeypatch.setattr(
        steps, "_validate_material_dir",
        lambda _: "资料目录无效",
    )

    def forbidden(*args, **kwargs):
        pytest.fail("校验失败后不应分析或规划")

    monkeypatch.setattr(steps, "analyse_materials", forbidden)
    monkeypatch.setattr(steps, "analyse_materials_agentic", forbidden)
    monkeypatch.setattr(steps, "create_study_plan", forbidden)

    result = steps.run_learning_assistant(
        "missing",
        "理解 ReAct",
        3,
        "agent",
        orchestrator=orchestrator,
    )

    assert result == ("资料目录无效", "", "")


@pytest.mark.parametrize("orchestrator", ["legacy", "langgraph"])
@pytest.mark.parametrize("failure", ["blank", "raised"])
def test_analysis_failure_never_reaches_plan(
    monkeypatch, orchestrator, failure
):
    monkeypatch.setattr(
        steps, "_validate_material_dir", lambda _: None
    )

    def analyse(*args, **kwargs):
        if failure == "raised":
            raise steps.AnalysisExecutionError(
                "model_round_limit", "模拟分析未完成"
            )
        return "   ", "原始证据"

    def forbidden_plan(*args, **kwargs):
        pytest.fail("分析失败后不应生成计划")

    monkeypatch.setattr(steps, "analyse_materials", analyse)
    monkeypatch.setattr(steps, "create_study_plan", forbidden_plan)

    expected_reason = (
        "model_round_limit"
        if failure == "raised"
        else "invalid_analysis_output"
    )

    with pytest.raises(steps.AnalysisExecutionError) as caught:
        steps.run_learning_assistant(
            "materials",
            "理解 ReAct",
            3,
            "pipeline",
            orchestrator=orchestrator,
        )

    assert caught.value.reason == expected_reason


def test_cli_passes_orchestrator(monkeypatch):
    import ai_agent.main as main_module

    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        # 空计划使 main 提前结束，避免实际导出或启动交互。
        return "测试提前结束", "", ""

    monkeypatch.setattr(
        main_module, "run_learning_assistant", fake_run
    )
    monkeypatch.setattr(sys, "argv", [
        "ai_agent",
        "--materials", "materials",
        "--goal", "理解 ReAct",
        "--mode", "pipeline",
        "--orchestrator", "langgraph",
    ])

    main_module.main()

    assert captured["orchestrator"] == "langgraph"
    assert captured["mode"] == "pipeline"
