"""学习助手的最小 LangGraph 编排。

只管理校验、分析、规划之间的数据流，不改写内部工具和重排流程。
本阶段不启用 checkpoint；图状态不等于持久化的 LearningState。
"""
from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import agents as steps

logger = logging.getLogger("ai_agent.orchestration")


class AssistantGraphState(TypedDict, total=False):
    # total=False 允许节点只返回变化字段，其余字段由图继续保留。
    # TypedDict 描述字段结构，不负责替代已有业务校验。
    material_dir: str
    learning_goal: str
    days: int
    mode: str
    materials_valid: bool
    analysis: str
    material_context: str
    plan: str


def run_learning_graph(
    material_dir: str,
    learning_goal: str,
    days: int,
    mode: str,
    *,
    plan_evidence_trace: dict | None = None,
) -> tuple[str, str, str]:
    """保持原入口的返回值和异常约定，便于与 legacy 对照。"""

    def validate_materials(state: AssistantGraphState) -> dict:
        logger.debug("流程节点开始：validate_materials")
        guidance = steps._validate_material_dir(state["material_dir"])

        if guidance is not None:
            logger.warning("流程分支：资料校验未通过，直接结束")
            return {
                "materials_valid": False,
                "analysis": guidance,
            }

        logger.debug("流程节点完成：validate_materials")
        return {"materials_valid": True}

    def analyse(state: AssistantGraphState) -> dict:
        logger.debug("流程节点开始：analyse mode=%s", state["mode"])

        analyser = (
            steps.analyse_materials_agentic
            if state["mode"] == "agent"
            else steps.analyse_materials
        )

        # 复用原业务函数；重排后端仍由现有检索配置决定。
        analysis, context = analyser(
            state["material_dir"],
            state["learning_goal"],
        )

        # 无效分析继续抛异常，不进入规划节点。
        # 动态评估：失败场景中规划函数调用次数必须为 0。
        analysis = steps._require_analysis_text(analysis)

        logger.debug("流程节点完成：analyse")
        return {
            "analysis": analysis,
            "material_context": context,
        }

    def plan(state: AssistantGraphState) -> dict:
        logger.debug("流程节点开始：plan")

        result = steps.create_study_plan(
            state["analysis"],
            state["material_context"],
            state["learning_goal"],
            state["days"],
            material_dir=state["material_dir"],
            mode=state["mode"],
            # 沿用调用方的审计对象，避免记录写进无关副本。
            plan_evidence_trace=plan_evidence_trace,
        )

        logger.debug("流程节点完成：plan")
        return {"plan": result}

    graph = StateGraph(AssistantGraphState)
    graph.add_node("validate_materials", validate_materials)
    graph.add_node("analyse", analyse)
    graph.add_node("plan", plan)

    graph.add_edge(START, "validate_materials")
    graph.add_conditional_edges(
        "validate_materials",
        lambda state: state["materials_valid"],
        {True: "analyse", False: END},
    )
    graph.add_edge("analyse", "plan")
    graph.add_edge("plan", END)

    # compile 生成可执行图，不请求模型，也不自动提供失败恢复。
    app = graph.compile()
    result = app.invoke({
        "material_dir": material_dir,
        "learning_goal": learning_goal,
        "days": days,
        "mode": mode,
        "analysis": "",
        "material_context": "",
        "plan": "",
    })

    return result["analysis"], result["plan"], result["material_context"]
