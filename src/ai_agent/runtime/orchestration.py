"""学习助手的最小 LangGraph 编排。

只管理校验、分析、规划之间的数据流，不改写内部工具和重排流程。
默认不启用checkpoint；底层可显式使用内存或SQLite检查点。
图状态不等于LearningState；普通CLI尚不开放工作流恢复入口。
"""
from __future__ import annotations

import logging
import time
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from ai_agent.runtime import agents as steps
from ai_agent.persistence import checkpoint_guard

logger = logging.getLogger("ai_agent.orchestration")


class AssistantGraphState(TypedDict, total=False):
    # total=False 表示字段在类型描述上可以暂时缺省。
    # 节点返回部分更新、保留其他字段，是LangGraph的状态更新机制。
    # TypedDict 不替代已有业务校验。
    material_dir: str
    learning_goal: str
    days: int
    mode: str
    materials_valid: bool
    analysis: str
    material_context: str
    plan: str
    # 与已完成分析一起保存；只有checkpoint路径需要。
    material_fingerprint: str
    # 保存创建任务时的兼容契约，不能在恢复时覆盖为当前版本。
    workflow_revision: str

def _state_summary(state: dict) -> dict:
    """只记录长度和程序标志，不把正文、目标或路径写进节点轨迹。"""
    summary = {}

    for key in ("analysis", "material_context", "plan"):
        value = state.get(key)
        summary[f"{key}_chars"] = (
            len(value) if isinstance(value, str) else None
        )

    summary["materials_valid"] = state.get("materials_valid")
    return summary


def _observe_node(name, action, trace):
    """包装原节点：观察执行事实，不重试、不替换异常或业务结果。"""

    def wrapped(state: AssistantGraphState) -> dict:
        started = time.perf_counter()
        event = {
            "node": name,
            "status": "running",
            "input": _state_summary(state),
        }
        trace["events"].append(event)

        logger.debug(
            "节点开始：run=%s node=%s input=%s",
            trace["run_id"],
            name,
            event["input"],
        )

        try:
            update = action(state)
        except Exception as exc:
            reason = getattr(exc, "reason", None)

            event.update(
                status="failed",
                error_type=type(exc).__name__,
                # 当前业务异常的reason是程序原因码。
                # 不记录可能包含密钥、路径或正文的异常全文。
                reason=reason if isinstance(reason, str) else None,
                elapsed_ms=round(
                    (time.perf_counter() - started) * 1000
                ),
            )
            trace["failed_node"] = name

            logger.warning(
                "节点失败：run=%s node=%s "
                "error_type=%s reason=%s elapsed_ms=%d",
                trace["run_id"],
                name,
                event["error_type"],
                event["reason"],
                event["elapsed_ms"],
            )

            # 原异常继续抛出，保留CLI和评测已有的失败处理。
            raise

        # 这里只合并数据以计算摘要，不修改传入状态或节点返回值。
        # returned不等于内容正确；目录被拒绝也是正常返回。
        event.update(
            status="returned",
            output=_state_summary({**state, **update}),
            elapsed_ms=round(
                (time.perf_counter() - started) * 1000
            ),
        )

        logger.debug(
            "节点返回：run=%s node=%s elapsed_ms=%d output=%s",
            trace["run_id"],
            name,
            event["elapsed_ms"],
            event["output"],
        )

        return update

    return wrapped

def run_learning_graph(
    material_dir: str,
    learning_goal: str,
    days: int,
    mode: str,
    *,
    plan_evidence_trace: dict | None = None,
    workflow_trace: dict | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
    resume: bool = False,
) -> tuple[str, str, str]:
    """保持原入口的返回值和异常约定，便于与 legacy 对照。"""
    run_trace = {
        "trace_version": "workflow-v1",
        "run_id": uuid4().hex,
        "execution_kind": "resume" if resume else "new",
        "thread_id": thread_id,
        "status": "running",
        "events": [],
    }

    if workflow_trace is not None:
        # 接收方可在正常返回或捕获异常后查看最近一次运行。
        # 每次创建新记录，不把上次事件混入本次。
        workflow_trace["latest_run"] = run_trace

    def validate_materials(state: AssistantGraphState) -> dict:
        guidance = steps._validate_material_dir(state["material_dir"])

        if guidance is not None:
            logger.warning("流程分支：资料校验未通过，直接结束")
            return {
                "materials_valid": False,
                "analysis": guidance,
            }

        return {"materials_valid": True}

    def analyse(state: AssistantGraphState) -> dict:

        analyser = (
            steps.analyse_materials_agentic
            if state["mode"] == "agent"
            else steps.analyse_materials
        )

        fingerprint = (
            checkpoint_guard.material_fingerprint(
                state["material_dir"]
            )
            if checkpointer is not None
            else None
        )

        # 复用原业务函数；重排后端仍由现有检索配置决定。
        analysis, context = analyser(
            state["material_dir"],
            state["learning_goal"],
        )

        # 无效分析继续抛异常，不进入规划节点。
        # 动态评估：失败场景中规划函数调用次数必须为 0。
        analysis = steps._require_analysis_text(analysis)

        update = {
            "analysis": analysis,
            "material_context": context,
        }

        if checkpointer is not None:
            # 分析期间资料若变化，不能把结果绑定到一个不确定版本。
            checkpoint_guard.require_material_fingerprint(
                state["material_dir"],
                fingerprint,
            )
            update["material_fingerprint"] = fingerprint

        return update

    def plan(state: AssistantGraphState) -> dict:
        if checkpointer is not None:
            checkpoint_guard.require_material_fingerprint(
                state["material_dir"],
                state.get("material_fingerprint"),
            )

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

        if checkpointer is not None:
            # 规划可能追加检索；执行期间发生资料变化时，不提交正常结果。
            checkpoint_guard.require_material_fingerprint(
                state["material_dir"],
                state.get("material_fingerprint"),
            )

        return {"plan": result}

    graph = StateGraph(AssistantGraphState)
    # 节点名字 "analyse" → 对应函数 / 监听工具包装后的函数
    graph.add_node(
        "validate_materials",
        _observe_node("validate_materials", validate_materials, run_trace),
    )
    graph.add_node(
        "analyse",
        _observe_node("analyse", analyse, run_trace),
    )
    graph.add_node(
        "plan",
        _observe_node("plan", plan, run_trace),
    )

    graph.add_edge(START, "validate_materials")
    graph.add_conditional_edges(
        "validate_materials",
        lambda state: state["materials_valid"],
        {True: "analyse", False: END},
    )
    graph.add_edge("analyse", "plan")
    graph.add_edge("plan", END)

    # compile 生成可执行图，不请求模型，也不自动提供失败恢复。
    # 节点已经注册；
    # 边已经声明；
    # 框架检查并组织图结构；
    # 得到能够执行的图对象。
    initial_state = {
        "material_dir": material_dir,
        "learning_goal": learning_goal,
        "days": days,
        "mode": mode,
        "analysis": "",
        "material_context": "",
        "plan": "",
    }

    if checkpointer is not None:
        # 新任务保存当时的版本。
        # 恢复时仍传None给图，不能用这个初始状态覆盖旧检查点。
        initial_state["workflow_revision"] = (
            checkpoint_guard.WORKFLOW_REVISION
        )

    try:
        if checkpointer is None:
            if resume or thread_id is not None:
                raise ValueError(
                    "恢复或指定thread_id时必须提供checkpointer"
                )
            config = None
        else:
            # 图依赖统一接口，存储方式由调用方选择。
            # 这里只放宽存储实现，不放宽后面的任务参数和资料版本检查。
            if not isinstance(checkpointer, BaseCheckpointSaver):
                raise ValueError(
                    "checkpointer必须实现LangGraph检查点接口"
                )
            if not isinstance(thread_id, str) or not thread_id.strip():
                raise ValueError(
                    "启用checkpoint时必须提供非空thread_id"
                )
            config = {
                "configurable": {"thread_id": thread_id}
            }

        app = graph.compile(checkpointer=checkpointer)
        graph_input = initial_state

        if checkpointer is not None:
            snapshot = app.get_state(config)

            if resume:
                # 只开放失败规划节点的显式重试；
                # 不能把已完成任务、分析失败或普通暂停当作规划失败。
                failed_plan = any(
                    task.name == "plan" and task.error is not None
                    for task in snapshot.tasks
                )
                if snapshot.next != ("plan",) or not failed_plan:
                    raise ValueError(
                        "没有可重试的失败规划节点"
                    )

                # 防止借用旧分析，悄悄换成另一个目标或执行模式。
                for key in (
                    "material_dir",
                    "learning_goal",
                    "days",
                    "mode",
                ):
                    if snapshot.values.get(key) != initial_state[key]:
                        raise ValueError(
                            f"恢复参数与原任务不一致：{key}"
                        )

                # 先确认当前程序仍能解释旧状态，再检查资料版本。
                # get返回None时也会拒绝，避免无版本的旧记录被默认放行。
                checkpoint_guard.require_workflow_revision(
                    snapshot.values.get("workflow_revision")
                )

                # 参数相同还不够：同一个目录里的资料内容也必须一致。
                checkpoint_guard.require_material_fingerprint(
                    material_dir,
                    snapshot.values.get("material_fingerprint"),
                )

                # None表示读取该任务的已有状态继续。
                # 重新传入初始空状态可能覆盖分析，不能用于恢复。
                graph_input = None

            elif snapshot.values:
                # 新任务不能覆盖同一ID已经保存的执行状态。
                raise ValueError(
                    "thread_id已有任务；请使用新ID或显式恢复"
                )

        result = app.invoke(graph_input, config=config)
    except Exception as exc:
        run_trace["status"] = "failed"
        run_trace["error_type"] = type(exc).__name__

        # 没有节点失败记录时，可能是图运行器本身失败，不能猜测节点。
        run_trace.setdefault("failed_node", None)
        raise

    run_trace["status"] = (
        "completed"
        if result["materials_valid"]
        else "rejected"
    )

    logger.debug(
        "图运行结束：run=%s status=%s nodes=%d",
        run_trace["run_id"],
        run_trace["status"],
        len(run_trace["events"]),
    )

    return result["analysis"], result["plan"], result["material_context"]
