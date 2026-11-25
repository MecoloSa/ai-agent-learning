"""Bind optional, read-only learning delegation to one interactive turn."""

import asyncio
import json
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from openai import APIError
from pydantic import BaseModel, Field

from ai_agent.config import create_llm
from ai_agent.resources.materials import iter_material_files, safe_material_path
from ai_agent.runtime.delegated_reading import Assignment

DELEGATION_ROUTING_HINT = "已有依据时直接回答，需要核对时可自行使用资料工具。适合分工的多步任务，可以作为一个整体委派。"
DELEGATION_TIMEOUT_SECONDS = 150
WORKER_TIMEOUT_SECONDS = 45
WORKER_REPORT_RESERVE_SECONDS = 12
SUBMITTED_TASKS_DESCRIPTION = (
    "执行你提交的一组子任务，返回各任务的发现、资料位置和缺口。"
    "每项说明要解决的问题和适用的用户要求，确实需要上游结果时填写依赖。"
    "依据资料解释时可以转述。只有用户要求或问题依赖原话、具体位置时，才安排逐字摘录或额外定位。"
)


class SubmittedTask(BaseModel):
    id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    sources: list[str] = Field(description="资料目录中的相对文件名。仅依赖上游结果时可留空。")
    depends_on: list[str] = Field(default_factory=list, description="需要先完成的任务 ID。")


def build_learning_delegation_tool(material_dir, user_request, *, main_plans=False, worker_thinking=None):
    started = False

    def execute(*, objective=None, tasks=None):
        nonlocal started
        if started:
            return _failure("delegation_already_started", "请使用本轮已返回的报告，缺失部分可用资料工具核实。")
        if tasks is None and not objective.strip():
            return _failure("missing_objective")

        # 只有实际调用才枚举资料和装配子图。路径取自服务器绑定的目录。
        root = Path(material_dir).resolve()
        scope = {}
        for path in iter_material_files(root):
            relative = path.relative_to(root).as_posix()
            try:
                safe_material_path(str(root), relative)
            except ValueError:
                continue
            scope[relative] = None
        if not scope:
            return _failure("no_readable_materials")
        if tasks is not None:
            from ai_agent.runtime.delegated_reading import validate
            try:
                validate(tasks, scope)
            except ValueError as error:
                return _failure("invalid_delegation_tasks", str(error), available_sources=sorted(scope))
        started = True

        async def ask(role, stage, messages, tools=()):
            model = create_llm(role=role)
            options = {}
            if role == "auxiliary" and worker_thinking is not None:
                options["extra_body"] = {"enable_thinking": worker_thinking}
            if tools:
                model = model.bind_tools(tools, **options)
            elif options:
                model = model.bind(**options)
            return await model.ainvoke(messages)

        async def fast_report(messages):
            model = create_llm(role="auxiliary").bind(extra_body={"enable_thinking": False})
            return await model.ainvoke(messages)

        async def run():
            from ai_agent.runtime import delegated_reading

            graph = delegated_reading.build_graph(
                root, scope, ask, [], planner_role="standard", roomy=True,
                reuse_retrieval=True, focused_planning=True, shared_pdf_reads=True,
                model_request_timeout=60, collect_only=True,
                turn_timeout=DELEGATION_TIMEOUT_SECONDS, delivery_reserve=2,
                worker_timeout=WORKER_TIMEOUT_SECONDS,
                branch_report_reserve=WORKER_REPORT_RESERVE_SECONDS, fast_report=fast_report,
                **({"submitted_tasks": tasks} if tasks is not None else {}),
            )
            # 先让分支带已有回执收尾，外层截止时间用于兜底。
            async with asyncio.timeout(DELEGATION_TIMEOUT_SECONDS):
                state = {"question": user_request}
                if objective is not None:
                    state["delegation_goal"] = objective.strip()
                return await graph.ainvoke(state)

        try:
            result = asyncio.run(run())["delegated_report"]
        except TimeoutError:
            return _failure("delegation_timed_out")
        except (APIError, ValueError, OSError):
            return _failure("delegation_unavailable")
        if result["status"] == "planning_timed_out":
            return _failure("planning_timed_out")
        return json.dumps({"ok": True, "data": result}, ensure_ascii=False)

    if main_plans:
        @tool("delegate_learning_task", description=SUBMITTED_TASKS_DESCRIPTION)
        def submit_tasks(tasks: Annotated[list[SubmittedTask], Field(min_length=1)]) -> str:
            return execute(tasks=[Assignment(**task.model_dump()) for task in tasks])

        return submit_tasks

    @tool
    def delegate_learning_task(objective: str) -> str:
        """规划并执行委派，返回发现、证据位置和缺口。objective 填写本次委派的整体目标。"""
        return execute(objective=objective)

    return delegate_learning_task


def _failure(code, message=None, **details):
    return json.dumps({"ok": False, "error": {"code": code, **({"message": message} if message else {}), **details}},
                      ensure_ascii=False)
