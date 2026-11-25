"""Online, single-variable probes of the two delegation handoffs."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

import ai_agent.main as app
from ai_agent.runtime import delegated_reading as graph
from ai_agent.runtime import task_delegation as delegation
from tests.helpers.task_delegation_smoke import CASES

MAIN = {
    "baseline": "objective 填写本次委派的整体目标。",
    "goal": "objective 概括交给子任务完成的目标及适用的用户要求。",
    "evidence": "objective 说明本次委派要解决的问题和适用的用户要求；需要核对资料时，发现附证据位置即可追溯。",
}
MAIN_FORCED = {
    **MAIN,
    "scope": "objective 写明本次委派的整体问题及用户指定的交付要求。通常交付可核对的发现与出处；原文摘录服务于引用、措辞比较等任务。",
}
MAIN_DESCRIPTION = {
    "baseline": "规划并执行委派，返回发现、证据位置和缺口。objective 填写本次委派的整体目标。",
    "roles": "你负责确定本次委派要解决的整体目标，并填入 objective。规划器据此拆解子任务。工具返回子任务的发现、证据位置和缺口。",
}
PLAN = {
    "baseline": graph.FOCUSED_PLANNER.splitlines()[0],
    "autonomy": "需要派发时，在 instruction 说明该分支要解决的问题及适用的用户要求；查阅方式由执行者按需决定。",
    "precedence": "需要派发时，按 delegation_goal 确定分工，以 question 为准保留用户的交付要求和范围。instruction 写清该分支负责的目标和要求。",
    "goal_only": graph.FOCUSED_PLANNER.splitlines()[0],
}
PLAN_HANDOFF = "围绕委派目标规划，保留原问题中适用的要求。"
PLAN_BOUNDARY = "按委派目标组织分工，以 question 确定用户要求。查阅资料时交付可追溯的发现，引用或核对措辞等任务按需摘录原文。"
PLAN_QUOTE_SCOPE = "按用户原问题和委派目标分工，说明每项要解决的问题。资料解释可以转述。只有用户明确要求或问题依赖原话、具体位置时，才安排逐字摘录或额外定位。"
QUESTIONS = {
    "open": CASES["requested_cross_book"],
    "quote": CASES["requested_quote"],
}
OBJECTIVES = {
    "open": (
        "分别核对两本教材如何配对工具结果、如何反馈 Observation，"
        "给出原句和证据位置，供组会比较和设计口头练习。"
    ),
    "quote": "并行核对两本教材关于工具结果处理的正文，给出证据位置供比较。",
}


async def compare(materials, reference, output, stage, repeats):
    variants = MAIN_FORCED if stage == "main-forced" else MAIN if stage == "main" else PLAN
    questions = QUESTIONS
    if stage == "main-roles":
        variants = MAIN_DESCRIPTION
        questions = {**QUESTIONS, **{case: CASES[case] for case in ("requested_dependency", "missing_branch", "direct")}}
    if stage == "main-architecture":
        variants = dict.fromkeys(("baseline", "direct", "prepared"))
        questions = {**QUESTIONS, **{case: CASES[case] for case in ("requested_dependency", "missing_branch", "direct")},
                     "advice": "我理解概念时总是觉得自己懂了，隔天却解释不清楚。请用你的一般知识给我三个改进办法。"}
    if stage == "plan-boundary":
        variants = {"baseline": PLAN["baseline"], "boundary": PLAN["baseline"]}
    if stage == "plan-quote-scope":
        variants = {"baseline": PLAN["baseline"], "quote_scope": PLAN["baseline"]}
    if stage == "plan-clean-goal":
        variants = {"baseline": PLAN["baseline"], "clean_goal": PLAN["baseline"]}
    output.mkdir(parents=True, exist_ok=True)
    planner_messages = reference["delegation_diagnostics"]["model_calls"][0]["messages"]
    system = planner_messages[0]["content"]
    assert system.startswith(PLAN["baseline"] + "\n")
    raw_scope = json.loads(planner_messages[-1]["content"])["authorized_scope"]
    scope = {key: None if pages is None else set(pages) for key, pages in raw_scope.items()}
    semaphore = asyncio.Semaphore(3)

    async def run(case, variant, repeat):
        question = questions[case]
        schemas = None
        if stage.startswith("main"):
            settings = app.get_settings()
            delegate = delegation.build_learning_delegation_tool(
                materials, question, main_plans=stage == "main-architecture" and variant != "baseline",
            )
            if stage == "main-roles":
                delegate.description = variants[variant]
            elif stage != "main-architecture":
                assert delegate.description.endswith(MAIN["baseline"])
                delegate.description = delegate.description.removesuffix(MAIN["baseline"]) + variants[variant]
            list_materials = app.build_list_material_files_tool(materials)
            prepared = stage == "main-architecture" and variant == "prepared"

            @tool
            def prepare_learning_tasks() -> str:
                """为需要分工的学习任务获取任务格式和执行者可用能力。收到后，你可以提交分工，也可以继续直接处理。"""
                return json.dumps({
                    "materials": json.loads(list_materials.invoke({})),
                    "worker_capabilities": ["locate_material", "read_pdf_pages", "read_material_lines", "use_upstream_reports"],
                }, ensure_ascii=False)

            execution_schema = convert_to_openai_tool(delegate)
            schemas = [convert_to_openai_tool(item) for item in [
                app.build_retrieve_learning_context_tool(
                    materials, structured_result=True,
                    max_queries_per_call=settings.interactive_max_queries_per_tool_call,
                    max_context_chars=settings.interactive_retrieval_max_context_chars,
                ),
                app.build_read_material_lines_tool(materials),
                list_materials,
                app.build_read_pdf_pages_tool(materials),
                prepare_learning_tasks if prepared else delegate,
            ]]
            messages = [
                SystemMessage(content="你是学习助手，依据用户问题和可用资料回答。\n\n" + delegation.DELEGATION_ROUTING_HINT),
                HumanMessage(content=question),
            ]
        else:
            payload = {"delegation_goal": OBJECTIVES[case], "authorized_scope": raw_scope}
            if variant == "clean_goal" and case == "open":
                payload["delegation_goal"] = "分别核对两本教材如何配对工具结果、如何反馈 Observation，给出证据位置，供组会比较和设计口头练习。"
            if variant != "goal_only":
                payload = {"question": question, **payload}
            prompt = system.replace(PLAN["baseline"], PLAN[variant] if variant in PLAN else PLAN["baseline"], 1)
            if variant in ("boundary", "quote_scope"):
                assert prompt.endswith(PLAN_HANDOFF)
                prompt = prompt.removesuffix(PLAN_HANDOFF) + (PLAN_QUOTE_SCOPE if variant == "quote_scope" else PLAN_BOUNDARY)
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        row = {
            "stage": stage, "variant": variant, "case": case, "repeat": repeat, "user_input": question,
            "request_messages": [{"type": item.type, "content": item.content} for item in messages],
            "tools": schemas, "tool_choice": "delegate_learning_task" if stage == "main-forced" else None,
        }
        async with semaphore:
            model = app.create_llm()
            if schemas is not None:
                model = model.bind_tools(schemas, **({"tool_choice": row["tool_choice"]} if row["tool_choice"] else {}))
            start = perf_counter()
            try:
                answer = await asyncio.wait_for(model.ainvoke(messages), timeout=60)
                row.update(content=answer.content, tool_calls=answer.tool_calls, usage=answer.usage_metadata)
                if stage in ("main-roles", "main-architecture"):
                    discovery = {list_materials.name: list_materials, "prepare_learning_tasks": prepare_learning_tasks}
                else:
                    discovery = {}
                if answer.tool_calls and all(call["name"] in discovery for call in answer.tool_calls):
                    # Allow the same real, read-only discovery step on both sides of the comparison.
                    prepared_now = any(call["name"] == "prepare_learning_tasks" for call in answer.tool_calls)
                    receipts = [discovery[call["name"]].invoke(call) for call in answer.tool_calls]
                    followup = [SystemMessage(content="你是学习助手，依据用户问题和可用资料回答。"),
                                messages[-1], answer, *receipts]
                    next_schemas = [*schemas[:-1], execution_schema] if prepared_now else schemas
                    next_answer = await asyncio.wait_for(app.create_llm().bind_tools(next_schemas).ainvoke(followup), timeout=60)
                    row["after_list"] = {
                        "request_messages": [{"type": item.type, "content": item.content} for item in followup],
                        "tool_calls": next_answer.tool_calls, "content": next_answer.content,
                        "usage": next_answer.usage_metadata, "tools": next_schemas,
                    }
                if stage.startswith("plan"):
                    plan = graph.parse(answer.content, graph.RoomyPlan)
                    graph.validate(plan.tasks, scope)
                    row["plan"] = plan.model_dump(exclude_defaults=True)
            except Exception as error:
                row["error_type"] = type(error).__name__
            row["seconds"] = round(perf_counter() - start, 2)
        (output / f"{stage}-{case}-{variant}-{repeat}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8",
        )
        print(stage, case, variant, repeat, row["seconds"], row.get("error_type", "ok"), flush=True)

    # Rotate variant order across repetitions to reduce time-of-run bias.
    names = list(variants)
    await asyncio.gather(*(
        run(case, variant, repeat)
        for repeat in range(repeats)
        for variant in names[repeat % len(names):] + names[:repeat % len(names)]
        for case in questions
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials")
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("stage", choices=("main", "main-forced", "main-roles", "main-architecture", "plan", "plan-boundary", "plan-clean-goal", "plan-quote-scope"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(compare(args.materials, json.loads(args.reference.read_text(encoding="utf8")),
                        args.output, args.stage, args.repeats))
