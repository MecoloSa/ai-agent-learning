"""Live smoke against the actual interactive tool loop with textbook fixtures."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from langchain_core.messages import SystemMessage

import ai_agent.main as app
from ai_agent.runtime.task_delegation import build_learning_delegation_tool


CASES = {
    "direct": "假设 Action 是执行动作，Observation 是执行后得到的结果。给我一个取快递的类比，用一句话回答。",
    "pdf": "只读《AI Agents in Depth》PDF 物理第41页，并用两句话解释该页的工具调用 ID 用途。",
    "dependency": (
        "只读《AI Agents in Depth》PDF 物理第41、44页，分别提取第41页两个调用 ID 对应的工具"
        "和第44页写回结果的字段。拿到两份结果后，再推演把天气结果挂到时间调用 ID 上会怎样。区分教材事实和推断。"
    ),
    "cross_book": (
        "准备一次组会讲解。分别从《AI Agents in Depth》和《Hello-Agents》查到能核对的正文，"
        "解释前者如何配对工具结果、后者如何把 Observation 反馈给模型，再比较两者分别解决什么问题。"
        "请依据两边找到的材料设计一个能暴露初学者误解的口头练习，并说明如何判分。"
    ),
    "missing_branch": (
        "请分工核对两本书的指定页：只看《AI Agents in Depth》PDF 物理第41页和"
        "《Hello-Agents》PDF 物理第9999页。若有一页不存在，不要用其他页代替。"
        "说明哪一页无法核对，并解释能读到的那一页。"
    ),
}
CASES["requested_dependency"] = "做一次分工核对：" + CASES["dependency"]
CASES["requested_cross_book"] = "请把两本教材的核对工作并行委派，再综合：" + CASES["cross_book"]
CASES["requested_quote"] = (
    "请并行委派核对两本教材。只看《AI Agents in Depth》PDF 物理第41页和"
    "《Hello-Agents》PDF 物理第17页。各逐字引用一句原文并标明 PDF 物理页，"
    "用两句话说明这两句分别解释了工具结果处理的什么问题。"
)
CASES["long_excerpt"] = "把《Hello-Agents》第17物理页能提取到的正文全部逐字贴出来，我想核对 Observation 这一节。"


def run(root, output, case, delegation_diagnostics=None, *, main_plans=False, production_prompt=False):
    question = CASES[case]
    audit = {}
    read = app.build_read_material_lines_tool(root)
    options = app.get_settings()
    retrieval = app.build_retrieve_learning_context_tool(
        root, structured_result=True,
        max_queries_per_call=options.interactive_max_queries_per_tool_call,
        max_context_chars=options.interactive_retrieval_max_context_chars,
    )
    tools = [app.build_list_material_files_tool(root), app.build_read_pdf_pages_tool(root),
             build_learning_delegation_tool(root, question, main_plans=main_plans)]
    system = (app._build_interactive_system_prompt("", "", "", "", retrieval_available=True, on_demand_eligible=True)
              if production_prompt else "你是学习助手，依据用户问题和可用资料回答。")
    start = perf_counter()
    app._invoke_interactive_turn(
        request_messages=[SystemMessage(content=system)],
        user_input=question, plain_llm=app.create_llm(),
        tool_llm=app.create_llm_with_tools([retrieval, read, *tools]), retrieval_tool=retrieval,
        read_tool=read, extra_tools=tools, audit_trace=audit,
    )
    audit["seconds"] = round(perf_counter() - start, 2)
    if delegation_diagnostics is not None:
        audit["delegation_diagnostics"] = delegation_diagnostics
    (output / f"{case}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf8")
    print(case, audit["seconds"], len(audit["model_responses"]),
          [row.get("tool_name") for row in audit["tool_results"]], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--report-reserve", type=float, default=0)
    parser.add_argument("--turn-timeout", type=float, default=150)
    parser.add_argument("--report-no-thinking", action="store_true")
    parser.add_argument("--capture-production", action="store_true")
    parser.add_argument("--main-plans", action="store_true")
    parser.add_argument("--production-prompt", action="store_true")
    args = parser.parse_args()
    diagnostics = None
    if args.report_reserve or args.capture_production:
        # 仅在独立冒烟进程注入候选。仍使用真实规划、执行、报告及主对话入口。
        from ai_agent.runtime import delegated_reading, task_delegation

        if len(args.cases) != 1:
            parser.error("diagnostics require exactly one case")
        if args.capture_production and args.report_reserve:
            parser.error("production capture and experimental reserve are mutually exclusive")
        build_graph = delegated_reading.build_graph
        diagnostics = {"trace": [], "model_calls": []}

        def instrumented_graph(root, scope, ask, trace, **kwargs):
            async def capture(invoke, stage, messages, tools=()):
                start = perf_counter()
                entry = {"stage": stage, "tool_count": len(tools), "start": start,
                         "messages": [message.model_dump(mode="json") for message in messages]}
                diagnostics["model_calls"].append(entry)
                try:
                    reply = await invoke()
                    entry.update(content=reply.content, tool_calls=reply.tool_calls, usage=reply.usage_metadata)
                    return reply
                except BaseException as error:
                    entry["outcome"] = type(error).__name__
                    raise
                finally:
                    entry["seconds"] = perf_counter() - start

            async def timed_ask(role, stage, messages, tools=()):
                if args.report_no_thinking and role == "auxiliary" and not tools:
                    # Only the no-tool report request changes in the historical comparison.
                    model = delegated_reading.create_llm(role=role).bind(extra_body={"enable_thinking": False})
                    return await capture(lambda: model.ainvoke(messages), stage, messages, tools)
                return await capture(lambda: ask(role, stage, messages, tools), stage, messages, tools)

            if args.capture_production:
                fast_report = kwargs["fast_report"]

                async def timed_fast_report(messages):
                    return await capture(lambda: fast_report(messages), "fast_report", messages)

                kwargs["fast_report"] = timed_fast_report
                return build_graph(root, scope, timed_ask, diagnostics["trace"], **kwargs)

            # 显式预留参数用于重跑历史实验。默认生产预算及快速回调见真实工具入口。
            kwargs["worker_timeout"] = None
            kwargs["branch_report_reserve"] = args.report_reserve
            kwargs["fast_report"] = None
            return build_graph(root, scope, timed_ask, diagnostics["trace"], **kwargs)

        delegated_reading.build_graph = instrumented_graph
        if args.report_reserve:
            task_delegation.DELEGATION_TIMEOUT_SECONDS = args.turn_timeout
    root = str(args.materials.resolve())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor() as pool:
        list(pool.map(lambda case: run(root, output, case, diagnostics, main_plans=args.main_plans,
                                       production_prompt=args.production_prompt),
                      args.cases or CASES))
