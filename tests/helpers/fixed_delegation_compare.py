"""Run the same two assignments with the observed and shorter worker instructions."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from ai_agent.runtime import delegated_reading, task_delegation


CONCISE = (
    "依据《AI Agents in Depth》正文，解释工具结果如何与对应的工具调用配对。",
    "依据《Hello-Agents》正文，解释工具执行后产生的 Observation 如何反馈给模型。",
)


def run(materials, reference, output, variant):
    record = json.loads(reference.read_text(encoding="utf8"))
    tasks = next(row["args"]["tasks"] for row in record["tool_results"]
                 if row["tool_name"] == "delegate_learning_task")
    assert len(tasks) == len(CONCISE)
    if variant == "concise":
        tasks = [{**task, "instruction": instruction} for task, instruction in zip(tasks, CONCISE)]
    trace, calls = [], []
    build = delegated_reading.build_graph

    def instrumented(root, scope, ask, _, **kwargs):
        async def timing(stage, invoke):
            entry = {"stage": stage}
            calls.append(entry)
            started = perf_counter()
            try:
                reply = await invoke()
                entry["usage"] = reply.usage_metadata
                return reply
            finally:
                entry["seconds"] = round(perf_counter() - started, 2)

        async def timed_ask(role, stage, messages, tools=()):
            return await timing(stage, lambda: ask(role, stage, messages, tools))

        fast = kwargs["fast_report"]

        async def timed_fast(messages):
            return await timing("fast_report", lambda: fast(messages))

        kwargs["fast_report"] = timed_fast
        return build(root, scope, timed_ask, trace, **kwargs)

    delegated_reading.build_graph = instrumented
    try:
        tool = task_delegation.build_learning_delegation_tool(materials, record["user_input"], main_plans=True)
        started = perf_counter()
        result = json.loads(tool.invoke({"tasks": tasks}))
        seconds = round(perf_counter() - started, 2)
    finally:
        delegated_reading.build_graph = build
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "variant": variant, "user_input": record["user_input"], "tasks": tasks, "seconds": seconds,
        "trace": trace, "model_calls": calls, "result": result,
    }, ensure_ascii=False, indent=2), encoding="utf8")
    print(variant, seconds, [(key, row["status"]) for key, row in
                            result.get("data", {}).get("reports", {}).items()], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials")
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("variant", choices=("observed", "concise"))
    args = parser.parse_args()
    run(args.materials, args.reference, args.output, args.variant)
