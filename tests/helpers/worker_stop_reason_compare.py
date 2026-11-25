"""Freeze real textbook receipts and compare the worker's next decision."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_from_dict

from ai_agent.runtime import delegated_reading as graph


class ToolsCaptured(Exception):
    pass


async def compare(record, materials, output, repeats, live_initial=False):
    if live_initial:
        calls = record["delegation_diagnostics"]["model_calls"]
        plan = json.loads(next(item["content"] for item in calls if item["stage"] == "planner"))
        task = next(item for item in plan["tasks"] if item["id"] == "T2")
        first = next(item["messages"] for item in calls if item["stage"] == "T2")
        scope = json.loads(first[-1]["content"])["authorized_scope"]
        question = record["user_input"]
    else:
        task = next(item for item in record["tasks"] if item["id"] == "T2")
        first = record["first_requests"]["T2"]
        scope, question = record["scope"], record["question"]
    original = task["instruction"]
    scope = {key: set(value) if value is not None else None for key, value in scope.items()}
    captured = []

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [task]}))
        captured.extend(tools)
        raise ToolsCaptured()

    try:
        await graph.build_graph(materials, scope, ask, [], roomy=True, reuse_retrieval=True,
                                focused_planning=True).ainvoke({"question": question})
    except ToolsCaptured:
        pass
    assert captured
    initial = messages_from_dict([{"type": item["type"], "data": item}
                                  for item in first])
    history = list(initial)
    trace = iter(item for item in record.get("trace", ()) if item.get("task") == "T2")
    for event in (item for item in record.get("calls", ()) if item["stage"] == "T2"):
        response = event["response"]
        history.append(AIMessage(content=response["content"], tool_calls=response["tool_calls"]))
        reached_page = False
        for call in response["tool_calls"]:
            receipt = next(trace)
            assert call["name"] == receipt["tool"] and call["args"] == receipt["args"]
            history.append(ToolMessage(content=json.dumps(receipt["result"], ensure_ascii=False),
                                       tool_call_id=call["id"]))
            reached_page |= call["name"] == "read_pdf" and any(
                item.get("page") == 85 for item in receipt["result"].get("evidence", ()))
        if reached_page:
            break
    else:
        if not live_initial:
            raise ValueError("The recording never read page 85")

    location_clause = "（写明 PDF 页序或书内印刷页码）+ 章节号/小节标题"
    backup_clause = "，另附 1 处备选出处"
    if live_initial:
        variants = {"baseline": original, "focused": (
            "从《Hello-Agents》找出可核对的正文，说明 ReAct 的 Observation 如何反馈给模型，"
            "以及这种反馈解决什么问题。"
        )}
    else:
        assert location_clause in original and backup_clause in original
        variants = {
            "baseline": original,
            "no-backup": original.replace(backup_clause, ""),
            "physical-page-only": original.replace(location_clause, "（PDF 物理页即可）"),
            "concise": "从《Hello-Agents》找一处正文解释 ReAct 中 Observation 的作用，交付解释和可核对出处。",
        }

    async def arm(name, instruction, trial):
        payload = json.loads(initial[-1].content)
        payload["assignment"]["instruction"] = instruction
        view = graph.project_task_history([initial[0], HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                                           *history[len(initial):]])
        start = perf_counter()
        data = {"request_messages": [item.model_dump(mode="json") for item in view],
                "tool_names": [item.name for item in captured]}
        try:
            async with asyncio.timeout(60):
                reply = await graph.create_llm(role="auxiliary").bind_tools(captured).ainvoke(view)
            data.update(content=reply.content, tool_calls=reply.tool_calls, usage=reply.usage_metadata)
        except TimeoutError:
            data["error"] = "timeout"
        data["seconds"] = round(perf_counter() - start, 2)
        (output / f"{name}-{trial}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
        print(name, trial, data["seconds"], [call["name"] for call in data.get("tool_calls", ())], flush=True)

    for trial in range(1, repeats + 1):
        await asyncio.gather(*(arm(name, instruction, trial) for name, instruction in variants.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("record", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--live-initial", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(compare(json.loads(args.record.read_text(encoding="utf8")),
                        args.materials.resolve(), args.output, args.repeats, args.live_initial))
