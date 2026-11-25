"""Compare the same Flash report request with and without provider thinking."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage, messages_from_dict

from ai_agent.runtime import delegated_reading as graph


def report_messages(record, task_id):
    if "delegation_diagnostics" in record:
        saved = next(call["messages"] for call in reversed(record["delegation_diagnostics"]["model_calls"])
                     if call["stage"] == task_id and call["tool_count"] == 0)
        return messages_from_dict([{"type": item["type"], "data": item} for item in saved])

    task = next(task for task in record["tasks"] if task["id"] == task_id)
    event = next(event for event in record["trace"]
                 if event["task"] == task_id and event["tool"] == "branch_closeout_evidence")
    store = {item["evidence_id"]: item for row in record["trace"]
             for item in row.get("result", {}).get("evidence", ())}
    return [
        SystemMessage(content="\n".join(filter(None, (graph.ROLE_OPENERS.get(task.get("role")),
                     graph.REPORT_FORMAT, graph.WORKER_ASSIGNMENT_RULE)))),
        HumanMessage(content=json.dumps({
            "assignment": task, "upstream_reports": {},
            "evidence": [store[key] for key in event["evidence_ids"]],
        }, ensure_ascii=False)),
    ]


async def compare(record, output, task_id, repeats, timeout, compact=False):
    messages = report_messages(record, task_id)
    payload = json.loads(messages[-1].content)
    allowed_ids = {item["evidence_id"] for item in payload["evidence"]}

    async def arm(name, trial):
        model = graph.create_llm(role="auxiliary")
        if name.endswith("no-thinking"):
            model = model.bind(extra_body={"enable_thinking": False})
        elif name == "low-thinking":
            model = model.bind(extra_body={"enable_thinking": True, "reasoning_effort": "low"})
        start = perf_counter()
        view = messages
        if name.startswith("compact-"):
            assignment = {**payload["assignment"], "instruction": (
                "依据《Hello-Agents》说明 ReAct 的 Observation 如何反馈给模型，"
                "以及这种反馈解决什么问题。给出可核对的出处。"
            )}
            view = [*messages[:-1], HumanMessage(content=json.dumps(
                {**payload, "assignment": assignment}, ensure_ascii=False))]
        data = {"task_id": task_id, "request_messages": [m.model_dump(mode="json") for m in view]}
        try:
            async with asyncio.timeout(timeout):
                reply = await model.ainvoke(view)
            data.update(content=reply.content, usage=reply.usage_metadata,
                        finish_reason=reply.response_metadata.get("finish_reason"))
            report = graph.parse(reply.content, graph.Report)
            ids = {key for item in report.findings for key in item.evidence_ids}
            data.update(findings=len(report.findings), unverified_ids=sorted(ids - allowed_ids))
        except (TimeoutError, ValueError) as error:
            data["error"] = type(error).__name__
        except Exception as error:
            data["error"] = type(error).__name__
        data["seconds"] = round(perf_counter() - start, 2)
        (output / f"{name}-{trial}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
        print(name, trial, data["seconds"], data.get("usage"), data.get("findings"), data.get("error"), flush=True)

    arms = ("default", "low-thinking", "no-thinking")
    if compact:
        arms += ("compact-default", "compact-no-thinking")
    for trial in range(1, repeats + 1):
        await asyncio.gather(*(arm(name, trial) for name in arms))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", default="T2")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(compare(json.loads(args.record.read_text(encoding="utf8")), args.output,
                        args.task, args.repeats, args.timeout, args.compact))
