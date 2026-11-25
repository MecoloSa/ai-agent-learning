"""Compare worker views after fixed, real PDF readings."""

import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

from langchain_core.messages import AIMessage, ToolMessage

from tests.helpers import delegated_reading as probe


CASES = {
    "quote": (
        (41, 44, 41),
        "第41页用于背景。准确引用 PDF 第44页中回填 tool_call_id 的一行代码，说明用途并提交报告。",
    ),
    "compare": (
        (36, 41, 44, 36),
        "比较 PDF 第36页的消息角色定义、第41页的两次工具调用示例、第44页的回填实现。"
        "分别给出三页的证据，并解释从协议定义到代码执行的关系。",
    ),
}


async def run(root, output, name, latest_only):
    book = probe.BOOKS[0]
    prefix, question = CASES[name]
    trace, calls = [], []
    started = perf_counter()

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [{
                "id": "A", "instruction": question, "sources": [book], "depends_on": [],
            }]}))
        if stage == "deliver":
            return AIMessage(content="探针只评估子任务交付。")
        count = sum(isinstance(message, ToolMessage) for message in messages)
        if count < len(prefix):
            page = prefix[count]
            return AIMessage(content="", tool_calls=[{
                "id": f"prefix-{count}", "name": "read_pdf",
                "args": {"relative_path": book, "start_page": page, "end_page": page},
            }])
        visible = []
        for message in messages:
            if isinstance(message, ToolMessage):
                visible.extend(row["page"] for row in json.loads(message.content).get("evidence", [])
                               if "page" in row and row.get("text"))
        model = probe.create_llm(role=role)
        response = await (model.bind_tools(tools) if tools else model).ainvoke(messages)
        calls.append({"visible_pages": visible, "tools": response.tool_calls, "content": response.content,
                      "usage": response.usage_metadata})
        return response

    graph = probe.build_graph(
        root, {book: set(prefix)}, ask, trace, roomy=True, model_request_timeout=60, latest_only=latest_only,
    )
    try:
        async with asyncio.timeout(60):
            result = await graph.ainvoke({"question": question})
    except TimeoutError:
        result = {"observation_stopped": True}
    result.update(question=question, prefix=prefix, calls=calls, trace=trace, seconds=perf_counter() - started)
    arm = "latest" if latest_only else "continuous"
    (output / f"{name}-{arm}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(name, arm, round(result["seconds"], 2), result.get("delivery_status", "observation_stopped"), flush=True)


async def main(root, output):
    output.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(*(run(root, output, name, latest_only) for name in CASES for latest_only in (True, False)))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2])))
