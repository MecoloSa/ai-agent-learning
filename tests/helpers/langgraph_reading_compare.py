"""比较现有单轮问答与无审查的动态多任务问答。"""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ai_agent.config import create_llm, create_llm_with_tools
from ai_agent.main import _invoke_interactive_turn
from ai_agent.resources.tools import build_read_pdf_pages_tool
from tests.helpers.reading_dependency import execute_ready, validate_plan

BOOKS = ("AI-Agents-in-Depth-zh-CN.pdf", "Hello-Agents-V1.0.3-20260717.pdf")
CASES = {
    "single": f"只读 {BOOKS[0]} 的 PDF 第36页，解释工具结果怎样关联工具调用。",
    "parallel": (
        f"读取 {BOOKS[0]} 的 PDF 第36页和 {BOOKS[1]} 的 PDF 第85页。"
        "两页分别取证，都取得后再比较 API tool 消息与 ReAct Observation。"
        "另外只根据第一篇出一道口头练习，不用等第二篇，不给答案。"
    ),
    "missing": (
        f"只读 {BOOKS[0]} 的 PDF 第36页和 {BOOKS[1]} 的 PDF 第9999页。"
        "有两页的正文才做跨书比较，任一页不可用就明确说明缺口，并解释已取得的内容。"
    ),
}


class Task(BaseModel):
    id: str
    goal: str
    depends_on: list[str]
    # 取证任务有书名和页码。纯解释或练习只消费依赖，不再次打开教材。
    book: str | None
    page: int | None = Field(ge=1)


class Plan(BaseModel):
    tasks: list[Task]


class State(TypedDict, total=False):
    question: str
    tasks: list[dict]
    results: dict[str, dict]
    events: list[dict]
    answer: str


def build_graph(read, ask):
    async def plan(state: State):
        raw = await ask("auxiliary", "planner", (
            "生成当前问题的任务图。只允许每个取证任务读取一本书的一页。"
            "解释或练习节点的 book、page 为 null，并声明实际依赖。"
            "返回 JSON 对象，tasks 每项字段为 id、goal、depends_on、book、page。"
        ), {"question": state["question"], "available_books": BOOKS})
        raw = raw.strip()
        if raw.startswith("```") and raw.endswith("```"):
            raw = raw.partition("\n")[2].removesuffix("```").strip()
        tasks = [task.model_dump() for task in Plan.model_validate_json(raw).tasks]
        validate_plan(tasks, BOOKS)
        return {"tasks": tasks}

    async def execute(state: State):
        async def worker(task, inputs):
            evidence = [item for result in inputs.values() for item in result["evidence"]]
            if task["book"]:
                try:
                    receipt = json.loads(await asyncio.to_thread(read.invoke, {
                        "relative_path": task["book"], "start_page": task["page"], "end_page": task["page"],
                    }))
                except (ValueError, OSError) as exc:
                    return {"status": "unavailable", "reason": type(exc).__name__}
                pages = receipt.get("data", {}).get("pages", [])
                if not receipt.get("ok") or not pages or not pages[0].get("text"):
                    return {"status": "unavailable", "reason": "没有取得正文"}
                evidence.append({"book": task["book"], "page": task["page"], "text": pages[0]["text"]})
            answer = await ask("auxiliary", task["id"], "完成当前子任务，依据所给证据简洁作答。", {
                "task": task, "inputs": inputs, "evidence": evidence,
            })
            if not answer.strip():
                return {"status": "unavailable", "reason": "执行模型没有返回正文"}
            return {"status": "done", "answer": answer, "evidence": evidence}

        results, events = await execute_ready(state["tasks"], worker)
        return {"results": results, "events": events}

    async def deliver(state: State):
        # 每份真实回执只传一份，避免比较节点和取证节点重复携带同一页正文。
        evidence = {}
        reports = []
        for task in state["tasks"]:
            result = state["results"][task["id"]]
            reports.append({"task": task, **{k: v for k, v in result.items() if k != "evidence"}})
            for item in result.get("evidence", []):
                evidence[item["book"], item["page"]] = item
        answer = await ask("standard", "deliver", "根据实际证据和缺口回答用户。", {
            "question": state["question"], "reports": reports, "evidence": list(evidence.values()),
        })
        return {"answer": answer}

    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("deliver", deliver)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "deliver")
    graph.add_edge("deliver", END)
    return graph.compile()


async def compare(materials, output, repeat):
    output.mkdir(parents=True, exist_ok=True)
    read = build_read_pdf_pages_tool(str(materials.resolve()))

    async def run(case, question, variant, index):
        metrics = []
        start = perf_counter()

        async def ask(role, stage, instructions, payload):
            before = perf_counter()
            response = await asyncio.to_thread(create_llm(role=role).invoke, [
                SystemMessage(content=instructions),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ])
            metrics.append({"stage": stage, "seconds": perf_counter()-before, "usage": response.usage_metadata})
            return response.content if isinstance(response.content, str) else ""

        def single_turn(trace):
            return _invoke_interactive_turn(
                request_messages=[SystemMessage(content="依据用户指定的书和 PDF 物理页取证。取证失败或证据不足时，说明对结论的影响。"),
                                  HumanMessage(content=question)],
                user_input=question, plain_llm=create_llm(), tool_llm=create_llm_with_tools([read]),
                retrieval_tool=None, extra_tools=(read,), audit_trace=trace,
            )

        trace = {}
        try:
            if variant == "baseline":
                result = {"answer": await asyncio.to_thread(single_turn, trace), "trace": trace}
            else:
                result = await build_graph(read, ask).ainvoke({"question": question})
        except Exception as exc:
            result = {"error": type(exc).__name__}
        result.update(case=case, variant=variant, trial=index, question=question,
                      seconds=perf_counter()-start, calls=metrics)
        path = output / f"{case}-{variant}-{index}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
        print(path.name, round(result["seconds"], 2), result.get("error", "ok"), flush=True)

    for index in range(1, repeat + 1):
        for case, question in CASES.items():
            await asyncio.gather(*(run(case, question, variant, index) for variant in ("baseline", "candidate")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("materials", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(compare(args.materials, args.output, args.repeat))
