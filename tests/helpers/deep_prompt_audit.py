"""Compare isolated prompt changes against the current C dispatch path."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tests.helpers import delegated_reading as graph
from tests.helpers.deep_learning_scenarios import CASES, FIRST, SECOND
from tests.helpers.deep_role_experiment import COMPLEX, invoke, save
from tests.helpers.report_delivery import deliver_reports


PLANNER = """为用户所需的交付安排子任务。可以用一个子任务完成，也可以按需要分工。
每项 instruction 写清要交付什么，并带上完成该任务所需的背景和适用的用户约束。
返回 JSON：tasks 含 id、instruction、sources。sources 使用 authorized_scope 中的资料名。
有前置任务时填写 depends_on（任务 ID 数组）；必须取得指定产物时填写 required_artifacts（前置任务 ID 到产物名数组的映射）。"""
WORKER = "按需使用资料工具或上游发现。有影响交付的缺口时，在报告中说明。"
PROGRESS = "还需要继续取证时，可先用 save_findings 保存已有依据的发现。"
DELIVERY = """根据用户请求与子任务报告完成最终答复。
区分资料事实、转述和推断；证据缺口影响结论时说明。
需要核对依据时，可回读相关原文。"""
ALLOCATION = "最终答复由汇总阶段完成；仅在单独执行能补充证据或产生独立成果时安排子任务。"
DIRECT = "直接完成用户请求，按用户指定的内容范围和输出形式交付。"
DIRECT_FIDELITY = DIRECT + " 改写用户提供的材料时，保留原意和各陈述之间的关系。"
HOLDOUT = {
    "inline-rewrite": "以下是我的笔记：『工具结果靠调用编号对应回原请求；观察是推理循环拿到的外部反馈。』只根据这两句，替我改成一句适合课堂讲的口语。不查文件。",
    "inline-practice": "请根据我给出的两个概念出一道口头判断题，不给答案：短期记忆保留最近的对话，长期记忆可以检索旧记录。只用我写的这两点，不查文件。",
    "coverage": "只依据已读的两页，用恰好两行写给初学者的复习卡。第一行解释 tool_call_id，第二行解释 Observation，每行加书名和 PDF 页码。不要举例、比较或加标题。",
    "practice": "只读《AI Agents in Depth》PDF 第36页。我已经理解四种角色，只给一道口头练习让我判断某条消息的角色，不给答案或提示，不需要重新讲教材。",
}


def read(directory, name):
    return json.loads((directory / f"{name}.json").read_text(encoding="utf8"))


async def compare(materials, saved, output, stages):
    output.mkdir(parents=True, exist_ok=True)
    capacity = asyncio.Semaphore(8)

    async def record(stage, name, candidate, question, operation):
        async with capacity:
            started = perf_counter()
            row = {"question": question, "calls": [], "trace": []}

            async def ask(role, step, messages, tools=()):
                start = perf_counter()
                entry = {"stage": step, "request": [message.model_dump(mode="json") for message in messages],
                         "tools": [tool.name for tool in tools]}
                row["calls"].append(entry)
                reply = await invoke(role, step, messages, tools)
                entry.update(seconds=perf_counter() - start, usage=reply.usage_metadata,
                             response=reply.content, tool_calls=reply.tool_calls)
                return reply

            try:
                async with asyncio.timeout(110):
                    row.update(await operation(candidate, ask, row["trace"]))
            except (ValueError, TimeoutError) as error:
                row["error"] = type(error).__name__
            row["seconds"] = perf_counter() - started
            save(output, f"{stage}-{name}-{'candidate' if candidate else 'baseline'}", row)

    def pairs(stage, name, question, operation):
        return [record(stage, name, candidate, question, operation) for candidate in (False, True)]

    def plan_operation(messages, scope, *, allocation=False):
        async def run(candidate, ask, trace):
            prompt = PLANNER if candidate else graph.FOCUSED_PLANNER
            if allocation:
                prompt = f"{prompt}\n{ALLOCATION}"
            reply = await ask("standard", "planner", [SystemMessage(content=f"{prompt}\n{graph.PLANNER_ROLES}"), *messages])
            tasks = graph.parse(reply.content, graph.RoomyPlan).tasks
            graph.validate(tasks, scope)
            return {"tasks": [task.model_dump(exclude_defaults=True) for task in tasks]}
        return run

    def worker_operation(question, scope, tasks):
        async def run(candidate, ask, trace):
            async def dispatch(role, step, messages, tools=()):
                if step == "planner":
                    return AIMessage(content=json.dumps({"tasks": tasks}, ensure_ascii=False))
                if step == "deliver":
                    return AIMessage(content="本组只观察执行者。")
                if candidate and graph.WORKER_CONTRACT in messages[0].content:
                    # 仅换正常执行提示。无工具收尾、命名产物规则和工具证据沿用原流程。
                    system = messages[0].content.replace(graph.WORKER_CONTRACT, WORKER).replace("\n" + PROGRESS, "")
                    messages = [messages[0].model_copy(update={"content": system}), *messages[1:]]
                return await ask(role, step, messages, tools)

            workflow = graph.build_graph(materials, scope, dispatch, trace, roomy=True, reuse_retrieval=True,
                                         progressive_reports=True, shared_pdf_reads=True, model_request_timeout=60)
            result = await workflow.ainvoke({"question": question})
            return {key: result[key] for key in ("tasks", "results", "events")}
        return run

    def delivery_operation(previous, question=None, *, instructions_override=None):
        question = previous["question"] if question is None else question
        store = {}
        for step in previous["trace"]:
            result = step.get("result", {})
            if not isinstance(result, dict):
                continue
            for item in [result, *result.get("evidence", [])]:
                if "evidence_id" in item and "text" in item:
                    store[item["evidence_id"]] = item

        async def run(candidate, ask, trace):
            instructions = instructions_override or (DELIVERY if candidate else graph.REPORT_DELIVERY)
            result = await deliver_reports(question, previous["results"], store, ask, trace, instructions)
            return {"answer": result.text, "available_evidence": sorted(store)}
        return run

    jobs = []
    def add_plan_case(name, question, messages, scope):
        if "planner" in stages:
            jobs.extend(pairs("planner", name, question, plan_operation(messages, scope)))
        if "allocation" in stages:
            jobs.append(record("allocation", name, True, question, plan_operation(messages, scope, allocation=True)))

    if "planner" in stages or "allocation" in stages:
        for name in ("exercise", "independent", "correction"):
            previous = read(saved, f"{name}-plan")
            request = previous["first_requests"]["planner"][1:]
            payload = json.loads(request[-1]["content"])
            scope = {path: set(pages) for path, pages in payload["authorized_scope"].items()}
            messages = [HumanMessage(content=item["content"]) if item["type"] == "human"
                        else AIMessage(content=item["content"]) for item in request]
            add_plan_case(name, previous["question"], messages, scope)
        question, scope = COMPLEX
        message = HumanMessage(content=json.dumps({"question": question, "authorized_scope": {
            path: sorted(pages) for path, pages in scope.items()}}, ensure_ascii=False))
        add_plan_case("dependency", question, [message], scope)

    if "worker" in stages:
        previous = read(saved, "independent-plan")
        jobs.extend(pairs("worker", "independent", previous["question"], worker_operation(
            previous["question"], {FIRST: {36}, SECOND: {85}}, previous["plan"])))
        question, scope, tasks = CASES["dependency"]
        jobs.extend(pairs("worker", "dependency", question, worker_operation(question, scope, tasks)))

    if "delivery" in stages:
        for name in ("exercise", "independent"):
            previous = read(saved, f"{name}-C")
            jobs.extend(pairs("delivery", name, previous["question"], delivery_operation(previous)))

    if "holdout" in stages:
        for name in ("inline-rewrite", "inline-practice"):
            question = HOLDOUT[name]
            message = HumanMessage(content=json.dumps({"question": question, "authorized_scope": {}}, ensure_ascii=False))
            jobs.extend(pairs("holdout-plan", name, question, plan_operation([message], {})))
        for name, previous in (("coverage", "independent-C"), ("practice", "exercise-C")):
            question = HOLDOUT[name]
            jobs.extend(pairs("holdout-delivery", name, question, delivery_operation(read(saved, previous), question)))

    if "direct" in stages or "fidelity" in stages:
        for name in ("inline-rewrite", "inline-practice"):
            question = HOLDOUT[name]
            # 用空报告模拟跳过子任务，由用户消息提供全部素材。
            previous = {"question": question, "trace": [], "results": {}}
            if "direct" in stages:
                operation = delivery_operation(previous)
                jobs.append(record("direct", name, True, question, operation))
                operation = delivery_operation(previous, instructions_override=DIRECT)
                jobs.append(record("direct-focused", name, True, question, operation))
            if "fidelity" in stages:
                operation = delivery_operation(previous, instructions_override=DIRECT_FIDELITY)
                jobs.append(record("direct-fidelity", name, True, question, operation))

    await asyncio.gather(*jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("saved", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("stages", nargs="+", choices=("planner", "worker", "delivery", "allocation", "holdout", "direct", "fidelity"))
    args = parser.parse_args()
    asyncio.run(compare(args.materials.resolve(), args.saved, args.output, args.stages))
