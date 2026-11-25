"""Run controlled worker-context and planner-prompt comparisons."""

import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tests.helpers import delegated_reading as graph
from tests.helpers.deep_learning_scenarios import CASES, FIRST, assignment
from tests.helpers.deep_role_experiment import COMPLEX, invoke, save
from tests.helpers.report_delivery import deliver_reports


PLANNER_HEAD = """需要派发时，根据交付目标划分可独立完成的工作。每项 instruction 写清交付物、完成所需素材和适用的约束。
有实际依赖时才安排先后；已有结果足以完成的综合或表达，由最终答复处理。"""
ARTIFACT_BOUNDARY = "下游需要分别使用的内容，作为不同交付物；需要引用其中某项时，为该项命名。"
DELIVERY_EVIDENCE_LIMIT = "说明后果时，区分资料展示的行为与未验证的结果；缺少依据时说明适用条件。"
WORKER_EVIDENCE_LIMIT = "报告中区分资料直接展示的行为和推断；未覆盖的运行环节，写明推断成立的前提。"


def project_context(payload, *, references=False):
    requested = payload["assignment"].get("required_artifacts", {})
    reports = payload["upstream_reports"]
    if not requested or set(reports) - requested.keys():
        # 普通依赖尚未表达所需产物，沿用整份交接。
        return payload
    selected = {}
    for task, report in reports.items():
        selected[task] = {**report}
        for key in ("findings", "deliverables"):
            if key in report:
                selected[task][key] = [item for item in report[key] if item.get("artifact") in requested[task]]
    ids = {key for report in selected.values() for item in report.get("findings", []) for key in item["evidence_ids"]}
    evidence_key = "upstream_evidence" if "upstream_evidence" in payload else "evidence"
    evidence = payload[evidence_key]
    if evidence_key == "upstream_evidence":
        evidence = [item for item in evidence if item["evidence_id"] in ids]
        if references:
            # 只缩短请求副本，工具原有的证据集合不变。独立收尾无回读工具，继续给正文。
            evidence = [{key: value for key, value in item.items() if key != "text"} for item in evidence]
    return {**payload, "upstream_reports": selected, evidence_key: evidence}


def separate_recorded_draft(payload):
    # 历史样本有已知分段标签。只用于构造受控对照，运行时不应靠解析标签拆任务报告。
    reports = json.loads(json.dumps(payload["upstream_reports"], ensure_ascii=False))
    for report in reports.values():
        for item in list(report.get("deliverables", [])):
            title = "【最终输出：口头练习题干（仅题干）】"
            marker = "【以下为审核用说明，不属于最终输出】"
            if item["content"].startswith(title) and marker in item["content"]:
                stem, note = item["content"].removeprefix(title).split(marker, 1)
                item["content"] = stem.strip()
                report["deliverables"].append({"artifact": "命题说明", "content": note.strip()})
    return {**payload, "upstream_reports": reports}


async def worker_compare(root, output, source, *, packaging=False):
    old = json.loads(source.read_text(encoding="utf8"))
    question, scope, tasks = old["question"], {FIRST: {36}}, old["tasks"]
    consumer = tasks[-1]["id"]
    recordings = {task["id"]: [call["response"] for call in old["calls"] if call["stage"] == task["id"]]
                  for task in tasks[:-1]}

    async def arm(mode):
        started = perf_counter()
        trace, calls = [], []
        replay = {task: iter(responses) for task, responses in recordings.items()}

        async def ask(role, stage, messages, tools=()):
            if stage == "planner":
                return AIMessage(content=json.dumps({"tasks": tasks}, ensure_ascii=False))
            if stage in replay:
                response = next(replay[stage])
                return AIMessage(content=response["content"], tool_calls=response["tool_calls"])
            if stage == "deliver":
                return AIMessage(content="本组只评估审核子任务。")
            if stage == consumer and mode != "full":
                messages = list(messages)
                for i, message in enumerate(messages):
                    if message.type != "human":
                        continue
                    payload = json.loads(message.content)
                    if "upstream_reports" in payload:
                        if mode == "separated":
                            payload = separate_recorded_draft(payload)
                        projected = project_context(payload, references=mode == "references")
                        messages[i] = message.model_copy(update={"content": json.dumps(projected, ensure_ascii=False)})
                        break
            start = perf_counter()
            row = {"stage": stage, "messages": [m.model_dump(mode="json") for m in messages],
                   "tools": [tool.name for tool in tools]}
            calls.append(row)
            reply = await invoke(role, stage, messages, tools)
            row.update(seconds=perf_counter()-start, usage=reply.usage_metadata, tool_calls=reply.tool_calls,
                       response=reply.content)
            return reply

        workflow = graph.build_graph(root, scope, ask, trace, roomy=True, reuse_retrieval=True,
                                     shared_pdf_reads=True, model_request_timeout=60)
        try:
            async with asyncio.timeout(110):
                result = await workflow.ainvoke({"question": question})
            row = {"results": result["results"], "consumer": consumer}
        except TimeoutError:
            row = {"error": "observation_timeout"}
        save(output, f"handoff-{mode}", {**row, "question": question, "plan": tasks,
             "trace": trace, "calls": calls, "seconds": perf_counter()-started})

    modes = ("selected", "separated") if packaging else ("full", "selected", "references")
    await asyncio.gather(*(arm(mode) for mode in modes))


async def planner_compare(output, selected=(), *, packaging=False):
    cases = [(name, CASES[name][0], CASES[name][1]) for name in
             ("exercise", "parallel", "inline-rewrite", "draft-review-live")]
    cases.append(("dependency", *COMPLEX))
    if selected:
        cases = [case for case in cases if case[0] in selected]

    async def arm(name, question, scope, candidate):
        started = perf_counter()
        fields = graph.FOCUSED_PLANNER.splitlines()[-1]
        if packaging:
            head, dependency = PLANNER_HEAD.splitlines()
            extra = f"{ARTIFACT_BOUNDARY}" if candidate else ""
            base = f"{head}{extra}\n{dependency}\n{fields}"
        else:
            base = f"{PLANNER_HEAD}\n{fields}" if candidate else graph.FOCUSED_PLANNER
        prompt = "\n".join((base, graph.PLAN_ROUTE_RULE, graph.PLAN_GENERATION_RULE, graph.PLANNER_ROLES))
        payload = {"question": question, "authorized_scope": {path: sorted(pages) for path, pages in scope.items()}}
        row = {"question": question, "prompt": prompt, "input": payload}
        try:
            reply = await invoke("standard", "planner", [SystemMessage(content=prompt),
                                 HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
            row.update(raw=reply.text, usage=reply.usage_metadata)
            plan = graph.parse(reply.content, graph.RoomyPlan)
            graph.validate(plan.tasks, scope)
            row["plan"] = plan.model_dump(exclude_defaults=True)
        except (ValueError, TimeoutError) as error:
            row["error"] = type(error).__name__
        save(output, f"planner-{name}-{'candidate' if candidate else 'current'}", {**row, "seconds": perf_counter()-started})

    await asyncio.gather(*(arm(*case, candidate) for case in cases for candidate in (False, True)))


async def two_artifact_compare(root, output, *, live_delivery=False, missing_only=False, worker_boundary=False):
    question = "只读《AI Agents in Depth》PDF 第41、44页。根据第41页的调用编号对照和第44页的回填代码，说明天气结果如果填到时间调用的编号下会造成什么混淆。区分教材事实和推断。"
    tasks = [
        assignment("mapping", "读取第41页，交付调用编号与工具对照。", [FIRST], role="reader"),
        assignment("code", "读取第44页，交付结果回填代码。", [FIRST], role="reader"),
        assignment("explain", "用两份产物解释天气结果错填到时间编号会发生什么，区分教材与推断。", [],
                   ["mapping", "code"], {"mapping": ["mapping"], "code": ["code"]}, role="analyst"),
    ]
    reference = {
        "mapping": (41, "call_abc123 → get_current_time；call_def456 → get_weather。", "本页还讲了第二次请求要带上第一次的完整历史。"),
        "code": (44, '程序执行工具后，向 messages 追加 {"role": "tool", "tool_call_id": tool_call.id, "content": result}。',
                 "本页还说明模型不再请求调用工具时退出循环。"),
    }

    async def arm(separate, missing=False):
        started = perf_counter()
        trace, calls = [], []

        async def ask(role, stage, messages, tools=()):
            if stage == "planner":
                return AIMessage(content=json.dumps({"tasks": tasks}, ensure_ascii=False))
            if missing and stage == "code":
                # 保留相同用户要求与授权，模拟该分支没有交出指定产物。
                return AIMessage(content='{"findings":[],"gaps":["未取得第44页的回填代码"]}')
            if stage in reference:
                page, fact, background = reference[stage]
                if messages[-1].type != "tool":
                    return AIMessage(content="", tool_calls=[{"name": "read_pdf", "id": stage,
                                      "args": {"relative_path": FIRST, "start_page": page, "end_page": page}}])
                item, = json.loads(messages[-1].content)["evidence"]
                for expected in ({41: ("call_abc123", "call_def456", "get_current_time", "get_weather"),
                                  44: ("tool_call_id", "tool_call. id", "content", "result")}[page]):
                    if expected not in item["text"]:
                        raise ValueError(f"教材第{page}页无法复核固定样本的字段")
                evidence_ids = [item["evidence_id"]]
                findings = [{"statement": fact, "artifact": stage, "evidence_ids": evidence_ids},
                            {"statement": background, "artifact": "背景" if separate else stage, "evidence_ids": evidence_ids}]
                return AIMessage(content=json.dumps({"findings": findings, "gaps": []}, ensure_ascii=False))
            if stage == "deliver" and not live_delivery:
                return AIMessage(content="本组只评估推演子任务。")
            if stage == "explain":
                # 只改变下游执行者的交接副本，真实汇总沿用完整的交付契约。
                messages = list(messages)
                payload = json.loads(messages[1].content)
                view = project_context(payload)
                messages[1] = messages[1].model_copy(update={"content": json.dumps(view, ensure_ascii=False)})
                if worker_boundary:
                    system = f"{messages[0].content}\n{WORKER_EVIDENCE_LIMIT}"
                    messages[0] = messages[0].model_copy(update={"content": system})
            row = {"stage": stage, "messages": [m.model_dump(mode="json") for m in messages]}
            calls.append(row)
            begin = perf_counter()
            try:
                reply = await invoke(role, stage, messages, tools)
            finally:
                row["seconds"] = perf_counter() - begin
            row.update(usage=reply.usage_metadata, tool_calls=reply.tool_calls, response=reply.content)
            return reply

        workflow = graph.build_graph(root, {FIRST: {41, 44}}, ask, trace, roomy=True,
                                     reuse_retrieval=True, shared_pdf_reads=True, model_request_timeout=60)
        try:
            async with asyncio.timeout(145 if live_delivery else 110):
                result = await workflow.ainvoke({"question": question})
        except TimeoutError:
            result = {"delivery_status": "observation_stopped"}
        tag = "missing-artifact" if missing else ("separated" if separate else "bundled")
        save(output, f"dual-{tag}", {"question": question, "plan": tasks, "consumer": "explain",
             "trace": trace, "calls": calls, "results": result.get("results", {}),
             "events": result.get("events", []), "answer": result.get("answer"),
             "delivery_status": result.get("delivery_status"), "seconds": perf_counter()-started})

    if missing_only:
        await arm(False, missing=True)
    elif worker_boundary:
        await arm(True)
    else:
        await asyncio.gather(arm(False), arm(True))


async def main(root, output, source):
    output.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(worker_compare(root, output, source), planner_compare(output))


async def delivery_evidence_compare(output, source, *, report_only=False):
    recorded = json.loads(source.read_text(encoding="utf8"))
    store = {}
    for step in recorded["trace"]:
        result = step.get("result")
        if isinstance(result, dict):
            for item in result.get("evidence", [result]):
                if "evidence_id" in item and "text" in item:
                    store[item["evidence_id"]] = item

    async def arm(candidate, repeat):
        started = perf_counter()
        calls, trace = [], []
        instructions = graph.REPORT_DELIVERY
        reports = recorded["results"]
        if candidate and report_only:
            # 受控实验只替换一条已有报告，校验前缀和唯一性，默认流程不匹配案例措辞。
            reports = json.loads(json.dumps(reports, ensure_ascii=False))
            matches = [finding for finding in reports["explain"]["findings"]
                       if finding["statement"].startswith("【推断·错误为何会静默传播】")]
            if len(matches) != 1:
                raise ValueError("回放缺少唯一的待替换发现")
            matches[0]["statement"] = (
                "【教材】第44页展示的回填语句将 tool_call.id 和 result 写入消息，示例未展示内容与编号的匹配校验。"
                "【推断】如果这条错配消息被接收并保留，它可能随历史进入后续调用。"
                "【未验证】两页未展示完整框架校验或实际错配执行结果，无法判断会报错还是继续。"
            )
        elif candidate:
            instructions += f"\n{DELIVERY_EVIDENCE_LIMIT}"

        async def ask(role, stage, messages, tools=()):
            row = {"messages": [m.model_dump(mode="json") for m in messages]}
            calls.append(row)
            reply = await invoke(role, stage, messages, tools)
            row.update(usage=reply.usage_metadata, tool_calls=reply.tool_calls)
            return reply

        answer = await deliver_reports(recorded["question"], reports, store, ask, trace, instructions)
        suffix = f"-{repeat}" if report_only else ""
        save(output, f"delivery-{'candidate' if candidate else 'baseline'}{suffix}", {
            "question": recorded["question"], "answer": answer.text, "trace": trace, "calls": calls,
            "seconds": perf_counter()-started,
        })

    output.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(*(arm(candidate, repeat) for repeat in range(1, 3 if report_only else 2)
                           for candidate in (False, True)))


if __name__ == "__main__":
    materials, output = map(Path, sys.argv[1:3])
    if sys.argv[3:] == ["two-artifact-e2e"]:
        output.mkdir(parents=True, exist_ok=True)
        asyncio.run(two_artifact_compare(materials.resolve(), output, live_delivery=True))
    elif sys.argv[3:] == ["missing-artifact-e2e"]:
        output.mkdir(parents=True, exist_ok=True)
        asyncio.run(two_artifact_compare(materials.resolve(), output, live_delivery=True, missing_only=True))
    elif sys.argv[3:] == ["worker-boundary-e2e"]:
        output.mkdir(parents=True, exist_ok=True)
        asyncio.run(two_artifact_compare(materials.resolve(), output, live_delivery=True, worker_boundary=True))
    elif sys.argv[4:] == ["delivery-evidence"]:
        asyncio.run(delivery_evidence_compare(output, Path(sys.argv[3])))
    elif sys.argv[4:] == ["delivery-report-boundary"]:
        asyncio.run(delivery_evidence_compare(output, Path(sys.argv[3]), report_only=True))
    elif sys.argv[4:] == ["packaging"]:
        source = Path(sys.argv[3])
        async def run_packaging():
            output.mkdir(parents=True, exist_ok=True)
            await asyncio.gather(worker_compare(materials.resolve(), output, source, packaging=True),
                                 two_artifact_compare(materials.resolve(), output),
                                 planner_compare(output, ("draft-review-live", "dependency"), packaging=True))

        asyncio.run(run_packaging())
    else:
        asyncio.run(main(materials.resolve(), output, Path(sys.argv[3])))
