"""Isolated role, handoff and citation experiments with the existing graph."""

import asyncio
import json
import re
import sys
from pathlib import Path
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_agent.persistence.memory import ConversationMemory
from tests.helpers import delegated_reading as probe
from tests.helpers.deep_learning_scenarios import CASES, FIRST, SECOND, assignment
from tests.helpers.report_delivery import deliver_reports
from tests.helpers.shared_reading_evidence import CitationArchive


ROLES = probe.ROLES
ROLE_OPENERS = probe.ROLE_OPENERS
ROLE_PLANNER = """按用户需要的交付划分工作。每项 instruction 给出目标、交付物及适用的用户约束。用户已指定要查多少证据时沿用其范围。
按职责为每项选择 role：""" + " ".join(f"{key}：{value}" for key, value in ROLES.items()) + """
可以独立取证的任务并行。单独依赖上游产物时填写依赖。已有发现足以支持的比较和简短练习交给最终答复。
返回 JSON：tasks 含 id、role、instruction、sources（取 authorized_scope 的准确文件名）、depends_on。必需产物用 required_artifacts 标明，键为上游任务 ID，值为产物名数组。"""
COMPLEX = (
    "只读《AI Agents in Depth》PDF 第36、41、44页和《Hello-Agents》PDF 第85页。"
    "先分别从第一本第41、44页提取调用 ID 对照表和回填代码，再用这两份产物推演错填结果。"
    "同时用第36、85页分别说明 API 消息角色和 Observation。最后给出一份有出处的教学解释，不需要练习。",
    {FIRST: {36, 41, 44}, SECOND: {85}},
)


async def invoke(role, stage, messages, tools=()):
    model = probe.create_llm(role=role)
    if stage in {"planner", "deliver"}:
        model = model.model_copy(update={"model_name": probe.DEEP_READING_MODEL})
    if tools:
        model = model.bind_tools(tools)
    async with asyncio.timeout(60):
        return await model.ainvoke(messages)


def save(output, name, row):
    (output / f"{name}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
    print(name, round(row["seconds"], 2), row.get("error", "recorded"), flush=True)


async def planning(output):
    scenarios = [(name, CASES[name][0], CASES[name][1]) for name in ("open", "missing")]
    scenarios.append(("complex", *COMPLEX))

    async def arm(name, question, scope, typed):
        payload = {"question": question, "authorized_scope": {
            source: sorted(pages) if pages is not None else None for source, pages in scope.items()
        }}
        prompt = ROLE_PLANNER if typed else probe.FOCUSED_PLANNER
        started = perf_counter()
        row = {"question": question, "prompt": prompt}
        try:
            reply = await invoke("standard", "planner", [
                SystemMessage(content=prompt), HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ])
            row.update(raw=reply.text, usage=reply.usage_metadata)
            data = json.loads(reply.text.strip().removeprefix("```json").removeprefix("```")
                              .removesuffix("```").strip())
            probe.validate(probe.RoomyPlan.model_validate(data).tasks, scope)
            if typed and any(task.get("role") not in ROLES for task in data["tasks"]):
                raise ValueError("规划角色不在注册表中")
            row["plan"] = data
        except (TimeoutError, ValueError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["seconds"] = perf_counter() - started
        save(output, f"plan-{name}-{'roles' if typed else 'baseline'}", row)

    await asyncio.gather(*(arm(*case, typed) for case in scenarios for typed in (False, True)))


def forward_named_artifacts(context):
    requested = context["assignment"].get("required_artifacts", {})
    reports = context["upstream_reports"]
    selected = {
        source: {**report, "findings": (
            [row for row in report.get("findings", []) if row.get("artifact") in requested[source]]
            if source in requested else report.get("findings", [])
        )}
        for source, report in reports.items()
    }
    used_ids = {key for report in selected.values() for finding in report["findings"]
                for key in finding["evidence_ids"]}
    return {**context, "upstream_reports": selected, "upstream_evidence": [
        row for row in context["upstream_evidence"] if row["evidence_id"] in used_ids
    ]}


async def worker(root, output, opening=False):
    question, scope, tasks = CASES["dependency"]
    roles = {task["id"]: task["role"] for task in tasks}

    async def arm(mode):
        trace, calls, first_requests = [], [], {}
        began = perf_counter()

        async def ask(role, stage, messages, tools=()):
            if stage == "planner":
                return AIMessage(content=json.dumps({"tasks": tasks}, ensure_ascii=False))
            if stage == "deliver":
                return AIMessage(content="本实验只评估分支交接。")
            visible = list(messages)
            if opening and mode in {"role-first", "both"}:
                assigned_role = roles[stage]
                # 角色作为系统消息首段。保留已有工具与报告契约。
                visible[0] = visible[0].model_copy(update={
                    "content": ROLE_OPENERS[assigned_role] + "\n" + visible[0].content,
                })
            if mode in {"role", "role-handoff"}:
                assigned_role = roles[stage]
                visible[0] = visible[0].model_copy(update={"content": visible[0].content + "\n" + ROLES[assigned_role]})
            if mode == "role-handoff" and stage == "teach":
                for i, message in enumerate(visible):
                    if message.type != "human":
                        continue
                    context = json.loads(message.content)
                    if "upstream_reports" not in context:
                        continue
                    findings = [f for report in context["upstream_reports"].values()
                                for f in report["findings"] if f.get("artifact") == "call_mapping"]
                    context["upstream_reports"] = {"call_mapping": findings}
                    context.pop("user_request", None)
                    visible[i] = message.model_copy(update={"content": json.dumps(context, ensure_ascii=False)})
                    break
            if opening and mode in {"handoff", "both"} and stage == "teach":
                for i, message in enumerate(visible):
                    if message.type != "human":
                        continue
                    context = json.loads(message.content)
                    if "upstream_evidence" in context:
                        context = forward_named_artifacts(context)
                        visible[i] = message.model_copy(update={"content": json.dumps(context, ensure_ascii=False)})
                        break
            first_requests.setdefault(stage, [
                {"type": message.type, "content": message.content} for message in visible
            ])
            began_call = perf_counter()
            answer = await invoke(role, stage, visible, tools)
            calls.append({"stage": stage, "seconds": perf_counter() - began_call, "content": answer.content,
                          "tool_calls": answer.tool_calls, "usage": answer.usage_metadata})
            return answer

        # 此对照由 ask 独立控制角色开场，避免基线也带上开场。
        graph = probe.build_graph(root, scope, ask, trace, roomy=True, reuse_retrieval=True,
                                  shared_pdf_reads=True, model_request_timeout=60, worker_roles=False,
                                  worker_full_question=True)
        try:
            async with asyncio.timeout(90):
                result = await graph.ainvoke({"question": question})
        except TimeoutError:
            result = {"observation_stopped": True}
        result.update(question=question, first_requests=first_requests,
                      calls=calls, trace=trace, seconds=perf_counter()-began)
        save(output, f"worker-{mode}", result)

    modes = ("baseline", "role-first", "handoff", "both") if opening else ("baseline", "role", "role-handoff")
    await asyncio.gather(*(arm(mode) for mode in modes))


async def citations(output, saved_trace, short_ids=False):
    old = json.loads(saved_trace.read_text(encoding="utf8"))
    store = {row["evidence_id"]: row for step in old["trace"]
             for row in step.get("result", {}).get("evidence", [])}
    trace = []
    archive = CitationArchive(output / "citations.sqlite3", "role-experiment",
                              "short-replay" if short_ids else "citation-replay")
    aliases = {key: f"E{index:02d}" for index, key in enumerate(store, 1)}
    reversed_aliases = {alias: key for key, alias in aliases.items()}
    original_render = archive.render

    def render(text, evidence, served, previous):
        text = re.sub(r"\[evidence=(E[0-9]+)\]",
                      lambda match: f"[evidence={reversed_aliases.get(match[1], match[1])}]", text)
        return original_render(text, evidence, served, previous)

    async def ask(role, stage, messages, tools=()):
        if not short_ids:
            return await invoke(role, stage, messages, tools)
        visible = list(messages)
        visible[0] = visible[0].model_copy(update={"content": visible[0].content.replace(
            "[evidence=证据ID]", "[evidence=短编号]（例如 [evidence=E01]）"
        )})
        payload = json.loads(visible[1].content)
        for key, fields in payload["evidence_catalog"].items():
            fields["display_id"] = aliases[key]
        visible[1] = visible[1].model_copy(update={"content": json.dumps(payload, ensure_ascii=False)})
        for index, message in enumerate(visible):
            if message.type != "tool":
                continue
            data = json.loads(message.content)
            if data.get("evidence_id") in aliases:
                data["display_id"] = aliases[data["evidence_id"]]
                visible[index] = message.model_copy(update={"content": json.dumps(data, ensure_ascii=False)})
        return await invoke(role, stage, visible, tools)

    if short_ids:
        archive.render = render
    started = perf_counter()
    async with asyncio.timeout(95):
        answer = await deliver_reports(old["question"], old["results"], store, ask, trace,
                                       probe.REPORT_DELIVERY, citation_archive=archive,
                                       citation_scope={FIRST: None, SECOND: None})
    save(output, "citation-short-display" if short_ids else "citation-display", {
        "question": old["question"], "answer": answer.text, "trace": trace, "seconds": perf_counter()-started,
    })


async def end_to_end(root, output):
    question, scope, _ = CASES["open"]
    saved = {key: json.loads((output / f"plan-open-{key}.json").read_text(encoding="utf8"))
             for key in ("baseline", "roles")}

    async def arm(plan_name, apply_roles):
        plan = saved[plan_name]["plan"]
        roles = {task["id"]: task.get("role") for task in plan["tasks"]}
        trace, calls = [], []
        started = perf_counter()

        async def ask(role, stage, messages, tools=()):
            if stage == "planner":
                return AIMessage(content=json.dumps(plan, ensure_ascii=False))
            if apply_roles and stage in roles:
                system = messages[0].model_copy(update={"content": messages[0].content + "\n" + ROLES[roles[stage]]})
                messages = [system, *messages[1:]]
            entry = {"stage": stage, "start": perf_counter() - started, "input_chars": sum(
                len(str(message.content)) for message in messages
            )}
            calls.append(entry)
            begin = perf_counter()
            try:
                reply = await invoke(role, stage, messages, tools)
                entry.update(tool_calls=reply.tool_calls, content=reply.content, usage=reply.usage_metadata)
                return reply
            finally:
                entry["seconds"] = perf_counter() - begin

        graph = probe.build_graph(root, scope, ask, trace, roomy=True, reuse_retrieval=True,
                                  shared_pdf_reads=True, source_footer=True, worker_roles=False,
                                  worker_full_question=True)
        try:
            async with asyncio.timeout(100):
                result = await graph.ainvoke({"question": question})
        except TimeoutError:
            result = {"observation_stopped": True}
        result.update(question=question, plan=plan, trace=trace, calls=calls, seconds=perf_counter()-started,
                      separate_planning_seconds=saved[plan_name]["seconds"])
        save(output, f"e2e-{plan_name}-{'role' if apply_roles else 'generic'}", result)

    await asyncio.gather(arm("baseline", False), arm("roles", False), arm("roles", True))


def project_delegation(messages, mode):
    if mode == "A":
        return messages
    projected = list(messages)
    for index, message in enumerate(projected):
        if message.type != "human":
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or "assignment" not in payload or "user_request" not in payload:
            continue
        # 两组共用一句交付指令。C 的指派已包含适用的约束，工具权限由图独立校验。
        guidance = probe.WORKER_ASSIGNMENT_RULE
        if mode == "C":
            del payload["user_request"]
        else:
            guidance += "原请求提供背景。"
        projected[0] = projected[0].model_copy(update={"content": projected[0].content + "\n" + guidance})
        projected[index] = message.model_copy(update={"content": json.dumps(payload, ensure_ascii=False)})
        break
    return projected


async def delegation_context(root, output, baseline_record, *, live=False):
    previous = json.loads(baseline_record.read_text(encoding="utf8"))
    scope = {FIRST: {36}, SECOND: {85}}
    cross = "只读《AI Agents in Depth》PDF 第36页和《Hello-Agents》PDF 第85页。为我核对两页能否支持‘Observation 就是 tool_call_id’这一说法。每页各给一个出处，并区分资料与推断。"
    exercise = "只读《AI Agents in Depth》PDF 第36页。用初学者能听懂的话解释四种消息角色，最后留一道只需口头回答的练习，不给答案。"
    cases = [
        ("independent", previous["question"], scope, previous["tasks"]),
        ("cross", cross, scope, [assignment("verify", "只读两份指定教材的 PDF 第36、85页，分别注明书名与出处。核对 Observation 是否就是 tool_call_id，区分两页写明的内容与比较所得推断。", [FIRST, SECOND], role="analyst")]),
        ("exercise", exercise, {FIRST: {36}}, [assignment("messages", "只读《AI Agents in Depth》PDF 第36页，向初学者解释四种消息角色，给出可核对出处。练习由最终答复统一交付。", [FIRST], role="reader")]),
    ]

    async def arm(name, question, allowed, tasks, mode, memory=None):
        trace, calls, first_requests = [], [], {}
        started = perf_counter()

        async def ask(role, stage, messages, tools=()):
            if stage == "planner":
                return AIMessage(content=json.dumps({"tasks": tasks}, ensure_ascii=False))
            if stage != "deliver":
                messages = project_delegation(messages, mode)
            first_requests.setdefault(stage, [message.model_dump(mode="json") for message in messages])
            begin = perf_counter()
            reply = await invoke(role, stage, messages, tools)
            calls.append({"stage": stage, "seconds": perf_counter() - begin, "usage": reply.usage_metadata})
            return reply

        graph = probe.build_graph(root, allowed, ask, trace, roomy=True, reuse_retrieval=True,
                                  shared_pdf_reads=True, source_footer=True, model_request_timeout=60,
                                  conversation_memory=memory, worker_full_question=True)
        try:
            async with asyncio.timeout(110):
                row = await graph.ainvoke({"question": question})
        except TimeoutError:
            row = {"error": "observation_timeout"}
        row.update(question=question, plan=tasks, first_requests=first_requests, calls=calls,
                   trace=trace, seconds=perf_counter()-started)
        save(output, f"{name}-{mode}", row)

    if live:
        correction = ConversationMemory("", strategy="full")
        correction.add_user("准备比较两本书。我先记作《AI Agents in Depth》PDF 第36页和《Hello-Agents》PDF 第84页。只学习材料，不需要练习。")
        correction.add_ai("收到。第一本是第36页，第二本暂记第84页。")
        correction.add_user("更正：第二本是 PDF 第85页。只按更正后的那一页看 Observation，别替换成相邻页。")
        correction.add_ai("已记下第二本改为 PDF 第85页。")
        cases.append(("correction", "按我刚才更正的页码，只核对第二本。用两三句话说明 Observation 是什么、如何帮助下一步推理，标明 PDF 页码。", {FIRST: {36}, SECOND: {84, 85}}, None))

        async def plan_and_compare(case):
            name, question, allowed, _ = case
            memory = correction if name == "correction" else None
            # 复用图的规划提示和人类消息结构。每案只规划一次，随后原样回放给两组。
            messages = [SystemMessage(content=f"{probe.FOCUSED_PLANNER}\n{probe.PLANNER_ROLES}"),
                        *(memory.get_messages_for_llm()[1:] if memory else []),
                        HumanMessage(content=json.dumps({"question": question, "authorized_scope": {
                            path: sorted(pages) for path, pages in allowed.items()
                        }}, ensure_ascii=False))]
            started = perf_counter()
            reply = await invoke("standard", "planner", messages)
            plan = probe.parse(reply.content, probe.RoomyPlan)
            probe.validate(plan.tasks, allowed)
            tasks = [task.model_dump(exclude_defaults=True) for task in plan.tasks]
            save(output, f"{name}-plan", {"question": question, "plan": tasks, "raw": reply.text,
                 "first_requests": {"planner": [m.model_dump(mode="json") for m in messages]},
                 "usage": reply.usage_metadata, "seconds": perf_counter() - started})
            await asyncio.gather(*(arm(name, question, allowed, tasks, mode, memory) for mode in ("B", "C")))

        await asyncio.gather(*(plan_and_compare(case) for case in cases))
        return

    # 各案独立初始化。同案三个对照并发，避免三个案例同时争抢服务端并发。
    for case in cases:
        await asyncio.gather(*(arm(*case, mode) for mode in ("A", "B", "C")))


EXERCISE_RULES = {
    "v1": "用户要求练习不给答案时，用未标注的新情境让学习者自行判断；题干保留解题所需信息，不直接给出或排除待判断的答案。",
    "v2": "用户要求练习不给答案时，先确定要考什么，再用新情境提供可推断的线索，把待判断的属性留空，也不要替学习者排除选项。",
    "v3": probe.UNANSWERED_EXERCISE_RULE,
}


async def replay_exercise_delivery(output, saved_directory, repetitions, *, rule=None):
    saved = {group: json.loads((saved_directory / f"exercise-{group}.json").read_text(encoding="utf8"))
             for group in ("B", "C")}
    question = saved["B"]["question"]
    instructions = saved["B"]["first_requests"]["deliver"][0]["content"]
    if saved["C"]["question"] != question or saved["C"]["first_requests"]["deliver"][0]["content"] != instructions:
        raise ValueError("对照组的用户问题或汇总提示不一致")
    evidence = next(item["result"] for item in saved["B"]["trace"] if item["tool"] == "read_delivery_evidence")
    other = next(item["result"] for item in saved["C"]["trace"] if item["tool"] == "read_delivery_evidence")
    if evidence != other:
        raise ValueError("汇总回读证据不一致")
    store = {evidence["evidence_id"]: evidence}
    if rule:
        instructions += "\n" + EXERCISE_RULES[rule]
    questions = {"roles": question}
    if rule:
        questions["pairing"] = "只依据《AI Agents in Depth》PDF 第36页讲清工具结果为什么需要和调用配对。最后出一道新情境的口头练习，让我判断两条结果各该配给哪次调用。不提供答案，也不要暗示配对关系。"
        questions["origin"] = "只依据《AI Agents in Depth》PDF 第36页讲清谁产生 system、user、assistant 和 tool 消息。最后给我一道不带答案的口头练习，让我判断两个新例子的消息来源。"

    async def replay(group, topic, number):
        question = questions[topic]
        trace, requests = [], []
        started = perf_counter()

        async def ask(role, stage, messages, tools=()):
            requests.append([message.model_dump(mode="json") for message in messages])
            return await invoke(role, stage, messages, tools)

        row = {"question": question, "group": group, "source_run": f"exercise-{group}.json",
               "trace": trace, "requests": requests}
        try:
            # 每次调用新建汇总上下文。仅子报告不同，允许按旧路径主动回读同一页。
            async with asyncio.timeout(100):
                answer = await deliver_reports(question, saved[group]["results"], store, ask, trace, instructions)
            row["answer"] = answer.text
        except (TimeoutError, ValueError) as error:
            row["error"] = type(error).__name__
        row["seconds"] = perf_counter() - started
        suffix = f"-{topic}" if rule else ""
        save(output, f"exercise-delivery-{group}{suffix}-{number}", row)

    await asyncio.gather(*(replay(group, topic, number) for topic in questions
                           for number in range(1, repetitions + 1) for group in saved))


if __name__ == "__main__":
    mode, materials, target, *extra = sys.argv[1:]
    output = Path(target)
    output.mkdir(parents=True, exist_ok=True)
    if mode == "planning":
        job = planning(output)
    elif mode == "worker":
        job = worker(Path(materials).resolve(), output)
    elif mode == "worker-opening":
        job = worker(Path(materials).resolve(), output, opening=True)
    elif mode == "e2e":
        job = end_to_end(Path(materials).resolve(), output)
    elif mode in {"delegation-context", "delegation-live"}:
        job = delegation_context(Path(materials).resolve(), output, Path(extra[0]), live=mode == "delegation-live")
    elif mode in {"exercise-delivery", "exercise-delivery-corrected"}:
        job = replay_exercise_delivery(output, Path(materials), int(extra[0]),
                                       rule=(extra[1] if len(extra) > 1 else "v3")
                                       if mode == "exercise-delivery-corrected" else None)
    else:
        job = citations(output, Path(extra[0]), short_ids=mode == "citations-short")
    asyncio.run(job)
