"""Run isolated learning tasks with the real graph, models and materials."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from langchain_core.messages import AIMessage

from tests.helpers import delegated_reading as probe


FIRST, SECOND = probe.BOOKS


def assignment(task_id, instruction, sources, depends_on=(), required_artifacts=None, *, role=None,
               generates_content=False):
    result = {"id": task_id, "instruction": instruction, "sources": sources, "depends_on": list(depends_on)}
    if required_artifacts:
        result["required_artifacts"] = required_artifacts
    if role is not None:
        result["role"] = role
    if generates_content:
        result["generates_content"] = True
    return result


CASES = {
    "dual-live": (
        "只读《AI Agents in Depth》PDF 第41、44页。根据第41页的调用编号对照和第44页的回填代码，说明天气结果如果填到时间调用的编号下会造成什么混淆。区分教材事实和推断。",
        {FIRST: {41, 44}}, None,
    ),
    "inline-rewrite": (
        "以下是我的笔记：『工具结果靠调用编号对应回原请求；观察是推理循环拿到的外部反馈。』只根据这两句，替我改成一句适合课堂讲的口语。不查文件。",
        {}, None,
    ),
    "inline-practice": (
        "请根据我给出的两个概念出一道口头判断题，不给答案：短期记忆保留最近的对话，长期记忆可以检索旧记录。只用我写的这两点，不查文件。",
        {}, None,
    ),
    "draft-review": (
        "只读《AI Agents in Depth》PDF 第36页。先核对教材中的消息角色，再根据核对结果拟一道不泄露答案的口头练习。由另一路检查题干，最后只给审过的一道题，不附答案。",
        {FIRST: {36}}, [
            assignment("read", "读取 PDF 第36页，交付包含四种角色定义的 named_roles 发现。", [FIRST], role="reader"),
            assignment("draft", "用上游定义拟一道简短的口头消息角色判断题，只交付一道题，不附答案。使用自编情境，把题干作为名为 quiz 的生成交付物。", [],
                       ["read"], {"read": ["named_roles"]}, role="practice_designer", generates_content=True),
            assignment("review", "审核上游 quiz。保留一道可直接口头回答的问题，题干不提前给出判断。直接交付审过的题干，不附答案或审核说明。", [],
                       ["draft"], {"draft": ["quiz"]}, role="generalist", generates_content=True),
        ],
    ),
    "exercise": (
        "只读《AI Agents in Depth》PDF 第36页。用初学者能听懂的话解释四种消息角色，最后留一道只需口头回答的练习，不给答案。",
        {FIRST: {36}}, None,
    ),
    "parallel": (
        "只读《AI Agents in Depth》PDF 第36页和《Hello-Agents》PDF 第85页。"
        "分别解释 tool_call_id 和 Observation 各解决什么问题，再用一个简短例子比较两者。区分书中事实和你举的例子。",
        {FIRST: {36}, SECOND: {85}}, None,
    ),
    "dependency": (
        "只读《AI Agents in Depth》PDF 第41页。先从示例提取两个调用ID和对应函数；"
        "再根据这份对应关系，解释把天气结果填到时间调用ID下面会造成什么混淆。不要把推断写成教材原话。",
        {FIRST: {41}},
        [
            assignment("extract", "从 PDF 第41页提取两个实际调用 ID 和对应函数，交付产物 call_mapping。", [FIRST],
                       role="reader"),
            assignment("teach", "根据上游的 call_mapping 解释把天气结果填到时间调用 ID 的影响，"
                       "说明这是推断。可直接使用已交接的证据。", [], ["extract"], {"extract": ["call_mapping"]},
                       role="analyst"),
        ],
    ),
    "shared": (
        "只读《AI Agents in Depth》PDF 第44页。分别解释工具结果怎样用 ID 回填，以及循环何时结束，"
        "再从这一页出一道不附答案的口头练习。",
        {FIRST: {44}},
        [
            assignment("pairing", "只读 PDF 第44页，解释工具结果如何用 ID 回填。", [FIRST]),
            assignment("loop", "只读 PDF 第44页，解释循环如何结束。", [FIRST]),
        ],
    ),
    "missing": (
        "只允许读《AI Agents in Depth》PDF 第36页和《Hello-Agents》PDF 第9999页。"
        "两页都能读时再比较；如果有一页不存在，不要改读别页，说明缺口并解释读到的内容。",
        {FIRST: {36}, SECOND: {9999}}, None,
    ),
    "open": (
        "我第一次学 Agent。请从《AI Agents in Depth》找一处正文解释工具结果如何跟调用配对，"
        "再从《Hello-Agents》找一处正文解释 ReAct 的 Observation。"
        "各给一个可核对的出处，用一小段讲清两者的关系，最后只出一道口头练习，不给答案。",
        {FIRST: None, SECOND: None}, None,
    ),
}

CASES["open-lean"] = (
    CASES["open"][0], CASES["open"][1],
    [
        assignment("T1", "从《AI Agents in Depth》找一处正文解释工具结果如何与调用配对，交付解释和可核对出处。", [FIRST]),
        assignment("T2", "从《Hello-Agents》找一处正文解释 ReAct 中 Observation 的作用，交付解释和可核对出处。", [SECOND]),
    ],
)

# 与固定任务图用同一个问题、同一份授权资料，由在线规划器独立派发。
CASES["draft-review-live"] = (*CASES["draft-review"][:2], None)


BRIEF_DELEGATION = "子任务指派写清交付目标、必要依据和适用约束。具体做法交给执行者；用户仅需最终成品时，审核任务交付可用终稿即可。"


async def run(root, output, name, observation_seconds, worker_full_question=False, brief_delegation=False):
    question, scope, fixed_tasks = CASES[name]
    started = perf_counter()
    trace, calls, first_requests = [], [], {}

    async def ask(role, stage, messages, tools=()):
        if brief_delegation and stage == "planner" and fixed_tasks is None:
            messages = [messages[0].model_copy(update={
                "content": f"{messages[0].content}\n{BRIEF_DELEGATION}",
            }), *messages[1:]]
        begin = perf_counter()
        item = {"stage": stage, "start_seconds": begin - started, "tools": [tool.name for tool in tools],
                "message_chars": sum(len(str(message.content)) for message in messages)}
        calls.append(item)
        first_requests.setdefault(stage, [message.model_dump(mode="json") for message in messages])
        try:
            if stage == "planner" and fixed_tasks:
                item["model"] = "fixed_plan"
                response = AIMessage(content=json.dumps({"tasks": fixed_tasks}, ensure_ascii=False))
            else:
                model = probe.create_llm(role=role)
                if stage in {"planner", "deliver"}:
                    model = model.model_copy(update={"model_name": probe.DEEP_READING_MODEL})
                item["model"] = model.model_name
                client = model.bind_tools(tools) if tools else model
                response = await client.ainvoke(messages)
            item["response"] = {"content": response.content, "tool_calls": response.tool_calls}
            item["usage"] = response.usage_metadata
            return response
        finally:
            item["seconds"] = perf_counter() - begin

    graph = probe.build_graph(
        root, scope, ask, trace, roomy=True, reuse_retrieval=True, focused_planning=True,
        planner_role="standard", source_footer=True, shared_pdf_reads=True, model_request_timeout=60,
        worker_full_question=worker_full_question,
    )
    try:
        async with asyncio.timeout(observation_seconds):
            result = await graph.ainvoke({"question": question})
    except TimeoutError:
        result = {"observation_stopped": True}
    except (ValueError, OSError) as error:
        result = {"error": type(error).__name__, "detail": str(error)}
    result.update(
        question=question, scope={path: sorted(pages) if pages is not None else None for path, pages in scope.items()},
        trace=trace, calls=calls, first_requests=first_requests, seconds=perf_counter() - started,
        task_only=not worker_full_question, brief_delegation=brief_delegation,
    )
    (output / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(name, round(result["seconds"], 2),
          result.get("delivery_status") or result.get("error") or "observation_stopped", flush=True)


async def main(root, output, cases, observe, worker_full_question=False, brief_delegation=False):
    output.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(*(run(root, output, name, observe, worker_full_question, brief_delegation) for name in cases))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--observe", type=float, default=130)
    context = parser.add_mutually_exclusive_group()
    context.add_argument("--task-only", action="store_true", help="仅发送子任务上下文（默认行为）")
    context.add_argument("--full-worker-question", action="store_true", help="向执行者传入原问题，用于旧版对照")
    parser.add_argument("--brief-delegation", action="store_true", help="实验：让现场规划交接简明指派及成品")
    args = parser.parse_args()
    cases = args.cases or [name for name in CASES if name != "open-lean"]
    if set(cases) - CASES.keys() or args.observe <= 0:
        parser.error("请输入有效的案例名和正数观察时长")
    asyncio.run(main(args.materials.resolve(), args.output, cases, args.observe,
                     args.full_worker_question, args.brief_delegation))
