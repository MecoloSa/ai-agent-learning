"""Isolate the cost of extra deliverables in open textbook retrieval."""

import argparse
import asyncio
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage

from ai_agent.retrieval.rag import LearningRAG
from tests.helpers import deep_learning_scenarios as scenarios
from tests.helpers import delegated_reading as graph
from tests.helpers.deep_role_experiment import COMPLEX, invoke, save


PLANNER_HEAD = "需要派发时，在 instruction 写清子任务的交付目标和适用约束，沿用用户指定的交付范围。额外取证由执行者按需决定。"
CONCISE_PLANNER_HEAD = "需要派发时，在 instruction 写清子任务应交付的结果、完成该结果所需的素材和适用约束。交付范围以用户请求为准。"
CURRENT_PLANNER_HEAD = "需要派发时，在 instruction 写清子任务的交付目标、所需素材和适用的用户约束。"
DELIVERABLE_PLANNER_HEAD = "需要派发时，在 instruction 写清子任务应交付的结果、所需素材和适用的用户约束。用户要求的结果或下游确实需要的成果，才列为必交内容。"
ACCEPTANCE_PLANNER_HEAD = "需要派发时，在 instruction 写清执行者要解决的问题，以及交付成立所需的条件。只传递会影响交付验收的用户要求或下游依赖；查找方法和中间材料由执行者按需决定。"
QUESTION_PLANNER_HEAD = "需要派发时，把 instruction 写成执行者可独立回答的问题，带上适用的资料范围和用户要求。"
READING_CAPABILITIES = "执行者可以在授权资料中搜索或读取指定位置，并提交带证据 ID 的发现。汇总可按证据 ID 回读对应正文。"

DUAL_CONCISE = {
    "t1_locate_pages": "只读《AI Agents in Depth》PDF 第41、44页，交付名为原文事实清单的产物，包含第41页实际调用编号与函数对应关系、第44页回填代码的关键字段，以及各自页码。",
    "t2_analyze_mismatch": "依据上游原文事实清单，解释把天气结果填到时间调用编号下会造成什么混淆。标明第41、44页依据，区分教材展示的行为与推断，不断言未验证的运行结果。",
}


def without_backup_sources(tasks):
    # 对已录制的计划做单变量实验。程序运行时不按措辞修改规划结果。
    replacements = {
        "T1": ("，并另附 1 处备选出处备查", ""),
        "T2": ("，另附 1 处备选出处", ""),
    }
    result = deepcopy(tasks)
    for task in result:
        old, new = replacements[task["id"]]
        if task["instruction"].count(old) != 1:
            raise ValueError("固定计划应包含一处备选出处要求")
        task["instruction"] = task["instruction"].replace(old, new)
    return result


async def compare(materials, saved, output, repeats, compare_dispatch=False):
    recorded = json.loads(saved.read_text(encoding="utf8"))
    question = recorded["question"]
    scope = {source: set(pages) if pages is not None else None for source, pages in recorded["scope"].items()}
    if compare_dispatch:
        LearningRAG(str(materials), data_dir=graph.get_settings().rag_data_dir)
    for repeat in range(1, repeats + 1):
        names = f"baseline-{repeat}", f"one-source-{repeat}"
        scenarios.CASES[names[0]] = (question, scope, recorded["tasks"])
        scenarios.CASES[names[1]] = (question, scope, without_backup_sources(recorded["tasks"]))
        if compare_dispatch:
            concise = {task["id"]: task["instruction"] for task in scenarios.CASES["open-lean"][2]}
            plan = [{**task, "instruction": concise[task["id"]]} for task in recorded["tasks"]]
            names = (*names, f"concise-{repeat}")
            scenarios.CASES[names[-1]] = (question, scope, plan)
        await scenarios.main(materials, output, names, 145)


async def compare_concise(materials, saved, dual_saved, output, repeats):
    sources = {name: json.loads(path.read_text(encoding="utf8"))
               for name, path in (("open", saved), ("dual", dual_saved))}
    replacements = {
        "open": {task["id"]: task["instruction"] for task in scenarios.CASES["open-lean"][2]},
        "dual": DUAL_CONCISE,
    }
    # 同一语料的对照组共用 Chroma 后端。先在单线程打开索引，避免并发首次初始化，
    # 本步骤不查询、不改变各组收到的检索素材，也不计入单组耗时。
    LearningRAG(str(materials))
    for repeat in range(1, repeats + 1):
        names = []
        for case, row in sources.items():
            scope = {source: set(pages) if pages is not None else None for source, pages in row["scope"].items()}
            tasks = row["tasks"]
            if {task["id"] for task in tasks} != replacements[case].keys():
                raise ValueError(f"{case} 固定任务与指派不一致")
            concise = [{**task, "instruction": replacements[case][task["id"]]} for task in tasks]
            for arm, plan in (("full", tasks), ("concise", concise)):
                name = f"{case}-{arm}-{repeat}"
                names.append(name)
                scenarios.CASES[name] = (row["question"], scope, plan)
        # 同轮并发比较，先完成一轮再启动下一轮，避免两轮间争抢同一后端。
        await scenarios.main(materials, output, names, 145)


async def compare_planner(output, *, concise=False, planner_head=None, capabilities=False, repeats=1):
    question, scope, _ = scenarios.CASES["open"]
    cases = {
        "open": question,
        "backup-requested": "请在《AI Agents in Depth》中找两处可核对的正文解释工具调用结果怎么配对，一处作为主要讲解依据，另一处给我备用。分别说明各自讲了什么。",
        "open-comparison": "请结合《AI Agents in Depth》和《Hello-Agents》，比较它们如何解释工具执行结果进入模型上下文。用足以支持比较的资料，不限定每本找几处。",
    }
    inputs = {name: (question, scope) for name, question in cases.items()}
    if planner_head is not None or capabilities:
        inputs["verbatim-requested"] = (
            "只读《AI Agents in Depth》PDF 第41页。请逐字引用教材说明工具结果如何关联调用的那一句，标明PDF页码，再用一句通俗的话解释。",
            {scenarios.FIRST: {41}},
        )
        inputs["two-artifact-dependency"] = COMPLEX
    if capabilities:
        inputs["direct-followup"] = (
            "已知 Action 是执行动作，Observation 是动作执行后的返回结果。用我这两句话给出一个日常类比，不需要读教材。",
            scope,
        )
    output.mkdir(parents=True, exist_ok=True)

    async def arm(name, question, scope, variant, repeat):
        started = perf_counter()
        base = graph.FOCUSED_PLANNER.splitlines()
        candidate = variant == "candidate"
        if capabilities:
            if variant in {"question", "combined"}:
                base[0] = QUESTION_PLANNER_HEAD
        elif planner_head is not None:
            base[0] = planner_head if candidate else CURRENT_PLANNER_HEAD
        elif candidate:
            base[0] = CONCISE_PLANNER_HEAD if concise else PLANNER_HEAD
        # 只比较本实验的固定阅读能力。实际图有条件开启的历史和结构工具未纳入。
        extra = [READING_CAPABILITIES] if capabilities and variant in {"capability", "combined"} else []
        system = "\n".join([*base, graph.PLAN_ROUTE_RULE, *extra, graph.PLAN_GENERATION_RULE, graph.PLANNER_ROLES])
        payload = {"question": question, "authorized_scope": {
            source: sorted(pages) if pages is not None else None for source, pages in scope.items()}}
        row = {"question": question, "system": system, "payload": payload}
        try:
            reply = await invoke("standard", "planner", [SystemMessage(content=system),
                                 HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
            row.update(raw=reply.content, usage=reply.usage_metadata)
            plan = graph.parse(reply.content, graph.RoomyPlan)
            graph.validate(plan.tasks, scope)
            row["plan"] = plan.model_dump(exclude_defaults=True)
        except (ValueError, TimeoutError) as error:
            row["error"] = type(error).__name__
        suffix = f"-{repeat}" if repeats > 1 else ""
        save(output, f"planner-{name}-{variant}{suffix}", {
            **row, "seconds": perf_counter() - started,
        })

    variants = ("baseline", "capability", "question", "combined") if capabilities else ("baseline", "candidate")
    for repeat in range(1, repeats + 1):
        # 按案例配对。两批并发覆盖全部案例，减少瞬时请求竞争。
        items = list(inputs.items())
        batch_size = 3 if capabilities else len(items)
        for start in range(0, len(items), batch_size):
            await asyncio.gather(*(arm(name, question, scope, variant, repeat)
                                   for name, (question, scope) in items[start:start + batch_size]
                                   for variant in variants))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("recorded", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--planner-only", action="store_true")
    parser.add_argument("--concise-planner", action="store_true")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--deliverable-planner", action="store_true")
    prompt_group.add_argument("--acceptance-planner", action="store_true")
    prompt_group.add_argument("--capability-planner", action="store_true")
    parser.add_argument("--dual-recorded", type=Path)
    parser.add_argument("--compare-dispatch", action="store_true")
    args = parser.parse_args()
    if args.capability_planner:
        asyncio.run(compare_planner(args.output, capabilities=True, repeats=args.repeats))
    elif args.deliverable_planner or args.acceptance_planner:
        head = ACCEPTANCE_PLANNER_HEAD if args.acceptance_planner else DELIVERABLE_PLANNER_HEAD
        asyncio.run(compare_planner(args.output, planner_head=head, repeats=args.repeats))
    elif args.planner_only:
        asyncio.run(compare_planner(args.output, concise=args.concise_planner))
    elif args.dual_recorded:
        asyncio.run(compare_concise(args.materials.resolve(), args.recorded, args.dual_recorded,
                                   args.output, args.repeats))
    else:
        asyncio.run(compare(args.materials.resolve(), args.recorded, args.output, args.repeats,
                            args.compare_dispatch))
