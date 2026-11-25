"""Run a controlled two-arm comparison with the same saved plan."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from langchain_core.messages import AIMessage

from tests.helpers import delegated_reading as probe


async def compare(root, record, output, observation_seconds, thresholds=(4,), compare_views=False,
                  compare_production=False, closeout_reserve=0):
    if compare_production:
        plan = json.dumps({"tasks": record["tasks"]}, ensure_ascii=False)
        question = record["question"]
        scope = {source: set(pages) if pages is not None else None
                 for source, pages in record["scope"].items()}
    else:
        plan = next(call["response"]["content"] for call in record["calls"] if call["stage"] == "planner")
        question, scope = probe.SCENARIOS["open"]
        assert question == record["question"]
    output.mkdir(parents=True, exist_ok=True)

    async def arm(name, threshold=None, latest_only=True, reserve=0):
        trace, calls = [], []
        started = perf_counter()

        async def ask(role, stage, messages, tools=()):
            if stage == "planner":
                return AIMessage(content=plan)
            model = probe.create_llm(role=role)
            if stage == "deliver":
                model = model.model_copy(update={"model_name": probe.DEEP_READING_MODEL})
            if tools:
                model = model.bind_tools(tools)
            begin = perf_counter()
            entry = {"stage": stage, "start_seconds": begin - started}
            calls.append(entry)
            try:
                response = await model.ainvoke(messages)
                entry["usage"] = response.usage_metadata
                entry["response"] = {"content": response.content, "tool_calls": response.tool_calls}
                return response
            finally:
                entry["seconds"] = perf_counter() - begin

        trace_options = {"collect_only": True, "shared_pdf_reads": True,
                         "turn_timeout": observation_seconds, "delivery_reserve": 3} if compare_production else {}
        compiled = probe.build_graph(
            root, scope, ask, trace, roomy=True, reuse_retrieval=True, project=True,
            source_footer=True, model_request_timeout=60, latest_only=latest_only,
            acquisition_notice=threshold is not None, notice_after=threshold or 4,
            branch_report_reserve=reserve,
            **trace_options,
        )
        try:
            async with asyncio.timeout(observation_seconds):
                result = await compiled.ainvoke({"question": question, "delegation_goal": question}
                                                if compare_production else {"question": question})
        except TimeoutError:
            result = {"observation_stopped": True}
        result.update(question=question, plan=plan, notice_after=threshold,
                      trace=trace, calls=calls, seconds=perf_counter() - started)
        (output / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
        status = result.get("delegated_report", {}).get("status", result.get("delivery_status", "observation_stopped"))
        statuses = {task_id: report["status"] for task_id, report in result.get("results", {}).items()}
        print(name, round(result["seconds"], 2), status, statuses, flush=True)

    # 同时开始，两路使用同一模型与相同任务，不比较个位数秒级的差异。
    if compare_production:
        from ai_agent.retrieval.rag import LearningRAG

        # 共用索引的各组先完成首次客户端初始化，再测模型与取证策略。
        LearningRAG(str(root), data_dir=probe.get_settings().rag_data_dir)
        if closeout_reserve:
            await asyncio.gather(arm("baseline", latest_only=False),
                                 arm("closeout", latest_only=False, reserve=closeout_reserve))
        else:
            await asyncio.gather(arm("production", latest_only=False), arm("latest"),
                                 arm("notice", threshold=thresholds[0], latest_only=False))
    elif compare_views:
        await asyncio.gather(arm("latest"), arm("continuous", latest_only=False))
    else:
        await asyncio.gather(arm("control"), *(arm(f"notice-{n}", n) for n in thresholds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("plan_trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--observe", type=float, default=125)
    parser.add_argument("--notice-after", type=int, nargs="+", default=[4])
    parser.add_argument("--compare-views", action="store_true")
    parser.add_argument("--compare-production", action="store_true")
    parser.add_argument("--closeout-reserve", type=float, default=0)
    args = parser.parse_args()
    if any(n < 1 for n in args.notice_after) or len(set(args.notice_after)) != len(args.notice_after):
        parser.error("--notice-after values must be positive and distinct")
    record = json.loads(args.plan_trace.read_text(encoding="utf8"))
    asyncio.run(compare(args.materials.resolve(), record, args.output, args.observe, args.notice_after,
                        args.compare_views, args.compare_production, args.closeout_reserve))


if __name__ == "__main__":
    main()
