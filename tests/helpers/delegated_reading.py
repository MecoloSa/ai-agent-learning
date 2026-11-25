"""CLI runner and fixtures for delegated reading experiments."""

import sys

from ai_agent.runtime import delegated_reading as _runtime
from ai_agent.runtime.delegated_reading import *  # noqa: F403
from tests.helpers.langgraph_reading_compare import BOOKS, CASES


def __getattr__(name):
    return getattr(_runtime, name)


OPEN_QUESTION = (
    "请分别在《AI Agents in Depth》和《Hello-Agents》中找正文证据，"
    "解释工具结果如何用 tool_call_id 关联调用，以及 Observation 在 ReAct 循环中的作用。"
    "比较两者，最后只出一道基于第一本的口头练习，别给答案。"
)
SCENARIOS = {
    "open": (OPEN_QUESTION, {book: None for book in BOOKS}),
    "single": (CASES["single"], {BOOKS[0]: {36}}),
    "missing": (CASES["missing"], {BOOKS[0]: {36}, BOOKS[1]: {9999}}),
    "parallel": (CASES["parallel"], {BOOKS[0]: {36}, BOOKS[1]: {85}}),
}


async def run(root, case, output, project=True, walkthrough=False, roomy=False, *,
              deep_model=DEEP_READING_MODEL, worker_timeout=None, turn_timeout=None, progressive_reports=False,
              reuse_retrieval=False, source_footer=False, structure_lookup=False, focused_planning=False,
              model_request_timeout=None, acquisition_notice=False, latest_only=False, shared_pdf_reads=False,
              citation_archive=None):
    trace, calls = [], []
    started = perf_counter()

    async def ask(role, stage, messages, tools=()):
        begin = perf_counter()
        model = create_llm(role=role)
        if roomy and (stage == "planner" or role == "standard"):
            # 只替换深入实验的复杂分派与交付模型，子任务仍使用辅助模型。
            model = model.model_copy(update={"model_name": deep_model})
        if tools:
            choice = {"tool_choice": "submit_report"} if (
                not walkthrough and [item.name for item in tools] == ["submit_report"]
            ) else {}
            model = model.bind_tools(tools, **choice)
        record = {"stage": stage, "message_chars": sum(len(str(m.content)) for m in messages)}
        if walkthrough:
            record.update(role=role, model=model.bound.model_name if tools else model.model_name,
                          messages=[{
                              "type": m.type,
                              "content": m.content,
                              **({"tool_calls": m.tool_calls} if getattr(m, "tool_calls", []) else {}),
                              **({"tool_call_id": m.tool_call_id} if isinstance(m, ToolMessage) else {}),
                          } for m in messages],
                          tools=[t.name for t in tools])
        calls.append(record)
        try:
            options = {"max_tokens": 3000} if walkthrough and not roomy else {}
            answer = await model.ainvoke(messages, **options)
        except asyncio.CancelledError:
            # 记录异步等待被中止的时刻，调用方负责区分超时和整轮取消。
            record.update(seconds=perf_counter()-begin, cancelled=True)
            raise
        except Exception as exc:
            record.update(seconds=perf_counter()-begin, error=type(exc).__name__)
            raise
        record.update(seconds=perf_counter()-begin, usage=answer.usage_metadata,
                      finish_reason=answer.response_metadata.get("finish_reason"))
        if walkthrough:
            record["response"] = {"content": answer.content, "tool_calls": answer.tool_calls}
        return answer

    question, scope = SCENARIOS[case]
    try:
        result = await build_graph(root, scope, ask, trace, project=project,
                                   planner_role="standard" if walkthrough else "auxiliary",
                                   bounded=walkthrough, roomy=roomy,
                                   worker_timeout=worker_timeout, turn_timeout=turn_timeout,
                                   model_request_timeout=model_request_timeout,
                                   progressive_reports=progressive_reports, reuse_retrieval=reuse_retrieval,
                                   source_footer=source_footer, structure_lookup=structure_lookup,
                                   focused_planning=focused_planning,
                                   acquisition_notice=acquisition_notice,
                                   latest_only=latest_only,
                                   shared_pdf_reads=shared_pdf_reads,
                                   citation_archive=citation_archive,
                                   ).ainvoke({"question": question})
    except Exception as error:
        result = {"error": type(error).__name__, "detail": str(error)}
    result.update(question=question, trace=trace, calls=calls, seconds=perf_counter()-started)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{case}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    status = result.get("error") or result.get("delivery_status", "ok")
    print(case, round(result["seconds"], 2), status, flush=True)


async def main(root, output, cases):
    archive_options = {
        flag: value for arg in cases
        for flag, sep, value in [arg.partition("=")]
        if sep and flag in {"--citation-db", "--conversation-id", "--turn-id"}
    }
    archive = None
    scenario_names = [case for case in cases if not case.startswith("--")]
    if archive_options:
        if len(archive_options) != 3 or not all(archive_options.values()):
            raise ValueError("引用归档需要同时提供 --citation-db=、--conversation-id=、--turn-id=")
        if "--roomy" not in cases or len(scenario_names) != 1:
            raise ValueError("引用归档一次只接受一个 --roomy 用例")
        output.mkdir(parents=True, exist_ok=True)
        archive = CitationArchive(Path(archive_options["--citation-db"]),
                                  archive_options["--conversation-id"], archive_options["--turn-id"])
    await asyncio.gather(*(run(root, case, output, project="--project" in cases or "--full" not in cases,
                               walkthrough="--walkthrough" in cases or "--roomy" in cases,
                               roomy="--roomy" in cases,
                               reuse_retrieval="--reuse-retrieval" in cases,
                               source_footer="--source-footer" in cases,
                               structure_lookup="--structure-lookup" in cases,
                               focused_planning="--focused-planning" in cases,
                               acquisition_notice="--acquisition-notice" in cases,
                               latest_only="--latest-only" in cases,
                               shared_pdf_reads="--shared-evidence" in cases,
                               citation_archive=archive,
                               model_request_timeout=DEEP_READING_MODEL_REQUEST_TIMEOUT if "--roomy" in cases else None)
                           for case in scenario_names))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3:] or list(SCENARIOS)))
