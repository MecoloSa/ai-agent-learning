"""Compare direct and report-based delivery using saved, confirmed tool receipts."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage
from openai import APIError

from ai_agent.config import create_llm
from tests.helpers.delegated_reading import (
    DEEP_READING_MODEL, REPORT_DELIVERY, Report, parse, rank_fallback_evidence,
)
from tests.helpers.report_delivery import deliver_reports


STAGE_PROMPT = """根据原问题和本路已取得的正文，写出可交给汇总者的阶段报告。
findings 每项包含 statement 和实际使用的 evidence_ids。证据只支持部分问题时，记录已能确认的发现，并在 gaps 写出影响回答的缺口。
返回 JSON 对象，含 findings 数组和 gaps 字符串数组。"""


def select_receipts(record, pages_per_branch):
    selected = {}
    for branch, result in record["results"].items():
        confirmed = set(result.get("evidence_ids", []))
        receipts = {
            item["evidence_id"]: item
            for row in record["trace"] if row["task"] == branch
            for item in row.get("result", {}).get("evidence", [])
            if item["evidence_id"] in confirmed and item.get("page") is not None
        }
        pages = {}
        for item in rank_fallback_evidence(record["question"], list(receipts.values())):
            key = (item["source"], item["page"])
            previous = pages.get(key)
            if previous is None or ("chunk_index" in previous and "chunk_index" not in item):
                pages[key] = item
        # 每个实验组共用同一批回执，每个物理页最多留一份正文。
        selected[branch] = list(pages.values())[:pages_per_branch]
    return selected


def validate_report(text, receipts):
    report = parse(text, Report)
    allowed = {item["evidence_id"] for item in receipts}
    if any(not set(finding.evidence_ids) <= allowed for finding in report.findings):
        raise ValueError("阶段报告引用了本路未取得的证据")
    return report


async def replay(record, pages_per_branch):
    selected = select_receipts(record, pages_per_branch)
    question = record["question"]
    model_calls = {"direct": [], "staged": []}
    trace = []

    async def invoke(arm, role, stage, messages, tools=()):
        model = create_llm(role=role)
        if role == "standard":
            model = model.model_copy(update={"model_name": DEEP_READING_MODEL})
        name = model.model_name
        if tools:
            model = model.bind_tools(tools)
        start = perf_counter()
        answer = await model.ainvoke(messages)
        model_calls[arm].append({
            "stage": stage, "model": name, "seconds": round(perf_counter() - start, 2),
            "usage": answer.usage_metadata, "message_chars": sum(len(str(m.content)) for m in messages),
        })
        return answer

    async def direct():
        start = perf_counter()
        reply = await invoke("direct", "standard", "deliver", [
            SystemMessage(content=REPORT_DELIVERY),
            HumanMessage(content=json.dumps({"question": question, "raw_receipts": selected}, ensure_ascii=False)),
        ])
        return {"answer": reply.text, "seconds": round(perf_counter() - start, 2)}

    async def staged():
        start = perf_counter()

        async def draft(branch, receipts):
            evidence_ids = [item["evidence_id"] for item in receipts]
            try:
                reply = await invoke("staged", "auxiliary", branch, [
                    SystemMessage(content=STAGE_PROMPT),
                    HumanMessage(content=json.dumps({"question": question, "evidence": receipts}, ensure_ascii=False)),
                ])
                report = validate_report(reply.text, receipts)
            except (APIError, ValueError) as error:
                # 某一路无法生成合格报告时，保留该路原回执供汇总回查。
                return branch, {"status": "unavailable", "reason": type(error).__name__,
                                "evidence_ids": evidence_ids, "findings": [], "gaps": []}
            return branch, {
                "status": "partial", **report.model_dump(),
                "evidence_ids": evidence_ids,
            }

        reports = dict(await asyncio.gather(*(draft(branch, items) for branch, items in selected.items())))
        stage_seconds = round(perf_counter() - start, 2)
        store = {item["evidence_id"]: item for receipts in selected.values() for item in receipts}

        async def ask(role, stage, messages, tools=()):
            return await invoke("staged", role, stage, messages, tools)

        answer = await deliver_reports(question, reports, store, ask, trace, REPORT_DELIVERY)
        return {"answer": answer.text, "reports": reports, "stage_seconds": stage_seconds,
                "seconds": round(perf_counter() - start, 2)}

    # 两组顺序运行，避免同时争抢同一汇总模型服务。
    direct_result = await direct()
    stage_result = await staged()
    return {"question": question, "selected": selected, "direct": direct_result, "staged": stage_result,
            "model_calls": model_calls, "delivery_tools": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pages-per-branch", type=int, default=8)
    args = parser.parse_args()
    if args.pages_per_branch < 1:
        parser.error("--pages-per-branch must be positive")
    record = json.loads(args.trace.read_text(encoding="utf-8"))
    result = asyncio.run(replay(record, args.pages_per_branch))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({arm: result[arm]["seconds"] for arm in ("direct", "staged")}))


if __name__ == "__main__":
    main()
