"""Report-based synthesis for the delegated-reading experiment."""

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool


class DeliveryFailure(ValueError):
    """The synthesis model could not deliver usable text."""


def build_report_input(question, results, store):
    reports = {}
    evidence_catalog = {}
    for task_id, result in results.items():
        report = {key: value for key, value in result.items() if key != "evidence_ids"}
        # 已提交的发现决定证据范围。未完成时，保留可供部分交付的真实回执。
        evidence_ids = [key for finding in report.get("findings", []) for key in finding["evidence_ids"]]
        if report["status"] != "done":
            evidence_ids.extend(result.get("evidence_ids", []))
        reports[task_id] = report
        for key in dict.fromkeys(evidence_ids):
            if key in store:
                item = store[key]
                evidence_catalog[key] = {field: item[field] for field in (
                    "source", "page", "lines", "chunk_index", "kind", "mode", "sources", "limitations",
                )
                                         if field in item}
                if item.get("kind") == "archived_conversation":
                    evidence_catalog[key]["citation_ref"] = item["citation_ref"]
    return {"question": question, "reports": reports, "evidence_catalog": evidence_catalog}


async def deliver_reports(question, results, store, ask, trace, instructions, *, frontload_findings=False,
                          conversation_context=(), citation_archive=None, citation_scope=None,
                          delivered_evidence_ids=None):
    payload = build_report_input(question, results, store)
    catalog = payload["evidence_catalog"]
    served = set()
    if frontload_findings:
        # 优先附每路的第一条发现所引用的正文，其余正文仍按需回查。
        selected = {}
        remaining = 12000
        for report in results.values():
            for finding in report.get("findings", [])[:1]:
                for key in finding["evidence_ids"]:
                    if key in catalog and key not in selected and len(store[key]["text"]) <= remaining:
                        selected[key] = store[key]["text"]
                        remaining -= len(selected[key])
        if selected:
            payload["already_read"] = selected
            served.update(selected)
    additional_receipts = {}
    for report in results.values():
        for key in report.get("evidence_ids", []):
            if key not in catalog and key in store:
                row = store[key]
                # 这里只索引本轮真实取得的回执，索引和正文均不放入初始模型请求。
                if "page" in row and "chunk_index" not in row:
                    additional_receipts.setdefault((row["source"], row["page"]), set()).add(key)

    @tool
    def read_delivery_evidence(evidence_id: str) -> str:
        """查看证据目录中某个 ID 的原文和页码。"""
        if evidence_id not in catalog:
            return json.dumps({"error": "unknown_evidence_id"})
        if evidence_id in served:
            return json.dumps({"status": "already_returned", "evidence_id": evidence_id})
        served.add(evidence_id)
        return json.dumps(store[evidence_id], ensure_ascii=False)

    @tool
    def read_additional_receipt(source: str, page: int) -> str:
        """按文件名和物理页回查子任务已读、但没有写进报告引用的证据。"""
        keys = additional_receipts.get((source, page))
        if not keys:
            return json.dumps({"error": "not_in_read_receipts"})
        rows = [store[key] for key in sorted(keys - served)]
        served.update(keys)
        return json.dumps({"evidence": rows} if rows else {"status": "already_returned"}, ensure_ascii=False)

    previous_citations = None
    if citation_archive is not None:
        previous_citations = citation_archive.bind_read(citation_scope or {})
        instructions += "\n引用本轮已读的 PDF 整页时，用 [evidence=证据ID] 标注引用。"
    messages = [
        SystemMessage(content=instructions),
        *conversation_context,
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    tools = [read_delivery_evidence] if catalog else []
    if additional_receipts:
        tools.append(read_additional_receipt)
    if previous_citations is not None:
        tools.append(previous_citations)
    # 新参数仍可试。记住自上次取得新正文后已经试过的请求，识别 A→B→A。
    stalled_calls = set()
    answer_only = False
    retried_empty = False
    previous_served = set()
    previous_refs = {}
    while True:
        reply = await ask("standard", "deliver", messages, [] if answer_only else tools)
        if not reply.tool_calls:
            if reply.text.strip():
                if delivered_evidence_ids is not None:
                    delivered_evidence_ids.update(served)
                if citation_archive is not None:
                    return reply.model_copy(update={
                        "content": citation_archive.render(reply.text, store, served, previous_refs),
                    })
                return reply
            if retried_empty or answer_only:
                raise DeliveryFailure("汇总未生成可交付正文")
            # 空回复没有新的信息，不把它追加到下一次请求。
            retried_empty = True
            continue

        if answer_only:
            raise DeliveryFailure("汇总收尾未生成正文")

        served_before = len(served) + len(previous_served)
        messages.append(reply)
        for call in reply.tool_calls:
            evidence_id = call["args"].get("evidence_id")
            if previous_citations is not None and call["name"] == previous_citations.name:
                try:
                    receipt = previous_citations.invoke(call["args"])
                except ValueError:
                    receipt = json.dumps({"error": "invalid_citation_locator"})
                for row in json.loads(receipt).get("historical_citations", []):
                    previous_served.add((row["turn"], row["evidence_id"]))
                    previous_refs.setdefault(row["evidence_id"], row)
            elif call["name"] == read_additional_receipt.name and additional_receipts:
                source, page = call["args"].get("source"), call["args"].get("page")
                if isinstance(source, str) and type(page) is int:
                    receipt = read_additional_receipt.invoke({"source": source, "page": page})
                else:
                    receipt = json.dumps({"error": "invalid_receipt_locator"})
            elif call["name"] != read_delivery_evidence.name or not catalog:
                receipt = json.dumps({"error": "unknown_delivery_tool"})
            elif not isinstance(evidence_id, str):
                receipt = json.dumps({"error": "invalid_evidence_id"})
            else:
                receipt = read_delivery_evidence.invoke({"evidence_id": evidence_id})
            trace.append({"task": "deliver", "tool": call["name"], "args": call["args"],
                          "result": json.loads(receipt)})
            messages.append(ToolMessage(content=receipt, tool_call_id=call["id"]))

        if len(served) + len(previous_served) > served_before:
            stalled_calls.clear()
        else:
            # 调用 ID 每次都可能变，只比较工具名和规范化参数。
            batch = {(call["name"], json.dumps(call["args"], sort_keys=True)) for call in reply.tool_calls}
            answer_only = batch <= stalled_calls
            stalled_calls.update(batch)
