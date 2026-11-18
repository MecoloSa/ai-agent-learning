"""Candidate projection for one inference request, leaving raw receipts intact."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_agent.runtime.request_context import prepare_model_messages


def project(history, *, show_evidence_gap=False):
    messages = prepare_model_messages(history)
    names = {}
    failed_reads = []
    has_body = False
    for message in messages:
        if isinstance(message, AIMessage):
            names.update((call["id"], call["name"]) for call in message.tool_calls)
        if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            continue
        name = names.get(message.tool_call_id)
        if name not in {"inspect_document_structure", "read_material_lines"}:
            continue
        try:
            receipt = json.loads(message.content)
        except ValueError:
            continue
        if not isinstance(receipt, dict):
            continue
        data = receipt.get("data")
        if name == "read_material_lines" and receipt.get("ok") is True:
            has_body = True
        if name == "inspect_document_structure" and receipt.get("ok") is True and isinstance(data, dict):
            if data.get("mode") == "whole_document":
                has_body = bool(data.get("text")) or has_body
                continue
            # 扫描完成只描述扫描条件，与返回目录是否穷尽是两回事。
            data.pop("inspection_complete", None)
            receipt["data"] = {"returned_scope": "主要章节定位；未返回正文，未列出的小节仍可能存在。", **data}
        elif name == "read_material_lines" and receipt.get("ok") is False:
            receipt["returned_body_text"] = False
            failed_reads.append(receipt.get("error", {}).get("code"))
        else:
            continue
        message.content = json.dumps(receipt, ensure_ascii=False)
    if show_evidence_gap and failed_reads and not has_body and not any(
        name in {"retrieve_learning_context", "read_pdf_pages"} for name in names.values()
    ):
        messages.append(HumanMessage(content=json.dumps({
            "context_type": "current_turn_material_evidence",
            "returned_body_text": False,
            "read_errors": failed_reads,
        }, ensure_ascii=False)))
    return messages
