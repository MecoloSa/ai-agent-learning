"""Replay archived interactive requests with a conservative receipt projection."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ai_agent.config import create_llm


def project_receipts(history):
    by_id, later_pages, later_success = {}, {}, {}
    for message in history:
        if isinstance(message, AIMessage):
            by_id.update({call["id"]: call for call in message.tool_calls})
    projected = list(history)
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if not isinstance(message, ToolMessage):
            continue
        call = by_id.get(message.tool_call_id)
        if not call or call["name"] != "read_pdf_pages":
            continue
        try:
            receipt = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if not isinstance(receipt, dict):
            continue
        signature = json.dumps(call["args"], sort_keys=True)
        if receipt.get("ok") is False and signature in later_success:
            replacement = {"ok": False, "superseded_by": later_success[signature]}
        elif receipt.get("ok") is True and isinstance(receipt.get("data"), dict):
            later_success[signature] = message.tool_call_id
            changed = False
            for page in receipt["data"].get("pages", ()):
                text = page.get("text")
                if not text:
                    continue
                identity = (page.get("citation"), text, page.get("partial"), page.get("text_status"))
                if page.get("citation") and identity in later_pages:
                    del page["text"]
                    page["text_in_receipt"] = later_pages[identity]
                    changed = True
                else:
                    later_pages[identity] = message.tool_call_id
            if not changed:
                continue
            replacement = receipt
        else:
            continue
        projected[index] = message.model_copy(update={"content": json.dumps(replacement, ensure_ascii=False)})
    return projected


def replay_prefix(record):
    messages = [
        SystemMessage(content="你是学习助手，依据用户问题和可用资料回答。"),
        HumanMessage(content=record["user_input"]),
    ]
    receipts = {row["tool_call_id"]: row["result"] for row in record["tool_results"]}
    for event in record["model_responses"][:-1]:
        calls = event["tool_calls"]
        if calls:
            messages.append(AIMessage(content=event["content"], tool_calls=calls))
            messages.extend(ToolMessage(content=receipts[call["id"]], tool_call_id=call["id"])
                            for call in calls)
    return messages


def add_retry_control(messages):
    last = next(message for message in reversed(messages) if isinstance(message, AIMessage)
                and any(call["name"] == "read_pdf_pages" for call in message.tool_calls))
    original = next(call for call in last.tool_calls if call["name"] == "read_pdf_pages")
    receipt = next(message for message in messages if isinstance(message, ToolMessage)
                   and message.tool_call_id == original["id"])
    failed = {**original, "id": "controlled_failed_read"}
    repeated = {**original, "id": "controlled_duplicate_read"}
    error = json.dumps({"ok": False, "error": {"code": "temporary_read_error", "retryable": True}})
    index = messages.index(last)
    return [*messages[:index], AIMessage(content="", tool_calls=[failed]),
            ToolMessage(content=error, tool_call_id=failed["id"]), *messages[index:],
            AIMessage(content="", tool_calls=[repeated]),
            receipt.model_copy(update={"tool_call_id": repeated["id"]})]


async def compare(record, output, retry_control=False):
    baseline = replay_prefix(record)
    if retry_control:
        baseline = add_retry_control(baseline)
    projected = project_receipts(baseline)
    output.mkdir(parents=True, exist_ok=True)
    for arm, messages in (("baseline", baseline), ("projection", projected)):
        start = perf_counter()
        reply = await create_llm().ainvoke(messages)
        report = {
            "user_input": record["user_input"],
            "request_messages": [message.model_dump(mode="json") for message in messages],
            "answer": reply.text,
            "usage": reply.usage_metadata,
            "seconds": round(perf_counter() - start, 2),
            "same_as_baseline": messages == baseline,
            "controlled_retry_and_duplicate": retry_control,
        }
        (output / f"{arm}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
        print(arm, report["seconds"], len(reply.text), reply.usage_metadata, report["same_as_baseline"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--retry-control", action="store_true")
    args = parser.parse_args()
    asyncio.run(compare(json.loads(args.record.read_text(encoding="utf8")), args.output, args.retry_control))
