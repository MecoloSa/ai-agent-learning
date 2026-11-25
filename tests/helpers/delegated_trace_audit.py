"""Measure duplicate acquisition, evidence growth, and delivery using saved deep-reading traces."""

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def evidence_anchor(item):
    text = unicodedata.normalize("NFKC", item.get("text", "")).casefold()
    source = item.get("source", "")
    if "AI-Agents-in-Depth" in source and "tool_call_id" in text and ("关联" in text or "对应" in text):
        return "tool_call_id"
    if "Hello-Agents" in source and "observation" in text and "action" in text and "返回" in text:
        return "observation"
    return None


def audit_branch(rows, calls):
    acquisitions = {"search_material", "read_pdf", "locate_pdf"}
    request_signatures, evidence_ids, locations = set(), set(), set()
    outcomes, rollup, first_anchor = Counter(), Counter(), {}
    no_gain = max_no_gain = 0
    stall_checkpoint = None
    for row in rows:
        name = row["tool"]
        outcomes[name] += 1
        if name not in acquisitions | {"recall_evidence"}:
            continue
        receipt = row.get("result", {})
        args = row.get("args", {})
        signature = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        rollup["repeated_args"] += signature in request_signatures
        request_signatures.add(signature)
        items = receipt.get("evidence", [])
        new_ids = {item["evidence_id"] for item in items} - evidence_ids
        new_locations = {(item["source"], item.get("page")) for item in items} - locations
        if name == "locate_pdf" and "relative_path" in args:
            new_locations.update({(args["relative_path"], item["page"])
                                  for item in receipt.get("candidates", [])} - locations)
        if name in acquisitions:
            rollup["acquisitions"] += 1
            if new_ids or new_locations:
                no_gain = 0
            else:
                rollup["acquisitions_without_new_receipt"] += 1
                no_gain += 1
                max_no_gain = max(max_no_gain, no_gain)
                if no_gain == 4 and stall_checkpoint is None:
                    stall_checkpoint = {"acquisition": rollup["acquisitions"], "anchors": list(first_anchor)}
            for item in items:
                if (anchor := evidence_anchor(item)) is not None:
                    first_anchor.setdefault(anchor, {"acquisition": rollup["acquisitions"], "page": item.get("page")})
            evidence_ids.update(new_ids)
            locations.update(new_locations)
    model_responses = [call["response"] for call in calls if call.get("response") is not None]
    return {
        "tools": dict(outcomes),
        "acquisitions": rollup["acquisitions"],
        "repeated_args_including_recall": rollup["repeated_args"],
        "acquisitions_without_new_receipt": rollup["acquisitions_without_new_receipt"],
        "longest_acquisition_stall": max_no_gain,
        "four_stall_checkpoint": stall_checkpoint,
        "first_anchors": first_anchor,
        "acquisitions_after_first_anchor": {
            key: rollup["acquisitions"] - item["acquisition"] for key, item in first_anchor.items()
        },
        "submit_requests": sum(call["name"] == "submit_report"
                               for reply in model_responses for call in reply.get("tool_calls", [])),
        "model_seconds_sum": round(sum(call.get("seconds", 0) for call in calls), 2),
        "max_model_seconds": round(max((call.get("seconds", 0) for call in calls), default=0), 2),
    }


def audit_record(record):
    trace, calls = record["trace"], record["calls"]
    delivery = [call for call in calls if call["stage"] == "deliver"]
    first_catalog = {}
    if delivery:
        for message in delivery[0].get("messages", []):
            try:
                payload = json.loads(message["content"])
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and "evidence_catalog" in payload:
                first_catalog = payload["evidence_catalog"]
                break
    elapsed = record.get("worker_seconds", {})
    starts = {}
    for event in record.get("events", []):
        task = event["id"]
        if event["status"] == "started":
            starts[task] = event["seconds"]
        elif task in starts and task not in elapsed:
            elapsed[task] = event["seconds"] - starts[task]
    return {
        "seconds": round(record["seconds"], 2),
        "delivery_status": record.get("delivery_status"),
        "planner_seconds": round(sum(call.get("seconds", 0) for call in calls if call["stage"] == "planner"), 2),
        "delivery_catalog_size": len(first_catalog),
        "delivery_reads": sum(row["task"] == "deliver" for row in trace),
        "delivery_model_seconds": [round(call.get("seconds", 0), 2) for call in delivery],
        "branches": {
            task: {"status": result["status"],
                   "elapsed_seconds": round(elapsed[task], 2) if task in elapsed else None,
                   **audit_branch(
                [row for row in trace if row["task"] == task], [call for call in calls if call["stage"] == task],
            )} for task, result in record["results"].items()
        },
    }


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        record = json.loads(Path(filename).read_text(encoding="utf8"))
        print(json.dumps({"trace_path": filename, **audit_record(record)}, ensure_ascii=False))
