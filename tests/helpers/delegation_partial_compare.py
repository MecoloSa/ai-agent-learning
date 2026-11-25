"""Replay a real deadline handoff through the main interaction loop."""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool

import ai_agent.main as app
from ai_agent.retrieval.rag import LearningRAG


def features(text):
    words = re.findall(r"[a-z_][a-z_0-9]+|[\u4e00-\u9fff]+", text.lower())
    return {word if word.isascii() else word[i:i + 2]
            for word in words for i in (range(len(word) - 1) if not word.isascii() else range(1))}


def rank_with_diversity(candidates, query, store):
    pages = [key for key in candidates if store[key].get("page") is not None
             and "chunk_index" not in store[key]]
    candidates = pages or candidates
    tokens = {key: features(store[key]["text"]) for key in candidates}
    counts = Counter(token for group in tokens.values() for token in group)

    def overlap(group):
        return sum(math.log1p(len(candidates) / (1 + counts[token])) for token in query & group)

    # 单页整体相关性突出短而聚焦的页面，单行匹配补充局部包含关键解释的长页。
    dense = sorted(candidates, key=lambda key: overlap(tokens[key]) / max(20, len(tokens[key])) ** .45,
                   reverse=True)
    focused = sorted(candidates, key=lambda key: max(
        (overlap(features(line)) for line in store[key]["text"].splitlines()), default=0), reverse=True)
    return list(dict.fromkeys([key for pair in zip(dense, focused) for key in pair]))


def seed_partial_evidence(report, results, tasks, store, *, balanced=False):
    """Preserve a bounded, source-balanced sample of actually returned text."""
    excerpts = {}
    seen_pages = set()
    for task in tasks:
        result = results.get(task["id"], {})
        if result.get("status") != "unavailable":
            continue
        query = features(task["instruction"])
        candidates = [key for key in result.get("evidence_ids", [])
                      if key in report["evidence_catalog"] and store.get(key, {}).get("text")]
        if balanced:
            candidates = rank_with_diversity(candidates, query, store)
        else:
            candidates.sort(key=lambda key: (
                len(query & features(store[key]["text"])), "chunk_index" not in store[key],
            ), reverse=True)
        branch_count = 0
        for key in candidates:
            if len(excerpts) >= 4:
                return excerpts
            row = store[key]
            page = (row.get("source"), row.get("page")) if row.get("page") is not None else key
            if page in seen_pages:
                continue
            text = row["text"]
            excerpts[key] = {"text": text[:2200], "truncated": len(text) > 2200 or row.get("partial") is True}
            seen_pages.add(page)
            branch_count += 1
            if branch_count >= (2 if balanced else 1):
                break
    return excerpts


def run(root, output, record, arm):
    question = record["question"]
    report = record["delegated_report"].copy()
    store = {row["evidence_id"]: row for event in record["trace"]
             for row in event.get("result", {}).get("evidence", []) if row.get("text")}
    if arm in {"excerpts", "balanced"}:
        report["partial_evidence"] = seed_partial_evidence(
            report, record["results"], record["tasks"], store, balanced=arm == "balanced",
        )

    @tool
    def delegate_learning_task(objective: str) -> str:
        """规划并执行委派，返回发现、证据位置和缺口。objective 填写本次委派的整体目标。"""
        return json.dumps({"ok": True, "data": report}, ensure_ascii=False)

    read = app.build_read_material_lines_tool(root)
    retrieval = app.build_retrieve_learning_context_tool(root, structured_result=True)
    others = [app.build_list_material_files_tool(root), app.build_read_pdf_pages_tool(root),
              delegate_learning_task]
    original = app.invoke_llm_response
    calls = 0

    def invoke(model, messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(content="", tool_calls=[{
                "id": "frozen_handoff", "name": "delegate_learning_task", "args": {"objective": question},
            }])
        return original(model, messages, **kwargs)

    audit = {}
    start = perf_counter()
    with patch.object(app, "invoke_llm_response", invoke):
        app._invoke_interactive_turn(
            request_messages=[SystemMessage(content="你是学习助手，依据用户问题和可用资料回答。")],
            user_input=question, plain_llm=app.create_llm(),
            tool_llm=app.create_llm_with_tools([read, retrieval, *others]),
            retrieval_tool=retrieval, read_tool=read, extra_tools=others, audit_trace=audit,
        )
    audit["seconds"] = round(perf_counter() - start, 2)
    audit["seed_locations"] = {key: report["evidence_catalog"][key]
                               for key in report.get("partial_evidence", {})}
    (output / f"{arm}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf8")
    print(arm, audit["seconds"], calls,
          dict(Counter(row["tool_name"] for row in audit["tool_results"] if row["executed"])), flush=True)


if __name__ == "__main__":
    root, record_path, output = sys.argv[1:4]
    root = str(Path(root).resolve())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = json.loads(Path(record_path).read_text(encoding="utf8"))
    LearningRAG(root)
    for arm in sys.argv[4:] or ("locators", "excerpts"):
        run(root, output, record, arm)
