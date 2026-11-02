"""汇总已完成的配对实验；只计算观察指标，不自动判断回答质量。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from ai_agent.runtime.context_handoff import format_acquisition_facts


def summarize(inputs: list[dict], rows: list[dict], repeats: int) -> dict:
    if not inputs or repeats < 1:
        raise ValueError("实验需要至少一个案例和一轮配对")
    expected = {(case["case_id"], repeat, arm) for case in inputs for repeat in range(1, repeats + 1) for arm in ("without_facts", "with_facts")}
    keys = [(r["case_id"], r["repeat"], r["arm"]) for r in rows]
    if set(keys) != expected or any(count != 1 for count in Counter(keys).values()):
        raise ValueError("配对记录缺失、重复或包含额外案例，不能直接汇总")
    indexed = {key: row for key, row in zip(keys, rows)}
    checks, metrics = [], []
    for case in inputs:
        for repeat in range(1, repeats + 1):
            left = indexed[(case["case_id"], repeat, "without_facts")]
            right = indexed[(case["case_id"], repeat, "with_facts")]
            if left["status"] != "completed" or right["status"] != "completed":
                raise ValueError("配对含失败记录，需要先单独分析失败")
            a, b = [row["request_messages"][0]["messages"] for row in (left, right)]
            checks.append({
                "case_id": case["case_id"], "repeat": repeat,
                "same_initial_evidence": left["plan_trace"]["initial_evidence"] == right["plan_trace"]["initial_evidence"] == case["evidence"],
                "same_system_message": a[0] == b[0],
                "only_facts_changed": b[1]["content"] == a[1]["content"] + format_acquisition_facts(case["acquisition_facts"]),
                "source_unchanged": left.get("source_unchanged") is True and right.get("source_unchanged") is True,
                "same_returned_tool_text": [o["content"] for o in left["plan_trace"]["tool_observations"]] == [o["content"] for o in right["plan_trace"]["tool_observations"]],
            })
        for arm in ("without_facts", "with_facts"):
            group = [r for r in rows if r["case_id"] == case["case_id"] and r["arm"] == arm]
            usage_complete = all(r["usage"]["usage_complete"] for r in group)
            metrics.append({
                "case_id": case["case_id"], "arm": arm, "n": len(group),
                "mean_latency_ms": round(mean(r["latency_ms"] for r in group), 1),
                "total_retrieval_calls": sum(len(r["retrieval_calls"]) for r in group),
                "total_queries": sum(len(call["queries"]) for r in group for call in r["retrieval_calls"]),
                "recorded_model_responses": sum(len(r["model_responses"]) for r in group),
                "usage_complete": usage_complete,
                **{f"mean_{key}": round(mean(r["usage"][key] for r in group), 1) if usage_complete else None for key in ("input_tokens", "output_tokens", "total_tokens")},
            })
    # 工具是否调用、实际输出是否相同都可能是处理效应，不能要求它们相同才承认实验有效。
    controls = ("same_initial_evidence", "same_system_message", "only_facts_changed", "source_unchanged")
    return {"records": len(rows), "paired_checks": checks, "metrics": metrics,
            "all_control_checks_passed": all(all(check[key] for key in controls) for check in checks)}


def write_report(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize(inputs, rows, manifest["repeats"])
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 获取限制配对实验：完整输入与输出", "", "这是冻结检索、真实聊天模型的筛查实验；不是端到端RAG评测。质量判断见单独评审，不由completed字段推断。", "", "## 运行条件", "", "```json", json.dumps(manifest, ensure_ascii=False, indent=2), "```", "", "## 数值结果", "", "|案例|组别|次数|补查调用合计|查询合计|平均输入token|平均输出token|平均耗时秒|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for m in summary["metrics"]:
        lines.append(f"|{m['case_id']}|{m['arm']}|{m['n']}|{m['total_retrieval_calls']}|{m['total_queries']}|{m['mean_input_tokens']}|{m['mean_output_tokens']}|{m['mean_latency_ms']/1000:.2f}|")
    for case in inputs:
        lines += ["", f"## 案例 {case['case_id']}", "", "提问原文：", "", case["goal"], "", "固定分析：", "", case["analysis"], "", "预设判定边界（不发给模型）：", "", case["criterion"], "", "### 初始实际证据", "", "```text", case["evidence"], "```", "", "### 工具实际生成的获取限制", "", "```json", json.dumps(case["acquisition_facts"], ensure_ascii=False, indent=2), "```", "", "### 固定补查材料"]
        for doc in case["supplement_materials"]:
            lines += ["", f"来源：{doc['source']}", "", doc["text"]]
        for row in [r for r in rows if r["case_id"] == case["case_id"]]:
            lines += ["", f"### 第{row['repeat']}轮 / {row['arm']}", "", f"实际补查调用：{len(row['retrieval_calls'])}；耗时：{row['latency_ms']/1000:.3f}秒。", "", "补查查询：", "", "```json", json.dumps(row["retrieval_calls"], ensure_ascii=False, indent=2), "```", "", "最终回答：", "", row["answer"], "", "完整逐轮输入与公开响应见results.jsonl的request_messages和model_responses；不记录私有思维链。"]
    (directory / "transcript.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    summary = write_report(args.directory)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
