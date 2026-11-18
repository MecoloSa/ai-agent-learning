"""小规模规划阶段AB/BA实验；默认预检，--llm才请求模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import ai_agent.runtime.agents as agents
from ai_agent.config import get_settings
from ai_agent.evaluation.acquisition_pair import message_record
from ai_agent.evaluation.interactive_metrics import summarize_chat_usage
from tests.helpers.scaffold_lab import replacement, fingerprint

SEP = "\n\n---\n\n"


def cases():
    normal = "[source=stops.md, lines=1-2]\n正常结束表示模型未再请求工具，程序结束循环；不证明回答正确。保护性停止由程序预算或执行保护触发；不证明学习目标已经完成。"
    blue = "[source=blue.md]\n蓝签：读完指定例句，并口头复述定义，不表示理解达标。"
    orange = "[source=orange.md]\n橙签：判断三个陌生例子，至少两个判断正确，并口头说明依据。"
    rows = ["def rule():", "    required_count = 2", "    total_count = 3", "    return required_count, total_count"]
    receipt = json.dumps({"ok": True, "data": {
        "source": "rules.py", "citation": "[source=rules.py, lines=301-304]",
        "requested_start_line": 301, "requested_end_line": 400,
        "returned_start_line": 301, "returned_end_line": 304, "truncated": False,
        "lines": [{"line": n, "text": text, "partial": False} for n, text in enumerate(rows, 301)],
    }}, ensure_ascii=False)
    return [
        {"case_id": "three_tasks", "profile": "freedom", "analysis": "资料说明了两类停止的区别。",
         "goal": "依据资料安排1天学习，区分正常结束与保护性停止。请明确拆成三个小任务：阅读定义、比较例子、口头自测。只阅读和口头说明，总计不超过30分钟。",
         "evidence": normal, "budget": 24000,
         "expected": "三项小任务，总计不超30分钟；解释两类停止，不以停止证明答案正确。"},
        {"case_id": "duplicate_budget", "profile": "evidence", "analysis": "资料涉及蓝签和橙签两类标记。",
         "goal": "依据资料安排1天学习，区分蓝签和橙签。只阅读和口头说明，今天最多30分钟。",
         "evidence": SEP.join([blue, blue, orange]), "budget": len(SEP.join([blue, orange])),
         "expected": "保留两份独有证据；蓝签不代表理解，橙签为三例至少两例正确并说明依据。不编造未提供规则。"},
        {"case_id": "reading_handoff", "profile": "evidence", "analysis": "资料包含返回两个变量的函数。",
         "goal": "依据资料安排1天学习，解释规则中required_count与total_count的含义，只阅读和口头说明，最多20分钟。",
         "evidence": receipt, "budget": 24000,
         "expected": "变量分别为2和3，顺序一致；引用实际301—304行，不将请求400作为取得范围。"},
    ]


def run(case, arm, repeat, output):
    profile = "baseline" if arm == "baseline" else case["profile"]
    row = {"case_id": case["case_id"], "arm": arm, "profile": profile, "repeat": repeat,
           "user_input": case["goal"], "request_messages": [], "model_responses": []}
    started = time.perf_counter()
    with replacement(profile, output / "replacement-files" / profile):
        invoke = agents._invoke_llm_raw
        select = agents._limit_evidence_context

        def observed(llm, messages, *, stage):
            row["request_messages"].append({"stage": stage, "messages": [message_record(m) for m in messages]})
            response = invoke(llm, messages, stage=stage)
            row["model_responses"].append({"content": response.content,
                "tool_calls": response.tool_calls or [], "usage": response.usage_metadata or {}})
            return response

        def budgeted(context, *, selection_trace=None):
            return select(context, case["budget"], selection_trace=selection_trace)

        trace = {}
        try:
            # 固定规划输入，只比较候选政策/证据整理，不测真实RAG或初始分析。
            with patch.object(agents, "_invoke_llm_raw", observed), patch.object(agents, "needs_rag", lambda _: False), patch.object(agents, "_limit_evidence_context", budgeted):
                row["final_answer"] = agents.create_study_plan(
                    case["analysis"], case["evidence"], case["goal"], 1,
                    material_dir="frozen_local_fixture", mode="pipeline", plan_evidence_trace=trace,
                )
            row["status"] = "completed"
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__)
        row["plan_trace"] = trace
    row["latency_ms"] = round((time.perf_counter() - started) * 1000)
    row["usage"] = summarize_chat_usage({"model_responses": row["model_responses"]})
    row["request_sha256"] = hashlib.sha256(json.dumps(row["request_messages"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--case-id", choices=[case["case_id"] for case in cases()])
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    output = args.output_dir
    if (output / "manifest.json").exists():
        parser.error("拒绝覆盖已有实验")
    output.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    before = fingerprint()
    selected_cases = [case for case in cases() if args.case_id is None or case["case_id"] == args.case_id]
    manifest = {"mode": "online" if args.llm else "preflight_only", "source_sha256": before,
                "model": settings.chat_model, "temperature": settings.llm_temperature,
                "timeout_s": settings.request_timeout, "repeats": args.repeats, "max_tasks": len(selected_cases) * args.repeats * 2,
                "helper_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), Path(__file__).with_name("scaffold_lab.py")]},
                "retrieval": "disabled", "stage": "planning_only", "cases": selected_cases}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.llm:
        print(f"预检完成：{len(selected_cases)}个案例，{args.repeats}轮，显式--llm才执行{manifest['max_tasks']}次规划。")
        return 0
    transcript = []
    with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
        for case in selected_cases:
            for repeat in range(1, args.repeats + 1):
                order = ("baseline", "candidate") if repeat == 1 else ("candidate", "baseline")
                for arm in order:
                    row = run(case, arm, repeat, output)
                    row["source_unchanged"] = before == fingerprint()
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    transcript.append(f"## {case['case_id']} {arm} repeat={repeat}\n\n输入：{case['goal']}\n\n{row.get('final_answer', row.get('error_type'))}\n")
                    (output / "transcript.md").write_text("\n".join(transcript), encoding="utf-8")
                    print(f"{case['case_id']} {arm} repeat={repeat}: {row['status']}", flush=True)
                    if row["status"] != "completed" or not row["source_unchanged"]:
                        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
