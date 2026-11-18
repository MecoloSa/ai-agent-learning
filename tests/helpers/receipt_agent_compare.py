"""生产分析循环对照：只脚本化首批读取，后续由真实模型与真实工具执行。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from contextlib import nullcontext
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

import ai_agent.runtime.agents as agents
from ai_agent.config import get_settings
from ai_agent.evaluation.acquisition_pair import message_record
from ai_agent.evaluation.interactive_metrics import summarize_chat_usage
from tests.helpers.receipt_view_compare import fingerprint

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/fixtures/evaluation/receipt_agent_cases.json"


def material_hash(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run_chain(case, arm, repeat, directory):
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in case["materials"].items():
        (directory / name).write_text(content, encoding="utf-8")
    source_before, materials_before = fingerprint(), material_hash(directory)
    row = {"case_id": case["case_id"], "arm": arm, "repeat": repeat,
        "user_input": case["user_input"], "expected": case["expected"],
        "materials": case["materials"], "seed_calls": case["seed_calls"],
        "requests": [], "model_responses": [], "tool_executions": [], "attempts": [],
        "quality_status": "pending_manual_review", "materials_sha256": materials_before}
    original_invoke = agents._invoke_llm_raw
    original_create = agents.create_llm_with_tools
    original_prepare = agents.prepare_model_messages
    original_tool_invoke = BaseTool.invoke
    seeded = False

    def prepare(history, **kwargs):
        before = deepcopy([m.model_dump() for m in history])
        view = original_prepare(history, tool_result_formatter=agents.stage_evidence if arm == "compact" else None)
        assert before == [m.model_dump() for m in history]
        row["requests"].append({"working_messages": [message_record(m) for m in history],
            "request_messages": [message_record(m) for m in view]})
        return view

    def create(tools):
        row["tool_schemas"] = [{"name": tool.name, "args": tool.args} for tool in tools]
        llm = original_create(tools)

        class Observed:
            def invoke(self, messages):
                attempt = {"index": len(row["attempts"]) + 1}
                row["attempts"].append(attempt)
                try:
                    response = llm.invoke(messages)
                    attempt["status"] = "completed"
                    return response
                except Exception as exc:
                    attempt.update(status="failed", error_type=type(exc).__name__)
                    raise
        return Observed()

    def invoke(llm, messages, *, stage):
        nonlocal seeded
        if not seeded and case.get("seed_calls"):
            seeded = True
            row["requests"][-1]["source"] = "scripted_seed_no_model_request"
            # 首批读取由实验固定；执行、校验、计费预算均经过正式循环。
            return AIMessage(content="", tool_calls=[{**call, "id": f"seed-{i}", "type": "tool_call"}
                for i, call in enumerate(case["seed_calls"], 1)])
        row["requests"][-1]["source"] = "model_request"
        response = original_invoke(llm, messages, stage=stage)
        row["model_responses"].append({"content": response.content, "tool_calls": response.tool_calls,
            "usage": response.usage_metadata or {}, "metadata": {key: response.response_metadata[key]
                for key in ("finish_reason", "model_name") if key in response.response_metadata}})
        return response

    def tool_invoke(tool, input, *args, **kwargs):
        entry = {"tool_name": tool.name, "args": deepcopy(input)}
        row["tool_executions"].append(entry)
        try:
            result = original_tool_invoke(tool, input, *args, **kwargs)
            entry.update(status="completed", result=str(result))
            return result
        except Exception as exc:
            entry.update(status="failed", error_type=type(exc).__name__)
            raise

    started = time.perf_counter()
    try:
        # 仅在本进程实验期间替换观察入口，不复制生产执行循环，不落盘改业务代码。
        with patch.object(agents, "prepare_model_messages", prepare), \
             patch.object(agents, "create_llm_with_tools", create), \
             patch.object(agents, "_invoke_llm_raw", invoke), \
             patch.object(BaseTool, "invoke", tool_invoke):
            answer, context = agents.analyse_materials_agentic(str(directory), case["user_input"])
        row.update(status="completed", final_answer=answer, material_context=context)
    except Exception as exc:
        row.update(status="failed", error_type=type(exc).__name__, error_reason=getattr(exc, "reason", None))
    row.update(latency_ms=round((time.perf_counter() - started) * 1000),
        source_unchanged=fingerprint() == source_before,
        materials_unchanged=material_hash(directory) == materials_before)
    row["usage"] = summarize_chat_usage(row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--policy", choices=("current", "selective"), default="current")
    parser.add_argument("--case-id", choices=("truncated_readback", "multiple_sources"))
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        parser.error("拒绝覆盖已有实验")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if args.case_id:
        cases = [case for case in cases if case["case_id"] == args.case_id]
    formatter_context = nullcontext()
    policy_metadata = {"policy": args.policy}
    if args.policy == "selective":
        from tests.helpers.selective_receipts import candidate_source, candidate_formatter
        source = candidate_source()
        (output / "stage_evidence_candidate.py").write_text(source, encoding="utf-8")
        policy_metadata["candidate_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        formatter_context = patch.object(agents, "stage_evidence", candidate_formatter())
    settings = get_settings()
    manifest = {"mode": "production_loop_seeded_reading" if args.llm else "preflight_only",
        "model": settings.chat_model, "temperature": settings.llm_temperature,
        "tool_limit": agents.settings.agent_max_tool_calls, "round_limit": agents.MAX_TOOL_ROUNDS,
        "source_sha256": fingerprint(), "cases_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        "chains": len(cases) * 4, "tools_bound": True, **policy_metadata,
        "scope": "首批读取脚本化并占正式预算，后续真实生产循环；不评价首次工具选择"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "inputs.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.llm:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    transcript = []
    with formatter_context, (output / "results.jsonl").open("x", encoding="utf-8") as stream:
        for case in cases:
            for repeat, order in ((1, ("raw", "compact")), (2, ("compact", "raw"))):
                for arm in order:
                    row = run_chain(case, arm, repeat, output / "materials" / f"{case['case_id']}-{arm}-{repeat}")
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    transcript.append(f"## {case['case_id']} {arm} {repeat}\n\n提问：{case['user_input']}\n\n"
                        + str(row.get("final_answer", row.get("error_type"))) + "\n\n工具过程：\n```json\n"
                        + json.dumps(row["tool_executions"], ensure_ascii=False, indent=2) + "\n```\n")
                    (output / "transcript.md").write_text("\n".join(transcript), encoding="utf-8")
                    print(f"{case['case_id']} {arm} {repeat}: {row['status']} models={len(row['model_responses'])} "
                        f"tools={len(row['tool_executions'])} usage={row['usage']}", flush=True)
                    if not row["source_unchanged"] or not row["materials_unchanged"]:
                        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
