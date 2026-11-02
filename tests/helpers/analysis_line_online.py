"""真实分析Agent长文件回读观察。默认仅预检，--llm才调用聊天模型。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from ai_agent.config import get_settings
import ai_agent.runtime.agents as agents
from ai_agent.runtime.context_handoff import unpack_material_context
from ai_agent.evaluation.acquisition_pair import message_record
from ai_agent.evaluation.interactive_metrics import summarize_chat_usage


ROOT = Path("tests/fixtures/evaluation/analysis_line_online/materials")
QUESTION = "解释rules.py中target_rule返回什么，以及每个返回值对应哪个变量。依据文件原文说明，不执行代码。"


def observe():
    original = agents._invoke_llm_raw
    requests, responses = [], []

    def invoke(llm, messages, *, stage):
        requests.append({"stage": stage, "messages": [message_record(m) for m in messages]})
        started = time.perf_counter()
        response = original(llm, messages, stage=stage)
        responses.append({"stage": stage, "content": response.content, "tool_calls": response.tool_calls or [], "usage": response.usage_metadata or {}, "latency_ms": round((time.perf_counter() - started) * 1000)})
        return response

    started = time.perf_counter()
    row = {"user_input": QUESTION}
    try:
        with patch.object(agents, "_invoke_llm_raw", invoke):
            analysis, context = agents.analyse_materials_agentic(str(ROOT), QUESTION)
        evidence, facts = unpack_material_context(context)
        row.update(status="completed", final_answer=analysis, evidence=evidence, acquisition_facts=facts)
    except Exception as exc:
        row.update(status="failed", error_type=type(exc).__name__)
    row.update(request_messages=requests, model_responses=responses,
               latency_ms=round((time.perf_counter() - started) * 1000),
               usage=summarize_chat_usage({"model_responses": responses}))
    calls = {call["id"]: call for response in responses for call in response["tool_calls"]}
    # 最后一轮请求包含此前所有ToolMessage；按ID去重保存实际观察。
    observations = {}
    for request in requests:
        for message in request["messages"]:
            if message["role"] == "tool":
                identity = message["tool_call_id"]
                observations[identity] = {"tool_name": calls[identity]["name"], "args": calls[identity]["args"], "result": message["content"]}
    row["tool_results"] = list(observations.values())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "observation.json"
    if path.exists():
        parser.error("观察文件已存在，请使用新目录")
    source = (ROOT / "rules.py").read_text(encoding="utf-8")
    settings = get_settings()
    manifest = {"mode": "online" if args.llm else "preflight_only", "user_input": QUESTION,
                "chat_model": settings.chat_model, "temperature": settings.llm_temperature,
                "file": str(ROOT / "rules.py"), "file_chars": len(source),
                "prefix_limit_chars": settings.tool_max_chars_per_file,
                "function_start_line": source[:source.index("def target_rule")].count("\n") + 1,
                "file_sha256": hashlib.sha256(source.encode()).hexdigest()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    if not args.llm:
        return 0
    row = observe()
    row["source_unchanged"] = hashlib.sha256((ROOT / "rules.py").read_text(encoding="utf-8").encode()).hexdigest() == manifest["file_sha256"]
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": row["status"], "tools": [r["tool_name"] for r in row["tool_results"]], "latency_ms": row["latency_ms"]}, ensure_ascii=False), flush=True)
    return 0 if row["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
