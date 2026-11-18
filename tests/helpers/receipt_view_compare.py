"""固定读取历史的A/B回答对照；默认预检，--llm才调用真实模型。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ai_agent.config import create_llm, get_settings
from ai_agent.evaluation.acquisition_pair import message_record
from ai_agent.evaluation.interactive_metrics import summarize_chat_usage
from ai_agent.resources.tools import build_agent_tools
from ai_agent.runtime.agents import invoke_llm_response, stage_evidence
from ai_agent.runtime.request_context import prepare_model_messages

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/fixtures/evaluation/receipt_view_cases.json"


def fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/ai_agent").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_history(case, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / case["filename"]).write_text(case["material"], encoding="utf-8")
    tools = {tool.name: tool for tool in build_agent_tools(str(directory))}
    history = [SystemMessage(content="依据实际取得的资料回答用户问题，引用来源；区分原文与补充解释，缺少依据时说明。"),
        HumanMessage(content=case["user_input"])]
    for index, call in enumerate(case["calls"], 1):
        identity = f"read-{index}"
        history.append(AIMessage(content="", tool_calls=[{
            "id": identity, "name": call["name"], "args": call["args"], "type": "tool_call"}]))
        try:
            result = tools[call["name"]].invoke(call["args"])
        except FileNotFoundError:
            # 与分析循环的确定性文件缺失回执保持一致，不转发本机路径。
            result = json.dumps({"ok": False, "error": {"code": "FileNotFoundError",
                "retryable": False, "message": "本次未取得结果；可在剩余预算内按错误类型调整。"}}, ensure_ascii=False)
        history.append(ToolMessage(content=result, tool_call_id=identity))
    return history


def request(history, arm):
    return prepare_model_messages(history, tool_result_formatter=stage_evidence if arm == "compact" else None)


def run_one(case, history, arm, repeat, llm, source_hash):
    messages = request(history, arm)
    row = {"case_id": case["case_id"], "arm": arm, "repeat": repeat,
        "user_input": case["user_input"], "expected": case["expected"],
        "request_messages": [message_record(m) for m in messages],
        "model_responses": [], "attempts": [], "quality_status": "pending_manual_review"}

    class Observed:
        # 包住每次真正invoke，外层项目重试发生时仍能计数。
        def invoke(self, actual):
            attempt = {"index": len(row["attempts"]) + 1}
            row["attempts"].append(attempt)
            try:
                response = llm.invoke(actual)
                attempt["status"] = "completed"
                return response
            except Exception as exc:
                attempt.update(status="failed", error_type=type(exc).__name__)
                raise

    started = time.perf_counter()
    try:
        response = invoke_llm_response(Observed(), messages, stage="读取回执表示对照")
        row.update(status="completed", final_answer=response.content,
            returned_tool_calls=getattr(response, "tool_calls", []))
        row["response_kind"] = ("tool_call" if row["returned_tool_calls"] else
            "answer" if isinstance(response.content, str) and response.content.strip() else "empty")
        # 只记录公开响应状态，不保存模型私有推理内容。
        row["response_metadata"] = {key: response.response_metadata[key]
            for key in ("finish_reason", "model_name") if key in response.response_metadata}
        row["model_responses"].append({"content": response.content,
            "usage": response.usage_metadata or {}})
    except Exception as exc:
        row.update(status="failed", error_type=type(exc).__name__)
    row["latency_ms"] = round((time.perf_counter() - started) * 1000)
    row["usage"] = summarize_chat_usage(row)
    row["source_unchanged"] = fingerprint() == source_hash
    row["request_sha256"] = hashlib.sha256(json.dumps(row["request_messages"],
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        parser.error("不覆盖已有实验；请使用新目录")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    settings = get_settings()
    source_hash = fingerprint()
    prepared = []
    for case in cases:
        history = build_history(case, output / "materials" / case["case_id"])
        before = deepcopy([m.model_dump() for m in history])
        views = {arm: [message_record(m) for m in request(history, arm)] for arm in ("raw", "compact")}
        assert [m.model_dump() for m in history] == before
        prepared.append({**case, "raw_history": [message_record(m) for m in history], "views": views})
    (output / "inputs.json").write_text(json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"mode": "llm_fixed_context" if args.llm else "preflight_only",
        "model": settings.chat_model, "temperature": settings.llm_temperature,
        "source_sha256": source_hash, "cases_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        "planned_generations": 12, "tools_bound": False,
        "scope": "固定工具历史的回答对照，不评估自主工具选择；仅合成资料发送到当前模型端点"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.llm:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    llm = create_llm()
    transcript = []
    with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
        for case in prepared:
            # 同一份真实工具结果复用给四次模型请求，不重新读取。
            from langchain_core.messages import messages_from_dict
            history = messages_from_dict([{"type": m["role"], "data": {
                key: value for key, value in m.items() if key != "role"}} for m in case["raw_history"]])
            for repeat, order in ((1, ("raw", "compact")), (2, ("compact", "raw"))):
                for arm in order:
                    row = run_one(case, history, arm, repeat, llm, source_hash)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    transcript.append(f"## {case['case_id']} {arm} {repeat}\n\n提问：{case['user_input']}\n\n"
                        + str(row.get("final_answer", row.get("error_type")))
                        + "\n响应类别：" + str(row.get("response_kind"))
                        + "\n工具请求：" + json.dumps(row.get("returned_tool_calls", []), ensure_ascii=False) + "\n")
                    (output / "transcript.md").write_text("\n".join(transcript), encoding="utf-8")
                    print(f"{case['case_id']} {arm} {repeat}: {row['status']} usage={row['usage']} attempts={len(row['attempts'])}", flush=True)
                    if row["status"] != "completed" or not row["source_unchanged"]:
                        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
