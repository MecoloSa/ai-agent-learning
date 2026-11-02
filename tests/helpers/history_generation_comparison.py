"""冻结已读证据后的生成对比；默认预检，--llm才调用两个已配置模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ai_agent.config import create_llm, get_settings
from ai_agent.runtime.agents import invoke_llm_response

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = [ROOT / ".evaluation-results" / name for name in (
    "history-citation-applied-014-20261031.jsonl",
    "history-citation-applied-015-20261031.jsonl",
    "history-scope-016-20261031.jsonl",
    "history-scope-017-20261031.jsonl",
)]


def message_from_record(row):
    role, content = row["role"], row["content"]
    if role == "ai":
        return AIMessage(content=content, tool_calls=row.get("tool_calls") or [])
    if role == "tool":
        return ToolMessage(content=content, tool_call_id=row["tool_call_id"])
    classes = {"human": HumanMessage, "system": SystemMessage}
    if role not in classes:
        raise ValueError("不支持的消息角色")
    return classes[role](content=content)


def load_frozen_case(path):
    result = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    observation = json.loads(Path(result["observation_path"]).read_text(encoding="utf-8"))
    request = observation["model_requests"][-1]
    # 本实验只接收读取成功后等待生成的请求，不能把恢复或强制收束混入对比。
    if not request["stage"].startswith("交互回答-工具判断"):
        raise ValueError("最后一次请求不是正常工具循环生成阶段")
    citations = []
    for row in request["request_messages"]:
        if row["role"] == "tool":
            payload = json.loads(row["content"])
            data = payload.get("data", {})
            if payload.get("ok") is True and data.get("evidence_kind") == "archived_conversation":
                citations.append(data["citation"])
    if not citations:
        raise ValueError("冻结输入没有成功回读的历史证据")
    messages = request["request_messages"]
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "case_id": result["case_id"], "source_run_id": result["run_id"],
        "source_py_sha256": result["manifest"]["source_py_sha256"],
        "question": observation["user_input"], "history": observation["fixture_history"],
        "request_messages": messages, "request_sha256": hashlib.sha256(encoded).hexdigest(),
        "verified_citations": list(dict.fromkeys(citations)),
    }


def run_comparison(cases, models):
    rows = []
    clients = {role: create_llm(role=role) for role in models}
    for index, case in enumerate(cases):
        # 交替先后顺序，减少始终先调用某模型的时间顺序偏差；不是统计显著性保证。
        roles = list(models) if index % 2 == 0 else list(reversed(models))
        for role in roles:
            print(f"生成对比：{case['case_id']} / {role}", flush=True)
            started = time.monotonic()
            row = {"case_id": case["case_id"], "role": role, "model": models[role],
                   "request_sha256": case["request_sha256"], "status": "completed",
                   "error_type": None, "answer": None, "usage": None,
                   "assessment": "not_evaluated"}
            try:
                # 两模型获得同一完整消息；不绑定工具，不添加额外提示，不运行写入流程。
                response = invoke_llm_response(clients[role],
                    [message_from_record(m) for m in case["request_messages"]],
                    stage="历史固定证据生成对比")
                answer = str(response.content or "")
                markers = re.findall(r"\[history=[^\]\r\n]*\]", answer)
                row.update(answer=answer, usage=getattr(response, "usage_metadata", None),
                           tool_call_count=len(getattr(response, "tool_calls", None) or []),
                           has_verified_citation=any(c in answer for c in case["verified_citations"]),
                           unknown_history_citations=[c for c in markers if c not in case["verified_citations"]])
            except Exception as exc:
                # 不输出异常正文中的地址/凭据，也不为本实验增加第二层重试。
                row.update(status="failed", error_type=type(exc).__name__)
            row["latency_ms"] = round((time.monotonic() - started) * 1000)
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=ROOT / ".evaluation-results/history-generation-comparison-20261031.json")
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args(argv)
    cases = [load_frozen_case(path) for path in args.results]
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("案例重复")
    settings = get_settings()
    models = {"standard": settings.chat_model, "enhanced": settings.enhanced_model}
    if models["standard"] == models["enhanced"]:
        raise ValueError("两个配置指向同一模型，不能作为模型对比")
    payload = {"mode": "frozen_generation_only", "models": models,
               "settings": {"temperature": settings.llm_temperature,
                            "timeout": settings.request_timeout,
                            "retry_attempts": settings.llm_retry_attempts},
               "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "cases": cases, "rows": []}
    if not args.llm:
        print(json.dumps({"mode": "preflight_only", "models": models,
                          "calls_planned": len(cases) * 2,
                          "cases": [{k: c[k] for k in ("case_id", "question", "history", "request_sha256")} for c in cases]},
                         ensure_ascii=False, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 先独占创建产物，避免执行模型后才发现结果文件已存在。
    with args.output.open("x", encoding="utf-8") as stream:
        payload["rows"] = run_comparison(cases, models)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "completed": sum(r["status"] == "completed" for r in payload["rows"]),
                      "failed": sum(r["status"] == "failed" for r in payload["rows"])}, ensure_ascii=False))
    return 0 if all(r["status"] == "completed" for r in payload["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
