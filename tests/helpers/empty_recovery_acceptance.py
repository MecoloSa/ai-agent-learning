"""空响应恢复接线：离线注入首次空响应，再真实重发；不估计自然故障恢复率。"""
import argparse
import json
from pathlib import Path

from langchain_core.messages import AIMessage
from ai_agent.config import create_llm
from ai_agent.runtime.agents import invoke_llm_response
from ai_agent.evaluation.acquisition_pair import message_record
from tests.helpers.empty_response_diagnosis import load_failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        parser.error("不覆盖已有验收")
    (root / "manifest.json").write_text(json.dumps({"mode": "injected_empty_then_live" if args.llm else "preflight_only",
        "cases": ["discovery_failure", "fixed_failure"], "synthetic_first_response": True,
        "scope": "证明生产恢复接线，不证明自然空响应发生率或线上恢复率"}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.llm:
        return 0
    with (root / "results.jsonl").open("x", encoding="utf-8") as stream:
        for case in ("discovery_failure", "fixed_failure"):
            messages, tools, _ = load_failed(case)
            llm = create_llm()
            live = llm.bind_tools(tools) if tools else llm
            row = {"case_id": case, "actual_requests": [], "injected_empty": True}
            class Injected:
                def invoke(self, actual):
                    row["actual_requests"].append([message_record(m) for m in actual])
                    if len(row["actual_requests"]) == 1:
                        return AIMessage(content="", usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                    return live.invoke(actual)
            try:
                response = invoke_llm_response(Injected(), messages, stage="空响应恢复接线验收")
                row.update(status="completed", answer=response.content, tool_calls=response.tool_calls,
                    usage=response.usage_metadata, audit=response.response_metadata.get("invocation_audit"))
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__, audit=getattr(exc, "invocation_audit", None))
            row["same_request"] = all(r == row["actual_requests"][0] for r in row["actual_requests"])
            row["real_model_requests"] = len(row["actual_requests"]) - 1
            row["tool_executions"] = 0
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{case}: {row['status']} real_requests={row['real_model_requests']} same_request={row['same_request']}", flush=True)
    return 0


if __name__ == "__main__":
    main()
