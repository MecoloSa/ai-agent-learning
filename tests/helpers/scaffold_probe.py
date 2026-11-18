"""相同固定工具调用下比较执行策略；不评价模型自主决策。"""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch
import argparse

from langchain_core.messages import AIMessage, ToolMessage
import ai_agent.runtime.agents as agents
from tests.helpers.scaffold_lab import replacement, fingerprint, stage_evidence


def probe(case, profile, output):
    calls = []
    observed = []

    class Tool:
        name = "read_learning_file"

        def invoke(self, args):
            calls.append(dict(args))
            if len(calls) <= case["failures"]:
                raise (TimeoutError if case["error"] == "timeout" else ValueError)("合成故障")
            return "[source=a.md]\n正文"

    responses = iter([AIMessage(content="", tool_calls=[{
        "id": f"call-{n}", "name": "read_learning_file",
        "args": {"relative_path": path}, "type": "tool_call",
    }]) for n, path in enumerate(case["paths"], 1)] + [AIMessage(content="固定终止回复，不代表内容质量")])

    def invoke(llm, messages, **kwargs):
        observed[:] = [{"tool_call_id": m.tool_call_id, "content": m.content}
                       for m in messages if isinstance(m, ToolMessage)]
        return next(responses)

    with replacement(profile, output / "replacement-files" / profile):
        settings = replace(agents.settings, agent_max_tool_calls=case["budget"], agent_max_no_progress_rounds=2)
        with patch.object(agents, "settings", settings), patch.object(agents, "MAX_TOOL_ROUNDS", 12), patch.object(agents, "build_agent_tools", lambda *a, **k: [Tool()]), patch.object(agents, "create_llm_with_tools", lambda *a: object()), patch.object(agents, "_invoke_llm_raw", invoke):
            try:
                agents.analyse_materials_agentic("unused", case["user_input"])
                status, reason = "completed", None
            except agents.AnalysisExecutionError as exc:
                status, reason = "stopped", exc.reason
    return {"case_id": case["case_id"], "profile": profile, "status": status, "reason": reason,
            "real_tool_calls": len(calls), "actual_args": calls, "tool_observations": observed,
            "mode": "offline_fixed_model_responses", "case_spec": case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        parser.error("拒绝覆盖结果")
    cases = [
        {"case_id": "seven_reads", "user_input": "读取七份资料并比较差异，不修改记录。", "paths": [f"{n}.md" for n in range(7)], "failures": 0, "error": "timeout", "budget": 12},
        {"case_id": "transient_retry", "user_input": "读取a.md；临时失败后继续原请求，不修改记录。", "paths": ["a.md", "a.md"], "failures": 1, "error": "timeout", "budget": 12},
        {"case_id": "persistent_timeout", "user_input": "读取a.md，不修改记录。", "paths": ["a.md"] * 4, "failures": 99, "error": "timeout", "budget": 12},
        {"case_id": "deterministic_failure", "user_input": "读取a.md，不修改记录。", "paths": ["a.md"] * 4, "failures": 99, "error": "value", "budget": 12},
        {"case_id": "total_budget", "user_input": "读取a.md，不修改记录。", "paths": ["a.md"] * 3, "failures": 99, "error": "timeout", "budget": 2},
    ]
    before = fingerprint()
    rows = [probe(case, profile, output) for case in cases for profile in ("baseline", "combined")]
    (output / "results.json").write_text(json.dumps({"source_unchanged": before == fingerprint(), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(row["case_id"], row["profile"], row["status"], "real_calls=", row["real_tool_calls"], row["reason"])


if __name__ == "__main__":
    main()
