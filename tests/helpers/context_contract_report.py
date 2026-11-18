"""只读聚合本轮实验；程序状态检查与内容审查分开，不发模型请求。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from ai_agent.persistence.state_store import load_learning_state
from tests.helpers.context_contract_lab import footer_sources


def known_sum(values):
    return sum(values) if all(isinstance(v, (int, float)) for v in values) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for root in args.roots:
        for original in json.loads((root / "results.json").read_text(encoding="utf-8")):
            row = dict(original)
            directory = Path(row["directory"])
            obs = json.loads((directory / "observation.json").read_text(encoding="utf-8"))
            results = [t for turn in obs["turns"] for t in turn["audit_trace"].get("tool_results", [])]
            row["confirmation_inputs"] = [e["text"] for e in obs["input_events"] if e["stage"] == "confirm"]
            row["tool_errors"] = []
            for result in results:
                try:
                    payload = json.loads(result["result"])
                except (ValueError, TypeError):
                    continue
                if isinstance(payload, dict) and payload.get("ok") is False:
                    row["tool_errors"].append({"tool": result["tool_name"], "code": payload.get("error", {}).get("code")})
            source = load_learning_state(directory / "source.json")
            target = load_learning_state(directory / "current.json")
            row["revision_delta"] = target.context_revision - source.context_revision
            row["tasks_feedback_preserved"] = source.tasks == target.tasks and source.feedback_records == target.feedback_records
            row["expected_revision_delta"] = 1 if "y" in row["confirmation_inputs"] else 0
            row["revision_matches_expected"] = row["revision_delta"] == row["expected_revision_delta"]
            # 不把后来扩展的渲染器生成结果冒充旧模型当时的交付。
            row["actual_footers"] = [t.get("response", "").split("【程序列出的本轮查阅资料", 1)[1]
                for t in obs["turns"] if "【程序列出的本轮查阅资料" in t.get("response", "")]
            rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["variant"])].append(row)
    summary = [{"case_id": key[0], "variant": key[1], "runs": len(values),
        "model_calls": sum(r["model_calls"] for r in values), "input_tokens": known_sum([r["input_tokens"] for r in values]),
        "output_tokens": known_sum([r["output_tokens"] for r in values]),
        "elapsed_ms": known_sum([r["elapsed_ms"] for r in values]),
        "controls_pass": all(r["revision_matches_expected"] and r["tasks_feedback_preserved"] for r in values)}
        for key, values in groups.items()]
    payload = {"rows": rows, "groups": summary, "totals": {"segments": len(rows),
        "model_calls": sum(r["model_calls"] for r in rows), "attempts": sum(r["attempts"] for r in rows),
        "input_tokens": known_sum([r["input_tokens"] for r in rows]), "output_tokens": known_sum([r["output_tokens"] for r in rows]),
        "expected_revision_and_task_controls": all(r["revision_matches_expected"] and r["tasks_feedback_preserved"] for r in rows)}}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"groups": summary, "totals": payload["totals"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
