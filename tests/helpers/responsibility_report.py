"""汇总三组真实轨迹，保留失败；引用检查不以“无警告”替代成功展开。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from tests.helpers.learning_closed_loop import DISCUSSION
from tests.helpers.learning_closed_loop import formal
from ai_agent.persistence.state_store import load_learning_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for root in args.roots:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/observation.json")):
            obs = json.loads(path.read_text(encoding="utf-8"))
            spec = json.loads((path.parent / "input.json").read_text(encoding="utf-8"))
            models = obs["models"]
            text = obs["turns"][0].get("response", "") if obs["turns"] else ""
            results = [t for turn in obs["turns"] for t in turn["audit_trace"].get("tool_results", [])]
            citations = []
            for result in results:
                if result["tool_name"] == "read_conversation_turn" and result["executed"]:
                    citations.append(json.loads(result["result"]).get("data", {}).get("citation"))
            row = {"experiment": manifest["experiment"], "directory": str(path.parent),
                "variant": spec["variant"], "category": path.parent.name.rsplit("-", 1)[1],
                "status": obs["status"], "error": obs.get("error_message"), "reviews": len(obs["plan_reviews"]),
                "model_calls": len(models), "attempts": sum((m.get("invocation_audit") or {}).get("attempt_count", 1) for m in models),
                "empty_responses": sum((m.get("invocation_audit") or {}).get("empty_responses", 0) for m in models),
                "exact_quote": DISCUSSION in text,
                "expanded_citation": any(c and c in text for c in citations) and "【历史引用未核实】" not in text,
                "elapsed_ms": round(sum(m.get("elapsed_ms", 0) for m in models), 2),
                "denial_phrase_signal": "资料否定因果" in text or "资料为什么否定" in text,
                "current_snapshot_written": (path.parent / "current.json").exists()}
            row["state_unchanged"] = formal(load_learning_state(path.parent / "current.json")) == formal(load_learning_state(path.parent / "source.json")) if row["current_snapshot_written"] else None
            for field in ("input_tokens", "output_tokens"):
                values = [(m.get("usage") or {}).get(field) for m in models]
                row[field] = sum(values) if all(isinstance(v, int) for v in values) else None
            transcript = ["# 实际用户输入", spec["probe_input"], "# 交付", text,
                "# 工具记录", json.dumps(results, ensure_ascii=False, indent=2)]
            stage = path.parent / "stage-results.json"
            if stage.exists():
                row["stage_results"] = json.loads(stage.read_text(encoding="utf-8"))
            (path.parent / "readable-transcript.md").write_text("\n\n".join(transcript), encoding="utf-8")
            rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["experiment"], row["variant"], row["category"])].append(row)
    summary = []
    for key, values in groups.items():
        item = {"experiment": key[0], "variant": key[1], "category": key[2], "runs": len(values),
            "completed": sum(r["status"] == "completed" for r in values),
            "reviews": sum(r["reviews"] for r in values),
            "citations": sum(r["expanded_citation"] for r in values),
            "denial_phrase_signals": sum(r["denial_phrase_signal"] for r in values)}
        for field in ("input_tokens", "output_tokens", "elapsed_ms"):
            item["mean_" + field] = round(sum(r[field] for r in values) / len(values), 2) if all(r[field] is not None for r in values) else None
        summary.append(item)
    payload = {"rows": rows, "groups": summary, "total_model_calls": sum(r["model_calls"] for r in rows),
        "total_attempts": sum(r["attempts"] for r in rows),
        "total_input_tokens": sum(r["input_tokens"] for r in rows),
        "total_output_tokens": sum(r["output_tokens"] for r in rows)}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
