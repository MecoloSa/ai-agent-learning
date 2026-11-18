"""从已保存的观察派生统计及可读全文，不发模型请求、不覆盖原结果。"""
import argparse
import json
from pathlib import Path

from tests.helpers.learning_holdout import FEEDBACK


def build_report(root):
    rows = []
    for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        directory = root / row["case_id"]
        models, transcript = [], []
        workflow_path = directory / "initial-workflow.json"
        if workflow_path.exists():
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            models.extend(workflow["models"])
            transcript += ["# 初始分析", workflow["analysis"], "# 初始计划", workflow["plan"]]
        tools = []
        for phase in ("first", "resumed"):
            path = directory / f"{phase}-observation.json"
            if not path.exists():
                continue
            obs = json.loads(path.read_text(encoding="utf-8"))
            models.extend(obs["models"])
            transcript += [f"# {phase}实际输入", json.dumps(obs["input_events"], ensure_ascii=False, indent=2)]
            for turn in obs["turns"]:
                transcript += ["## 用户", turn["user_input"], "## 助手", turn.get("response", "未交付回答")]
                results = turn["audit_trace"].get("tool_results", [])
                tools.extend({"phase": phase, "tool_name": t["tool_name"], "executed": t["executed"]} for t in results)
                transcript += ["## 实际工具", json.dumps(results, ensure_ascii=False, indent=2)]
            if phase == "first":
                observations = obs.get("state_observations", [])
                # 输入顺序是task、feedback、discussion、candidate；比较讨论前后的落盘状态。
                row["discussion_preserves_state"] = len(observations) >= 4 and observations[2]["state"] == observations[3]["state"]
                row["reviews"] = [{"decision": r["decision"], "before_revision": r["before"]["context_revision"],
                    "after_revision": r["after"]["context_revision"]} for r in obs["plan_reviews"]]
            if phase == "resumed":
                text = obs["turns"][0].get("response", "") if obs["turns"] else ""
                row["history_citation_warning"] = "【历史引用未核实】" in text
                row["feedback_text_in_restored_answer"] = FEEDBACK in text
        row["logical_model_calls"] = len(models)
        for field in ("input_tokens", "output_tokens"):
            values = [(m.get("usage") or {}).get(field) for m in models]
            row[field] = sum(values) if all(isinstance(v, int) for v in values) else None
        row["model_elapsed_ms"] = round(sum(m.get("elapsed_ms", 0) for m in models), 2)
        row["interactive_tools"] = tools
        (directory / "readable-transcript.md").write_text("\n\n".join(transcript), encoding="utf-8")
        rows.append(row)
    (root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    print(json.dumps(build_report(args.directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
