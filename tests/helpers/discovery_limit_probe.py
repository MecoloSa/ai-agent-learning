"""清单边界信息ABBA；同一生产循环和整理策略，只比较旧/新清单返回。"""
import argparse
from dataclasses import replace
import json
import hashlib
from pathlib import Path
from unittest.mock import patch

import ai_agent.resources.tools as tools
import ai_agent.runtime.agents as agents
from tests.helpers.receipt_agent_compare import ROOT, run_chain
from tests.helpers.receipt_view_compare import fingerprint

CASE = {"case_id": "hidden_advanced_rule", "seed_calls": [],
    "user_input": "本目录是否规定了高级标记的通过条件？若有请准确说明并注明来源；若未找到，请说明本次检查范围。",
    "materials": {"a_notes.md": "完成标记表示已提交记录，不代表掌握。\n",
        "b_notes.md": "掌握标记表示通过独立检验。\n",
        "z_condition.md": "高级标记要求完成一次口头答辩，并说明两项判断依据。\n"},
    "expected": "取得z_condition.md后回答口头答辩与两项依据；未取得时不能把有限清单当作全部目录。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        parser.error("拒绝覆盖已有实验")
    settings = replace(tools.get_settings(), tool_file_list_limit=2)
    baseline_path = ROOT / "tests/fixtures/evaluation/discovery_baseline_factory.py"
    namespace = {**tools.__dict__, "get_settings": lambda: settings}
    exec(compile(baseline_path.read_text(encoding="utf-8"), str(baseline_path), "exec"), namespace)
    baseline_factory = namespace["build_agent_tools"]
    manifest = {"mode": "online" if args.llm else "preflight_only", "chains": 4,
        "model": settings.chat_model, "temperature": settings.llm_temperature,
        "source_sha256": fingerprint(), "list_limit": 2,
        "baseline_factory_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "comparison": "旧版清单/新版边界提示；两组均用当前完整回执整理策略"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "inputs.json").write_text(json.dumps(CASE, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.llm:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    transcript = []
    with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
        for repeat, order in ((1, ("baseline", "notice")), (2, ("notice", "baseline"))):
            for arm in order:
                factory = baseline_factory if arm == "baseline" else tools.build_agent_tools
                with patch.object(tools, "get_settings", lambda: settings), patch.object(agents, "build_agent_tools", factory):
                    row = run_chain(CASE, "compact", repeat, output / "materials" / f"{arm}-{repeat}")
                row["discovery_arm"] = arm
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                transcript.append(f"## {arm} {repeat}\n\n{CASE['user_input']}\n\n"
                    + str(row.get("final_answer", row.get("error_type"))) + "\n\n```json\n"
                    + json.dumps(row["tool_executions"], ensure_ascii=False, indent=2) + "\n```\n")
                (output / "transcript.md").write_text("\n".join(transcript), encoding="utf-8")
                print(f"{arm} {repeat}: {row['status']} models={len(row['model_responses'])} "
                    f"tools={len(row['tool_executions'])} usage={row['usage']}", flush=True)
                if not row["source_unchanged"] or not row["materials_unchanged"]:
                    return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
