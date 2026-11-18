"""无脚本起点的正式分析实验；默认预检，仅合成资料，--llm才运行4条链。"""
import argparse
import hashlib
import json
from pathlib import Path

from ai_agent.config import get_settings
from tests.helpers.receipt_agent_compare import ROOT, run_chain
from tests.helpers.receipt_view_compare import fingerprint

CASES = ROOT / "tests/fixtures/evaluation/autonomous_material_cases.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        parser.error("拒绝覆盖已有实验")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    settings = get_settings()
    manifest = {"mode": "autonomous_production_loop" if args.llm else "preflight_only",
        "model": settings.chat_model, "temperature": settings.llm_temperature,
        "source_sha256": fingerprint(), "cases_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        "chains": 4, "scripted_tool_requests": 0, "policy": "current_selective_receipts"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "inputs.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.llm:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    transcript = []
    with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
        for case in cases:
            for repeat in (1, 2):
                row = run_chain(case, "compact", repeat, output / "materials" / f"{case['case_id']}-{repeat}")
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                transcript.append(f"## {case['case_id']} {repeat}\n\n问题：{case['user_input']}\n\n"
                    + str(row.get("final_answer", row.get("error_type"))) + "\n\n工具过程：\n```json\n"
                    + json.dumps(row["tool_executions"], ensure_ascii=False, indent=2) + "\n```\n")
                (output / "transcript.md").write_text("\n".join(transcript), encoding="utf-8")
                print(f"{case['case_id']} {repeat}: {row['status']} models={len(row['model_responses'])} "
                    f"tools={len(row['tool_executions'])} usage={row['usage']}", flush=True)
                if not row["source_unchanged"] or not row["materials_unchanged"]:
                    return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
