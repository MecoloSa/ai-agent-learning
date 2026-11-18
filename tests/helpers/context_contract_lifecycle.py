"""组合候选的采用及新进程恢复验收；默认预检。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from tests.helpers.context_contract_lab import ROOT, FIXTURE, prepare, summarize
from tests.helpers.learning_closed_loop import formal
from tests.helpers.receipt_view_compare import fingerprint
from ai_agent.persistence.state_store import load_learning_state

RESUME = "请复述当前正式计划和read-001的活动进度与理解反馈；说明它与未采用的旧候选的关系。只查询，不修改任何记录。"


def launch(spec, directory):
    (directory / "input.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    with (directory / "console.log").open("w", encoding="utf-8") as stream:
        return subprocess.run([sys.executable, "-X", "utf8", "-m", "tests.helpers.context_contract_lab",
            "--worker", str(directory / "input.json"), "--llm"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=900).returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--llm", action="store_true")
    p.add_argument("--variant", choices=["combined", "combined_v2"], default="combined")
    args = p.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case = {**fixture["groups"]["history"][2], "case_id": "combined_adoption", "seed_program_receipt": True}
    before = fingerprint()
    (output / "manifest.json").write_text(json.dumps({"mode": "online" if args.llm else "preflight_only",
        "case": case, "variant": args.variant, "confirmation": "y", "resume": RESUME, "source_sha256": before}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.llm:
        print("组合采用/恢复预检完成；未调用模型。")
        return 0
    started = perf_counter()
    adopt = output / "adopt"
    adopt.mkdir()
    spec = prepare(adopt, fixture, case)
    spec.update(variant=args.variant, observe_only=False, events=[{"stage": "chat", "text": case["query"]},
        {"stage": "confirm", "text": "y"}, {"stage": "chat", "text": "exit"}])
    code = launch(spec, adopt)
    result = {"adopt_code": code, "case": case}
    if code == 0:
        source = load_learning_state(adopt / "source.json")
        updated = load_learning_state(adopt / "current.json")
        first_bytes = (adopt / "current.json").read_bytes()
        resumed = output / "resumed"
        resumed.mkdir()
        second = {**spec, "source": str(adopt / "current.json"), "current": str(resumed / "current.json"),
            "observation": str(resumed / "observation.json"), "lifecycle": str(resumed / "lifecycle.json"),
            "events": [{"stage": "chat", "text": RESUME}, {"stage": "chat", "text": "exit"}], "probe_input": RESUME}
        # 保存同目录只读source副本供统一报告比较；模型实际恢复上面的采用快照。
        (resumed / "source.json").write_bytes(first_bytes)
        code = launch(second, resumed)
        result["resume_code"] = code
        if code == 0:
            final = load_learning_state(resumed / "current.json")
            result["checks"] = {"revision_once": updated.context_revision == source.context_revision + 1,
                "tasks_feedback_preserved": updated.tasks == source.tasks and updated.feedback_records == source.feedback_records,
                "restore_unchanged": formal(updated) == formal(final),
                "adopted_source_not_overwritten": first_bytes == (adopt / "current.json").read_bytes()}
        rows = [summarize(adopt, args.variant, case, before)]
        if (resumed / "observation.json").exists():
            rows.append(summarize(resumed, args.variant, {"case_id": "combined_restore"}, before))
        (output / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    result.update(source_unchanged=fingerprint() == before, elapsed_s=round(perf_counter() - started, 2))
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if code == 0 and all(result.get("checks", {}).values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
