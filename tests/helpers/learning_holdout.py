"""两条新任务闭环；默认预检，--llm才运行真实分析及交互。"""
import argparse
import json
from pathlib import Path
from time import perf_counter

from ai_agent.config import get_settings
from tests.helpers.learning_closed_loop import run_case, DISCUSSION, RESUME
from tests.helpers.receipt_view_compare import fingerprint

CASES = Path(__file__).resolve().parents[1] / "fixtures/evaluation/learning_holdout_cases.json"
CANDIDATE = "请依据已有反馈调整后续学习计划，总计最多20分钟，只阅读和口头说明，保留学习目标和任务反馈，提交完整候选计划供我确认。"
FEEDBACK = "阅读已经完成，但我仍分不清这两个概念在实际例子中的区别。"


def flow_events():
    commands = [
        '/task {"task_id":"read-001","title":"阅读资料并口头比较"}',
        '/feedback ' + json.dumps({"task_id": "read-001", "text": FEEDBACK,
            "activity_status": "completed", "understanding_status": "self_reported_unclear"}, ensure_ascii=False),
        DISCUSSION, CANDIDATE,
    ]
    return [{"stage": "chat", "text": text} for text in commands] + [
        {"stage": "confirm", "text": "n"}, {"stage": "chat", "text": CANDIDATE},
        {"stage": "confirm", "text": "y"}, {"stage": "chat", "text": "exit"}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--llm", action="store_true")
    args = p.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    for case in cases:
        case["planning_request"] = case["goal"] + "。请依据资料安排一天的学习，帮助我理解这两个概念的区别。活动采用阅读和口头说明，给出可检查的完成标准。"
    settings = get_settings()
    before = fingerprint()
    manifest = {"mode": "online" if args.llm else "preflight_only", "cases": cases,
        "events": flow_events(), "resume": RESUME, "source_sha256": before,
        "settings": {name: getattr(settings, name) for name in (
            "chat_model", "llm_temperature", "request_timeout", "llm_retry_attempts",
            "interactive_max_retrieval_calls", "interactive_max_no_progress_calls")}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.llm:
        print("预检完成：2条独立任务，未调用模型。")
        return 0
    failed = False
    with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
        for case in cases:
            started = perf_counter()
            try:
                row = run_case(case, output / case["case_id"], flow_events=flow_events(), resume_text=RESUME)
            except Exception as exc:
                row = {"case_id": case["case_id"], "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            row.update(source_unchanged=fingerprint() == before, elapsed_s=round(perf_counter() - started, 2))
            failed |= row["status"] != "completed" or not all(row.get("checks", {}).values()) or not row["source_unchanged"]
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
    # 非零表示流程/控制验收失败；即使零退出也不表示内容质量通过。
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
