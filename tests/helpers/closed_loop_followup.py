"""保留非技术失败，从真实快照进行明确用户澄清及新进程恢复补验。"""
import argparse
import json
from pathlib import Path

from tests.helpers.learning_closed_loop import launch, formal, RESUME, DISCUSSION
from ai_agent.persistence.state_store import load_learning_state

CLARIFY = "我之前拒绝了上个候选。现在请重新提交一个候选计划进入确认流程，总计最多20分钟，只阅读和口头说明，目标和记录不变。"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_dir, output = args.source_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = source_dir / "first.json"
    common = {"archives": str(source_dir / "archives")}
    adopted = output / "adopted.json"
    code = launch({**common, "source": str(source), "current": str(adopted),
        "events": [{"stage": "chat", "text": CLARIFY}, {"stage": "confirm", "text": "y"},
                   {"stage": "chat", "text": "exit"}],
        "observation": str(output / "adopt-observation.json")}, output, "adopt")
    result = {"initial_failure_preserved": True, "clarification": CLARIFY, "adopt_process_code": code}
    if not code:
        after = load_learning_state(adopted)
        before = load_learning_state(source)
        resumed = output / "resumed.json"
        code = launch({**common, "source": str(adopted), "current": str(resumed),
            "events": [{"stage": "chat", "text": RESUME}, {"stage": "chat", "text": "exit"}],
            "observation": str(output / "resumed-observation.json")}, output, "resumed")
        result["resume_process_code"] = code
        if not code:
            final = load_learning_state(resumed)
            obs = json.loads((output / "resumed-observation.json").read_text(encoding="utf-8"))
            result["checks"] = {"revision_once": after.context_revision == before.context_revision + 1,
                "feedback_preserved": after.tasks == before.tasks and after.feedback_records == before.feedback_records,
                "restore_unchanged": formal(after) == formal(final),
                "quote_in_answer": DISCUSSION in obs["turns"][0]["response"]}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    main()
