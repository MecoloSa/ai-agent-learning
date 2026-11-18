"""生产循环实验统计，不自动判定语义质量。"""
import argparse
import json
from pathlib import Path


def summarize(rows):
    summary = []
    for row in rows:
        summary.append({"case_id": row["case_id"], "arm": row["arm"], "repeat": row["repeat"],
            "status": row["status"], "online_generations": len(row["model_responses"]),
            "actual_requests": len(row["attempts"]), "tool_attempts": len(row["tool_executions"]),
            "usage": row["usage"], "latency_ms": row["latency_ms"],
            "source_unchanged": row["source_unchanged"], "materials_unchanged": row["materials_unchanged"]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    result = summarize(rows)
    (args.directory / "observations.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
