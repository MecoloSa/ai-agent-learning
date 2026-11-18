"""从原始结果派生可读统计；保留原始JSONL，不把调用成功当作质量通过。"""
import argparse
import json
from pathlib import Path


def summarize(rows):
    groups = []
    for case in dict.fromkeys(row["case_id"] for row in rows):
        for arm in ("raw", "compact"):
            selected = [row for row in rows if row["case_id"] == case and row["arm"] == arm]
            def total(field):
                values = [row["usage"].get(field) for row in selected]
                return sum(values) if values and all(type(v) is int for v in values) else None
            kinds = ["tool_call" if row.get("returned_tool_calls") else
                "answer" if isinstance(row.get("final_answer"), str) and row["final_answer"].strip()
                else "empty" for row in selected]
            groups.append({"case_id": case, "arm": arm, "responses": len(selected),
                "response_kinds": kinds, "input_tokens": total("input_tokens"),
                "output_tokens": total("output_tokens"), "total_tokens": total("total_tokens"),
                "latency_ms": [row["latency_ms"] for row in selected],
                "attempts": sum(len(row["attempts"]) for row in selected)})
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    groups = summarize(rows)
    (args.directory / "observations.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    transcript = []
    for row in rows:
        calls = row.get("returned_tool_calls", [])
        kind = "tool_call" if calls else "answer" if row.get("final_answer", "").strip() else "empty"
        transcript.append(f"## {row['case_id']} {row['arm']} repeat={row['repeat']}\n\n"
            f"问题：{row['user_input']}\n\n响应类别：{kind}\n\n"
            + (row.get("final_answer") or "（没有回答正文）")
            + "\n\n工具请求：\n```json\n" + json.dumps(calls, ensure_ascii=False, indent=2) + "\n```\n")
    (args.directory / "observed-transcript.md").write_text("\n".join(transcript), encoding="utf-8")
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
