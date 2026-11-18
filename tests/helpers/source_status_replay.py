"""Compare proposed source status labels on saved real tool trajectories."""

import json
from pathlib import Path


def classify(audit, target):
    evidence = set()
    for call in audit["tool_results"]:
        if not call["executed"]:
            continue
        kind = call["tool_name"]
        result = json.loads(call["result"])
        data = result.get("data", {})
        if result.get("ok") is False and call["args"].get("relative_path") == target:
            evidence.add("read_failed")
        if result.get("ok") is not True:
            continue
        if kind == "list_material_files" and data.get("offset") == 0 and data.get("has_more") is False:
            names = {file["relative_path"] for file in data["files"]}
            if target not in names:
                evidence.add("not_listed")
        if data.get("source") == target:
            if kind == "inspect_document_structure":
                evidence.add("text" if data.get("mode") == "whole_document" and data.get("text", "").strip() else "location")
            elif kind == "read_material_lines" and any(line["text"].strip() for line in data["lines"]):
                evidence.add("text")
        if kind == "retrieve_learning_context" and data.get("text", "").strip():
            prefix = f"[source={target}, "
            if any(row["citation"].startswith(prefix) for row in data.get("sources", [])):
                evidence.add("text")
    # 失败后成功取得正文，应显示恢复后的事实。
    return next((status for status in ("text", "not_listed", "read_failed", "location") if status in evidence), "unverified")


def main():
    root = Path(".evaluation-results")
    cases = [
        ("越界后全文恢复", "line-feedback-live/recovery", "从重读到回想.md", "text"),
        ("按行读取失败", "line-feedback-live/missing", "从重读到回想.md", "read_failed"),
        ("完整目录未列出", "line-feedback-live/missing", "下次组会前的复习.txt", "not_listed"),
        ("仅见一级目录", "line-feedback-live/long", "研习台学习指南.md", "location"),
        ("检索返回正文", "learning-coverage-next/long-feedback", "研习台学习指南.md", "text"),
        ("目录找到但没读", "learning-coverage-next/both-3", "从重读到回想.md", "unverified"),
    ]
    for label, folder, name, expected in cases:
        audit = json.loads((root / folder / "result.json").read_text(encoding="utf8"))["audits"][0]
        actual = classify(audit, name)
        print(f"{label}: {actual}")
        assert actual == expected


if __name__ == "__main__":
    main()
