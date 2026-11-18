"""Matched main-agent runs with and without the structure tool."""

import argparse
import json
from pathlib import Path
import time
from langchain_core.messages import SystemMessage

import ai_agent.main as app
from ai_agent.learning.learning_context import build_learning_state_message
from ai_agent.learning.learning_state import LearningState
from ai_agent.resources.document_structure import build_inspect_document_structure_tool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--case-id", choices=["short", "long", "focused"])
    args = parser.parse_args()
    questions = [
        ("short", "请按文章顺序列出 short.md 的全部主要章节，并说明每章的关键事实。"),
        ("long", "请完整概览 long.md，按主要章节顺序介绍内容，包括末章的结论。"),
        ("focused", "short.md 里试验是哪天开始的？只回答日期和依据。"),
    ]
    if args.case_id:
        questions = [item for item in questions if item[0] == args.case_id]
    print(json.dumps({"cases": questions, "variants": ["baseline", "structure"],
                      "live": args.llm}, ensure_ascii=False))
    if not args.llm:
        return
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    materials = root / "materials"
    materials.mkdir()
    (materials / "short.md").write_text(
        "# 试验记录\n## 起点\n试验于2026年3月12日开始。\n"
        "## 方法\n每组使用七个样本。\n## 结论\n最终编号是BLUE-3617。\n", encoding="utf-8")
    chapters = [
        ("第一章 起点", "试验于2026年3月12日开始。"),
        ("第二章 方法", "每组使用七个样本。"),
        ("第三章 例外", "雨天必须暂停记录。"),
        ("第四章 结论", "最终编号是BLUE-3617。"),
    ]
    text = "# 试验长文\n"
    for title, fact in chapters:
        text += f"## {title}\n{fact}\n"
        text += "### 阅读练习\n" + ("本段提供背景解释，请结合本章事实理解研究安排。\n" * 170)
    (materials / "long.md").write_text(text, encoding="utf-8")
    rows = []
    for index, (case_id, question) in enumerate(questions):
        variants = ["baseline", "structure"] if index % 2 == 0 else ["structure", "baseline"]
        for variant in variants:
            state = LearningState(learning_goal="理解试验文章", plan_markdown="阅读并说明文章内容")
            prompt = app._build_interactive_system_prompt(
                "", state.plan_markdown, "", state.learning_goal,
                retrieval_available=False, on_demand_eligible=True, plan_only=True,
            )
            read = app.build_read_material_lines_tool(str(materials))
            extra = [app.build_list_material_files_tool(str(materials)), app.build_read_pdf_pages_tool(str(materials))]
            if variant == "structure":
                extra.append(build_inspect_document_structure_tool(str(materials)))
            audit = {}
            started = time.perf_counter()
            try:
                answer = app._invoke_interactive_turn(
                    request_messages=[SystemMessage(content=prompt), build_learning_state_message(state)],
                    user_input=question, plain_llm=app.create_llm(),
                    tool_llm=app.create_llm_with_tools([read, *extra]),
                    retrieval_tool=None, read_tool=read, extra_tools=tuple(extra), audit_trace=audit,
                )
                row = {"case_id": case_id, "variant": variant, "question": question,
                       "answer": answer, "error": None}
            except Exception as exc:
                row = {"case_id": case_id, "variant": variant, "question": question,
                       "answer": None, "error": f"{type(exc).__name__}: {exc}",
                       "failed_invocation_audit": getattr(exc, "invocation_audit", None)}
            row.update(elapsed_s=round(time.perf_counter() - started, 3), audit=audit)
            rows.append(row)
            (root / f"{case_id}-{variant}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"case_id": case_id, "variant": variant, "elapsed_s": row["elapsed_s"],
                              "error": row["error"]}, ensure_ascii=False), flush=True)
    (root / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
