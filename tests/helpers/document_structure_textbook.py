"""Check structure and page numbering against the two local textbooks."""

import argparse
import json
from pathlib import Path
import time

from pypdf import PdfReader

from ai_agent.config import create_llm
from ai_agent.resources.document_structure import build_inspect_document_structure_tool, inspect_document_structure
from ai_agent.resources.pdf_materials import read_pdf_pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--book", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--question")
    args = parser.parse_args()
    with (args.materials / args.book).open("rb") as stream:
        reader = PdfReader(stream, strict=False)
        labels = reader.page_labels
        outline = [
            {"title": item.title, "physical_page": index + 1, "pdf_label": labels[index]}
            for item in reader.outline if not isinstance(item, list)
            if (index := reader.get_destination_page_number(item)) is not None and index >= 0
        ]
        total = len(reader.pages)
    # Sample the opening pages and a chapter beyond the default scan window.
    picks = [item for item in outline if item["physical_page"] >= 9]
    samples = [*picks[:3], *[item for item in picks if item["physical_page"] > 160][:2]]
    for sample in samples:
        page = sample["physical_page"]
        row = read_pdf_pages(str(args.materials), args.book, page, page)["pages"][0]
        sample.update(text_preview=row["text"][:350], text_tail=row["text"][-150:],
                      citation=row["citation"], text_status=row["text_status"])
    result = {"book": args.book, "total_pages": total, "outline": outline, "samples": samples}
    if args.question:
        result["question"] = args.question
        started = time.perf_counter()
        audit = {}
        try:
            import ai_agent.main as app
            from langchain_core.messages import SystemMessage
            from ai_agent.learning.learning_context import build_learning_state_message
            from ai_agent.learning.learning_state import LearningState

            root = str(args.materials.resolve())
            state = LearningState(learning_goal="核对教材章节与页码", plan_markdown="阅读并核对原文")
            prompt = app._build_interactive_system_prompt(
                "", state.plan_markdown, "", state.learning_goal,
                retrieval_available=False, on_demand_eligible=True, plan_only=True,
            )
            read = app.build_read_material_lines_tool(root)
            extra = [app.build_list_material_files_tool(root), app.build_read_pdf_pages_tool(root),
                     build_inspect_document_structure_tool(root)]
            result["answer"] = app._invoke_interactive_turn(
                request_messages=[SystemMessage(content=prompt), build_learning_state_message(state)],
                user_input=args.question, plain_llm=app.create_llm(),
                tool_llm=app.create_llm_with_tools([read, *extra]),
                retrieval_tool=None, read_tool=read, extra_tools=tuple(extra), audit_trace=audit,
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result.update(elapsed_s=round(time.perf_counter() - started, 3), audit=audit)
    elif args.llm:
        started = time.perf_counter()
        try:
            result["structure"] = inspect_document_structure(
                str(args.materials), args.book, model_factory=lambda: create_llm(role="document_structure"),
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_s"] = round(time.perf_counter() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({
        "book": args.book, "total_pages": total, "outline_count": len(outline),
        "model_sections": len(result.get("structure", {}).get("sections", [])),
        "elapsed_s": result.get("elapsed_s"), "error": result.get("error"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
