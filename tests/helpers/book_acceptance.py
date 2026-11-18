"""真实教材多方向验收；复用现有独立进程交互观察。"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import ai_agent.runtime.agents as agents
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.session_materials import create_material_binding
from ai_agent.persistence.state_store import save_learning_state, load_learning_state
from tests.helpers.learning_closed_loop import launch, events, RESUME, formal
from ai_agent.evaluation.acquisition_pair import message_record

BOOKS = Path("C:/Users/21363/Desktop/ai-agent-book")
QUESTIONS = [
    ("exact_page", "请实际读取 AI-Agents-in-Depth-zh-CN.pdf 的物理第18页，解释通用工具和专用工具分别适合什么任务，给出该页提出的安全边界。只依据这一页，不修改记录。"),
    ("cross_book", "请结合两本教材，比较 Function Calling 的工具调用与 Hello-Agents 的 Thought-Action-Observation 循环：共同的数据流是什么？协议表达有什么不同？请分别核对两本书的原文，不修改记录。"),
    ("insufficient", "仅依据 AI-Agents-in-Depth-zh-CN.pdf 物理第18页，告诉我完成本页全部学习目标精确需要多少分钟，必须给出书中规定的数值；不要查其他页，不修改记录。"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--chapter-mode", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    questions = QUESTIONS
    if args.chapter_mode:
        questions = [("cross_book_pages", "请实际读取 AI-Agents-in-Depth-zh-CN.pdf 物理第18页和 Hello-Agents-V1.0.3-20260717.pdf 物理第17—18页，比较 Function Calling 与 Thought-Action-Observation 的共同数据流和协议表达差异。仅依据这些页面，不修改记录。"), QUESTIONS[2]]
    (args.output_dir / "manifest.json").write_text(json.dumps({"books": str(BOOKS), "questions": questions,
        "chapter_mode": args.chapter_mode, "events": events(), "resume": RESUME}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.llm:
        return
    rows = []

    def record(row):
        rows.append(row)
        (args.output_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(row, flush=True)

    for name, question in questions:
        directory = args.output_dir / name
        directory.mkdir()
        state = LearningState(learning_goal="理解工具调用与安全边界", plan_markdown="阅读教材并口头说明，暂不修改安排。",
                              material_binding=create_material_binding(str(BOOKS)))
        save_learning_state(state, directory / "source.json")
        code = launch({"source": str(directory / "source.json"), "current": str(directory / "current.json"),
            "archives": str(directory / "archives"), "observation": str(directory / "observation.json"),
            "events": [{"stage": "chat", "text": question}, {"stage": "chat", "text": "exit"}]}, directory, "probe")
        final = load_learning_state(directory / "current.json") if (directory / "current.json").exists() else state
        record({"case_id": name, "code": code, "state_unchanged": formal(state) == formal(final), "artifacts": str(directory)})

    directory = args.output_dir / "closed_loop"
    directory.mkdir()
    models = []
    original = agents._invoke_llm_raw

    def observe(llm, messages, *, stage):
        response = original(llm, messages, stage=stage)
        models.append({"stage": stage, "request_messages": [message_record(m) for m in messages],
            "content": response.content, "tool_calls": response.tool_calls, "usage": response.usage_metadata})
        return response

    analysis_root = BOOKS
    if args.chapter_mode:
        from ai_agent.resources.pdf_materials import read_pdf_pages
        analysis_root = directory / "chapter-materials"
        analysis_root.mkdir()
        extracts = [read_pdf_pages(str(BOOKS), "AI-Agents-in-Depth-zh-CN.pdf", 18, 18),
                    read_pdf_pages(str(BOOKS), "Hello-Agents-V1.0.3-20260717.pdf", 17, 18)]
        for i, extract in enumerate(extracts):
            text = "\n\n".join(page["citation"] + "\n" + page["text"] for page in extract["pages"])
            (analysis_root / f"chapter-{i}.md").write_text(text, encoding="utf-8")
        (directory / "extracts.json").write_text(json.dumps(extracts, ensure_ascii=False, indent=2), encoding="utf-8")
        from ai_agent.retrieval.rag import needs_rag
        if needs_rag(str(analysis_root)):
            raise RuntimeError("章节替代实验仍触发RAG，停止而非绕过真实路由")
    with patch.object(agents, "_invoke_llm_raw", observe):
        analysis, plan, context = agents.run_learning_assistant(str(analysis_root),
            "依据两本教材，制定一天的学习计划，目标是理解工具调用的数据流与工具安全边界。只阅读和口头说明，给出可核对的完成标准。", 1, "agent")
    (directory / "initial-workflow.json").write_text(json.dumps({"analysis": analysis, "plan": plan,
        "context": context, "models": models}, ensure_ascii=False, indent=2), encoding="utf-8")
    state = LearningState(learning_goal="理解工具调用的数据流与工具安全边界", plan_markdown=plan,
                          material_binding=create_material_binding(str(BOOKS)))
    save_learning_state(state, directory / "source.json")
    common = {"archives": str(directory / "archives"), "analysis": analysis, "context": context}
    code = launch({**common, "source": str(directory / "source.json"), "current": str(directory / "first.json"),
        "observation": str(directory / "first-observation.json"), "events": events()}, directory, "first")
    row = {"case_id": "closed_loop", "code": code, "artifacts": str(directory)}
    if code == 0:
        before = load_learning_state(directory / "first.json")
        original_bytes = (directory / "first.json").read_bytes()
        rc = launch({**common, "source": str(directory / "first.json"), "current": str(directory / "resumed.json"),
            "observation": str(directory / "resumed-observation.json"), "events": [
                {"stage": "chat", "text": RESUME}, {"stage": "chat", "text": "exit"}]}, directory, "resume")
        row["resume_code"] = rc
        if rc == 0:
            after = load_learning_state(directory / "resumed.json")
            reviews = json.loads((directory / "first-observation.json").read_text(encoding="utf-8"))["plan_reviews"]
            row["checks"] = {"two_reviews": len(reviews) == 2,
                "decline_unchanged": bool(reviews) and reviews[0]["decision"] == "cancelled" and reviews[0]["before"] == reviews[0]["after"],
                "adopt_once": before.context_revision == 2,
                "feedback_preserved": len(before.feedback_records) == 1 and before.tasks[0].activity_status == "completed" and before.tasks[0].understanding_status == "self_reported_unclear",
                "resume_unchanged": formal(before) == formal(after),
                "source_unmodified": original_bytes == (directory / "first.json").read_bytes()}
    record(row)


if __name__ == "__main__":
    main()
