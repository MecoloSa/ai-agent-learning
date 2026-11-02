"""离线演示：上下文没有原话，通过真实历史工具回读；模型决策为固定替身。"""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

import ai_agent.main as app
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.history_session import ArchiveCapture
from ai_agent.persistence.memory_state import MemorySnapshot
from ai_agent.persistence.state_store import load_learning_state


def main():
    root = Path(__file__).resolve().parents[2]
    (root / "tmp").mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="history-readback-demo-", dir=root / "tmp"))
    archive_root = folder / "archives"
    fixture = json.loads((root / "tests/fixtures/evaluation/memory_probe_case.json").read_text(encoding="utf-8"))
    capture = ArchiveCapture(archive_root, None, True)
    for turn in fixture["history"][:2]:
        capture.capture(turn["user"], turn["assistant"])
    question = "请实际调用 read_conversation_turn，读取来源1第2轮，逐字引用用户更正时间的完整原话；不要修改计划、任务或反馈。"
    initial = LearningState(
        learning_goal="核对历史原话", plan_markdown="原计划，保持不变。",
        conversation_memory=MemorySnapshot(
            strategy="summary", max_chars=2000, window_size=6,
            summary="此前讨论过练习安排；这里没有保留具体条件和逐字原话。",
            history_archive=capture.state,
        ),
    )
    observation = {"mode": "offline_readback_demo", "user_input": question,
                   "initial_state": initial.model_dump(mode="json"), "model_requests": []}

    def fake_model(llm, messages, *, stage):
        observation["model_requests"].append({"stage": stage, "request_messages": [
            {"role": m.type, "content": m.content,
             "tool_calls": getattr(m, "tool_calls", None), "tool_call_id": getattr(m, "tool_call_id", None)}
            for m in messages
        ]})
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            return AIMessage(content="", tool_calls=[{
                "name": "read_conversation_turn", "args": {"source_index": 1, "turn_number": 2},
                "id": "history-read-demo", "type": "tool_call",
            }])
        result = json.loads(results[-1].content)
        assert result["ok"] is True
        data = result["data"]
        return AIMessage(content=f"> {data['user_text']}\n{data['citation']}\n以上是历史对话原话，未修改正式记录。")

    original_invoke = app._invoke_interactive_turn
    def observed_turn(**kwargs):
        audit = {}
        answer = original_invoke(**kwargs, audit_trace=audit)
        observation["audit_trace"] = audit
        return answer

    inputs = iter([question, "exit"])
    current = folder / "current.json"
    with ExitStack() as stack:
        stack.enter_context(patch("builtins.input", lambda _: next(inputs)))
        stack.enter_context(patch.object(app, "create_llm", lambda **kw: object()))
        stack.enter_context(patch.object(app, "create_llm_with_tools", lambda *a, **kw: object()))
        stack.enter_context(patch.object(app, "needs_rag", lambda _: False))
        stack.enter_context(patch.object(app, "invoke_llm_response", fake_model))
        stack.enter_context(patch.object(app, "_invoke_interactive_turn", observed_turn))
        app.run_interactive_session("", initial.plan_markdown, "", initial.learning_goal, None, str(folder),
            initial_state=initial, current_state_path=current, archive_history=None, history_archive_root=archive_root)
    saved = load_learning_state(current)
    quote = fixture["history"][1]["user"]
    first_request = observation["model_requests"][0]["request_messages"]
    assert quote not in str(first_request)
    assert quote in observation["audit_trace"]["final_answer"]
    assert saved.plan_markdown == initial.plan_markdown and not saved.tasks and not saved.feedback_records
    assert len(saved.conversation_memory.history_archive.sources) == 2
    observation["state_path"] = str(current)
    observation["business_state_unchanged"] = True
    output = folder / "observation.json"
    output.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "mode": observation["mode"], "observation_path": str(output), "state_path": str(current),
        "user_input": question, "original_quote_in_first_request": False,
        "business_state_unchanged": True, "tool_results": observation["audit_trace"]["tool_results"],
        "final_answer": observation["audit_trace"]["final_answer"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
