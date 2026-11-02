"""真实会话保存接线演示；模型和摘要均为替身，不请求网络。"""
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import ai_agent.main as app
import ai_agent.persistence.memory as memory_module
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.history_archive import ConversationArchive
from ai_agent.persistence.memory_state import MemorySnapshot
from ai_agent.persistence.state_store import load_learning_state


def main():
    root = Path(__file__).resolve().parents[2]
    (root / "tmp").mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="history-session-demo-", dir=root / "tmp"))
    materials = folder / "materials"
    materials.mkdir()
    (materials / "notes.md").write_text("本文件仅用于离线接线演示。", encoding="utf-8")
    case = json.loads((root / "tests/fixtures/evaluation/memory_probe_case.json").read_text(encoding="utf-8"))
    turns = case["history"][:2]
    inputs = iter([turns[0]["user"], turns[1]["user"], "exit"])
    replies = iter([turns[0]["assistant"], turns[1]["assistant"]])
    current = folder / "current.json"
    initial = LearningState(
        learning_goal="离线原文归档演示", plan_markdown="仅测试保存链路，不登记任务。",
        conversation_memory=MemorySnapshot(strategy="summary", max_chars=1, window_size=6),
    )
    with ExitStack() as stack:
        stack.enter_context(patch("builtins.input", lambda _: next(inputs)))
        stack.enter_context(patch.object(app, "create_llm", lambda **kw: object()))
        stack.enter_context(patch.object(app, "create_llm_with_tools", lambda *a, **kw: object()))
        stack.enter_context(patch.object(app, "needs_rag", lambda _: False))
        stack.enter_context(patch.object(app, "_invoke_interactive_turn", lambda **kw: next(replies)))
        stack.enter_context(patch.object(memory_module, "_request_summary", lambda _: SimpleNamespace(
            content="已讨论松针练习的安排；正式状态不变。")))
        app.run_interactive_session(
            "", initial.plan_markdown, "", initial.learning_goal, None, str(materials),
            initial_state=initial, current_state_path=current,
            archive_history=True, history_archive_root=folder / "archives",
        )
    saved = load_learning_state(current)
    source = saved.conversation_memory.history_archive.sources[0]
    archive = ConversationArchive(folder / "archives" / f"{source.archive_id}.sqlite3")
    first, second = archive.read_turn(source.session_id, 1), archive.read_turn(source.session_id, 2)
    assert first.user_text == turns[0]["user"] and second.user_text == turns[1]["user"]
    assert len(saved.conversation_memory.messages) == 2
    print(json.dumps({
        "mode": "offline_session_demo", "state_path": str(current),
        "schema_version": json.loads(current.read_text(encoding="utf-8"))["schema_version"],
        "retained_memory_messages": len(saved.conversation_memory.messages),
        "archive_source": source.model_dump(mode="json"),
        "archive_turn_1": asdict(first), "archive_turn_2": asdict(second),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
