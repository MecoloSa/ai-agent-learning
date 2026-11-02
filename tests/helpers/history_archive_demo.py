"""独立临时数据库演示：不调用模型、不写产品存档。"""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from uuid import uuid4

from ai_agent.persistence.history_archive import ConversationArchive


def main():
    root = Path(__file__).resolve().parents[2]
    parent = root / "tmp"
    parent.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="history-archive-demo-", dir=parent))
    archive = ConversationArchive(folder / "history.sqlite3")
    archive.initialize()
    session = archive.create_session()
    case = json.loads((root / "tests/fixtures/evaluation/memory_probe_case.json").read_text(encoding="utf-8"))
    turns = []
    for item in case["history"][:2]:
        turns.append(archive.append_turn(session, uuid4().hex, item["user"], item["assistant"]))
    last = turns[-1]
    repeated = archive.append_turn(session, last.turn_id, last.user_text, last.assistant_text)
    restored = archive.read_turn(session, 2)
    assert restored == last == repeated
    assert archive.read_turn(session, 3) is None
    print(json.dumps({
        "mode": "offline_storage_demo", "archive_db": str(archive.path),
        "repeated_submit_unchanged": repeated == last,
        "restored_turn": asdict(restored),
        "user_message_id": restored.message_id("user"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
