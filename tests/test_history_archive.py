"""原文归档候选：真实SQLite与独立进程，不调用模型、不修改学习存档。"""
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import sqlite3
import subprocess
import sys
from uuid import uuid4

import pytest


@pytest.fixture
def api():
    return importlib.import_module("ai_agent.persistence.history_archive")


@pytest.fixture
def archive(api, tmp_path):
    store = api.ConversationArchive(tmp_path / "history.sqlite3")
    store.initialize()
    return store


def test_exact_text_ids_and_read_only_lookup(api, archive):
    session = archive.create_session()
    user = "  更正：今天最多35分钟，其他条件保持。\n"
    assistant = "了解，不修改正式记录。\n "
    turn = archive.append_turn(session, uuid4().hex, user, assistant)
    before = archive.path.read_bytes()
    restored = archive.read_turn(session, 1)
    assert restored == turn
    assert restored.user_text == user and restored.assistant_text == assistant
    assert restored.message_id("user") != restored.message_id("assistant")
    assert restored.message_id("user") == f"{turn.turn_id}:user"
    assert restored.recorded_at.endswith("+00:00")
    assert archive.path.read_bytes() == before


def test_repeated_submit_is_idempotent_without_consuming_sequence(archive):
    session, turn_id = archive.create_session(), uuid4().hex
    first = archive.append_turn(session, turn_id, "问题1", "回答1")
    assert archive.append_turn(session, turn_id, "问题1", "回答1") == first
    assert archive.append_turn(session, uuid4().hex, "问题2", "回答2").turn_number == 2
    assert archive.read_turn(session, 3) is None


@pytest.mark.parametrize("changed", ["session", "user", "assistant"])
def test_reused_id_cannot_change_source_or_content(api, archive, changed):
    session = archive.create_session()
    first = archive.append_turn(session, uuid4().hex, "问题", "回答")
    args = [session, first.turn_id, "问题", "回答"]
    if changed == "session":
        args[0] = archive.create_session()
    else:
        args[2 if changed == "user" else 3] = "修改后的文本"
    with pytest.raises(api.HistoryArchiveError):
        archive.append_turn(*args)
    assert archive.read_turn(session, 1) == first


def test_sessions_are_not_mixed_by_turn_number(archive):
    first, second = archive.create_session(), archive.create_session()
    a = archive.append_turn(first, uuid4().hex, "会话A问题", "A回答")
    assert archive.read_turn(second, 1) is None
    b = archive.append_turn(second, uuid4().hex, "会话B问题", "B回答")
    assert a.turn_number == b.turn_number == 1
    assert archive.read_turn(first, 1) == a
    assert archive.read_turn(second, 1) == b


def test_budget_rejects_instead_of_silently_truncating(api, archive):
    session = archive.create_session()
    first = archive.append_turn(session, uuid4().hex, "用户原话", "助手回复")
    with pytest.raises(api.HistoryReadLimitError):
        archive.read_turn(session, 1, max_chars=7)
    assert archive.read_turn(session, 1, max_chars=8) == first


@pytest.mark.parametrize("value", [0, True, 20001])
def test_invalid_read_budget_rejected(api, archive, value):
    with pytest.raises(api.HistoryReadLimitError):
        archive.read_turn(archive.create_session(), 1, max_chars=value)


@pytest.mark.parametrize("number", [0, -1, True, 1.2])
def test_invalid_turn_number_rejected(api, archive, number):
    with pytest.raises(api.HistoryArchiveError):
        archive.read_turn(archive.create_session(), number)


@pytest.mark.parametrize("identifier", ["bad", "../escape", "a" * 31, None])
def test_invalid_identifier_rejected(api, archive, identifier):
    with pytest.raises(api.HistoryArchiveError):
        archive.read_turn(identifier, 1)


@pytest.mark.parametrize("user,assistant", [(None, "回答"), ("", "回答"), ("问题", " \n ")])
def test_invalid_text_does_not_create_half_turn(api, archive, user, assistant):
    session = archive.create_session()
    with pytest.raises(api.HistoryArchiveError):
        archive.append_turn(session, uuid4().hex, user, assistant)
    assert archive.read_turn(session, 1) is None


def test_unknown_session_is_not_created_implicitly(api, archive):
    with pytest.raises(api.HistoryArchiveError):
        archive.append_turn(uuid4().hex, uuid4().hex, "问题", "回答")


def test_read_missing_database_does_not_create_files(api, tmp_path):
    path = tmp_path / "missing" / "history.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        api.ConversationArchive(path).read_turn(uuid4().hex, 1)
    assert not path.parent.exists()


def test_foreign_database_is_not_adopted_or_changed(api, tmp_path):
    path = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated(value TEXT)")
    before = path.read_bytes()
    with pytest.raises(api.HistoryArchiveError):
        api.ConversationArchive(path).initialize()
    assert path.read_bytes() == before


def test_unknown_schema_is_rejected(api, archive):
    with sqlite3.connect(archive.path) as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(api.HistoryArchiveError):
        archive.initialize()
    with pytest.raises(api.HistoryArchiveError):
        archive.read_turn(uuid4().hex, 1)


def test_initialize_again_preserves_archive(archive):
    session = archive.create_session()
    turn = archive.append_turn(session, uuid4().hex, "问题", "回答")
    before = archive.path.read_bytes()
    archive.initialize()
    assert archive.read_turn(session, 1) == turn
    assert archive.path.read_bytes() == before


def test_failed_insert_leaves_no_partial_turn_or_sequence_hole(archive):
    session = archive.create_session()
    with sqlite3.connect(archive.path) as connection:
        connection.execute("CREATE TRIGGER reject_turn BEFORE INSERT ON history_turns "
                           "BEGIN SELECT RAISE(ABORT, 'simulated failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        archive.append_turn(session, uuid4().hex, "问题", "回答")
    assert archive.read_turn(session, 1) is None
    with sqlite3.connect(archive.path) as connection:
        connection.execute("DROP TRIGGER reject_turn")
    assert archive.append_turn(session, uuid4().hex, "问题", "回答").turn_number == 1


def test_concurrent_writers_get_unique_order(archive):
    session = archive.create_session()

    def append(index):
        return archive.append_turn(session, uuid4().hex, f"问题{index}", f"回答{index}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        turns = list(pool.map(append, range(8)))
    assert sorted(t.turn_number for t in turns) == list(range(1, 9))
    assert len({t.turn_id for t in turns}) == 8
    assert all(archive.read_turn(session, t.turn_number) == t for t in turns)


def test_new_process_reads_same_original_text(api, archive):
    session = archive.create_session()
    user = "更正：今天最多35分钟，其他条件保持。这里只更正对话条件，不修改任何计划或记录。"
    turn = archive.append_turn(session, uuid4().hex, user, "了解，只更正对话条件。")
    worker = """
import importlib.util, json, sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('archive_worker',sys.argv[1])
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)
t=m.ConversationArchive(Path(sys.argv[2])).read_turn(sys.argv[3],1)
print(json.dumps({'user':t.user_text,'id':t.message_id('user')},ensure_ascii=False))
"""
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", worker,
                             api.__file__, str(archive.path), session],
                            capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data == {"user": user, "id": turn.message_id("user")}
