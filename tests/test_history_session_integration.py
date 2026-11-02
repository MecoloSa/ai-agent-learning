"""真实会话归档与待写恢复；固定模型回答，只检查程序链路。"""
import builtins
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

import ai_agent.main as app
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.history_archive import ConversationArchive
from ai_agent.persistence.memory_state import MemorySnapshot
from ai_agent.persistence.state_store import load_learning_state, save_learning_state


@pytest.fixture
def api():
    return importlib.import_module("ai_agent.persistence.history_session")


@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "create_llm", lambda **kw: object())
    monkeypatch.setattr(app, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(app, "needs_rag", lambda _: False)
    monkeypatch.setattr(app, "_invoke_interactive_turn", lambda **kw: "显示给用户的回答")
    return tmp_path / "archives"


def run(monkeypatch, tmp_path, root, inputs, *, initial=None, enabled=True, filename="current.json"):
    iterator = iter(inputs)
    monkeypatch.setattr(builtins, "input", lambda _: next(iterator))
    initial = initial or LearningState(learning_goal="原目标", plan_markdown="原计划")
    path = tmp_path / filename
    app.run_interactive_session("", initial.plan_markdown, "", initial.learning_goal,
                                None, str(tmp_path), initial_state=initial,
                                current_state_path=path, archive_history=enabled,
                                history_archive_root=root)
    return load_learning_state(path)


def read_source(root, source, number):
    return ConversationArchive(root / f"{source.archive_id}.sqlite3").read_turn(source.session_id, number)


def test_real_session_preserves_raw_input_before_strip(monkeypatch, tmp_path, local):
    raw = "  请解释区别。  "
    saved = run(monkeypatch, tmp_path, local, [raw, "exit"])
    state = saved.conversation_memory.history_archive
    assert state.enabled and state.pending is None and len(state.sources) == 1
    turn = read_source(local, state.sources[0], 1)
    assert turn.user_text == raw and turn.assistant_text == "显示给用户的回答"
    assert saved.conversation_memory.messages[0].content == raw.strip()
    assert saved.context_revision == 1


def test_default_stays_disabled_and_state_commands_are_not_chat_turns(monkeypatch, tmp_path, local):
    saved = run(monkeypatch, tmp_path, local, ["/state", "只讨论", "exit"], enabled=None)
    assert saved.conversation_memory.history_archive is None
    assert not local.exists()


@pytest.mark.parametrize("decision", ["y", "n"])
def test_displayed_candidate_is_separate_from_program_confirmation(monkeypatch, tmp_path, local, decision):
    reply = "===NEW_PLAN_START===\n阅读10分钟。\n===NEW_PLAN_END==="
    monkeypatch.setattr(app, "_invoke_interactive_turn", lambda **kw: reply)
    saved = run(monkeypatch, tmp_path, local, ["请给候选", decision, "exit"])
    turn = read_source(local, saved.conversation_memory.history_archive.sources[0], 1)
    assert turn.assistant_text == reply
    memory_reply = saved.conversation_memory.messages[-1].content
    assert ("confirmed" if decision == "y" else "cancelled") in memory_reply
    assert saved.plan_markdown == ("阅读10分钟。" if decision == "y" else "原计划")


def test_archive_precedes_summary_and_survives_compaction(monkeypatch, tmp_path, local):
    import ai_agent.persistence.memory as memory
    monkeypatch.setattr(memory, "_request_summary", lambda _: SimpleNamespace(content="压缩后的背景"))
    initial = LearningState(learning_goal="原目标", plan_markdown="原计划",
                           conversation_memory=MemorySnapshot(strategy="summary", max_chars=1, window_size=6))
    saved = run(monkeypatch, tmp_path, local, ["第一轮原文", "第二轮原文", "exit"], initial=initial)
    assert len(saved.conversation_memory.messages) == 2
    source = saved.conversation_memory.history_archive.sources[0]
    assert source.through_turn == 2
    assert read_source(local, source, 1).user_text == "第一轮原文"


@pytest.mark.parametrize("committed", [False, True])
def test_failed_archive_stops_before_confirmation_and_pending_recovers(
    api, monkeypatch, tmp_path, local, committed,
):
    original = ConversationArchive.append_turn

    def fail(self, *args):
        if committed:
            original(self, *args)  # 模拟数据库已提交，但调用方未收到成功回执。
        raise OSError("PRIVATE_STORAGE_DETAIL")

    reply = "===NEW_PLAN_START===\n阅读10分钟。\n===NEW_PLAN_END==="
    monkeypatch.setattr(app, "_invoke_interactive_turn", lambda **kw: reply)
    monkeypatch.setattr(ConversationArchive, "append_turn", fail)
    monkeypatch.setattr(app, "_review_context_change", lambda *a, **kw: pytest.fail("归档失败不能继续确认"))
    with pytest.raises(api.HistoryPersistenceError, match="暂存"):
        run(monkeypatch, tmp_path, local, ["  请给候选  ", "y"])
    source = tmp_path / "current.json"
    saved = load_learning_state(source)
    pending = saved.conversation_memory.history_archive.pending
    assert pending.user_text == "  请给候选  " and pending.assistant_text == reply
    assert saved.plan_markdown == "原计划" and saved.context_revision == 1
    assert len(saved.conversation_memory.messages) == 2
    before = source.read_bytes()
    monkeypatch.setattr(ConversationArchive, "append_turn", original)
    monkeypatch.setattr(app, "_invoke_interactive_turn", lambda **kw: "只解释，不产生候选")
    resumed = run(monkeypatch, tmp_path, local, ["继续解释", "exit"], initial=saved,
                  enabled=None, filename="resumed.json")
    state = resumed.conversation_memory.history_archive
    assert state.pending is None and len(state.sources) == 2
    old = next(s for s in state.sources if s.archive_id == pending.archive_id)
    assert old.through_turn == 1
    assert read_source(local, old, 1).turn_id == pending.turn_id
    assert read_source(local, old, 2) is None
    assert source.read_bytes() == before


def test_pending_cannot_be_silently_disabled(api, monkeypatch, tmp_path, local):
    from ai_agent.persistence.memory_state import HistoryArchiveState, PendingArchiveTurn
    pending = PendingArchiveTurn(archive_id=uuid4().hex, session_id=uuid4().hex,
                                 turn_id=uuid4().hex, user_text="用户原话", assistant_text="显示回复")
    initial = LearningState(learning_goal="原目标", plan_markdown="原计划", conversation_memory=MemorySnapshot(
        strategy="window", max_chars=2000, window_size=6,
        history_archive=HistoryArchiveState(pending=pending)))
    with pytest.raises(api.HistoryPersistenceError, match="待归档"):
        run(monkeypatch, tmp_path, local, ["exit"], initial=initial, enabled=False)


def test_snapshot_watermark_does_not_expand_when_origin_continues(monkeypatch, tmp_path, local):
    import ai_agent.learning.learning_commands as commands
    history = tmp_path / "manual.json"
    monkeypatch.setattr(commands, "save_state_snapshot", lambda state: save_learning_state(state, history))
    saved = run(monkeypatch, tmp_path, local, ["第一轮", "/save-state", "第二轮", "exit"])
    snapshot = load_learning_state(history)
    old = snapshot.conversation_memory.history_archive.sources[0]
    assert old.through_turn == 1
    assert saved.conversation_memory.history_archive.sources[0].through_turn == 2
    db = local / f"{old.archive_id}.sqlite3"
    before_db, before_snapshot = db.read_bytes(), history.read_bytes()
    resumed = run(monkeypatch, tmp_path, local, ["分支问题", "exit"], initial=snapshot,
                  enabled=None, filename="branch.json")
    refs = resumed.conversation_memory.history_archive.sources
    assert len(refs) == 2
    inherited = next(s for s in refs if s.archive_id == old.archive_id)
    assert inherited.through_turn == 1
    assert db.read_bytes() == before_db and history.read_bytes() == before_snapshot


def test_disabling_is_persisted_without_losing_old_sources(monkeypatch, tmp_path, local):
    saved = run(monkeypatch, tmp_path, local, ["第一轮", "exit"])
    disabled = run(monkeypatch, tmp_path, local, ["第二轮不归档", "exit"], initial=saved,
                   enabled=False, filename="disabled.json")
    state = disabled.conversation_memory.history_archive
    assert state.enabled is False and state.sources == saved.conversation_memory.history_archive.sources
    assert len(list(local.glob("*.sqlite3"))) == 1


@pytest.mark.parametrize("version", range(1, 8))
def test_legacy_snapshots_have_no_fabricated_archive_and_new_writes_are_v8(tmp_path, version):
    old = tmp_path / "old.json"
    state = LearningState(learning_goal="旧目标", plan_markdown="旧计划").model_dump(mode="json")
    state["conversation_memory"] = {"strategy": "window", "max_chars": 2000, "window_size": 6}
    old.write_text(json.dumps({"schema_version": f"learning-state-v{version}", "state": state}), encoding="utf-8")
    before = old.read_bytes()
    restored = load_learning_state(old)
    assert restored.conversation_memory.history_archive is None
    path = save_learning_state(restored, tmp_path / "new.json")
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "learning-state-v8"
    assert old.read_bytes() == before


def test_double_storage_failure_is_not_reported_as_saved(api, monkeypatch, tmp_path, local, capsys):
    def fail(*args, **kwargs):
        raise OSError("PRIVATE_STORAGE_DETAIL")

    monkeypatch.setattr(ConversationArchive, "append_turn", fail)
    monkeypatch.setattr(app, "save_current_learning_state", fail)
    with pytest.raises(api.HistoryPersistenceError, match="均保存失败"):
        run(monkeypatch, tmp_path, local, ["问题", "exit"])
    assert "待归档原文已暂存" not in capsys.readouterr().out


@pytest.mark.parametrize("flag,value", [("--archive-history", True), ("--no-archive-history", False)])
def test_cli_forwards_archive_switch(monkeypatch, tmp_path, flag, value):
    source = save_learning_state(LearningState(learning_goal="目标", plan_markdown="计划"), tmp_path / "source.json")
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "note.md").write_text("资料", encoding="utf-8")
    received = []
    monkeypatch.setattr(app, "new_current_state_path", lambda: tmp_path / "current.json")
    monkeypatch.setattr(app, "run_interactive_session", lambda **kw: received.append(kw))
    monkeypatch.setattr("sys.argv", ["agent", "--resume-state", str(source), "--interactive",
                                     "--materials", str(materials), flag])
    app.main()
    assert received[0]["archive_history"] is value


def test_archive_failure_bypasses_summary_with_existing_history(api, monkeypatch, tmp_path, local):
    import ai_agent.persistence.memory as memory

    def fail(*args, **kwargs):
        raise OSError("模拟归档失败")

    monkeypatch.setattr(ConversationArchive, "append_turn", fail)
    monkeypatch.setattr(memory, "_request_summary", lambda _: pytest.fail("归档失败后不能压缩"))
    snapshot = MemorySnapshot(strategy="summary", max_chars=1, window_size=6, messages=[
        {"role": "human", "content": "旧问题"}, {"role": "ai", "content": "旧回答"},
    ])
    initial = LearningState(learning_goal="原目标", plan_markdown="原计划", conversation_memory=snapshot)
    with pytest.raises(api.HistoryPersistenceError):
        run(monkeypatch, tmp_path, local, ["新问题"], initial=initial)
    saved = load_learning_state(tmp_path / "current.json")
    assert len(saved.conversation_memory.messages) == 4
    assert saved.conversation_memory.history_archive.pending.user_text == "新问题"


def test_pending_recovery_failure_stops_before_any_model(api, monkeypatch, tmp_path, local):
    from ai_agent.persistence.memory_state import HistoryArchiveState, PendingArchiveTurn
    pending = PendingArchiveTurn(archive_id=uuid4().hex, session_id=uuid4().hex,
                                 turn_id=uuid4().hex, user_text="原话", assistant_text="回复")
    initial = LearningState(learning_goal="原目标", plan_markdown="原计划", conversation_memory=MemorySnapshot(
        strategy="window", max_chars=2000, window_size=6,
        history_archive=HistoryArchiveState(pending=pending)))

    def fail(*args, **kwargs):
        raise OSError("模拟归档不可用")

    monkeypatch.setattr(ConversationArchive, "append_turn", fail)
    monkeypatch.setattr(app, "create_llm", lambda **kw: pytest.fail("不得启动模型"))
    with pytest.raises(api.HistoryPersistenceError, match="未启动新问答"):
        run(monkeypatch, tmp_path, local, ["exit"], initial=initial, enabled=None)
    assert initial.conversation_memory.history_archive.pending == pending


def test_missing_committed_source_is_not_recreated_as_empty_database(api, tmp_path):
    capture = api.ArchiveCapture(tmp_path / "archives", None, True)
    capture.capture("第一轮", "回答")
    source = capture.state.sources[0]
    path = capture.root / f"{source.archive_id}.sqlite3"
    path.unlink()
    with pytest.raises(api.HistoryArchiveError):
        capture.capture("第二轮", "回答")
    assert capture.state.pending.user_text == "第二轮"
    assert not path.exists()


def test_archive_linkage_survives_two_independent_cli_processes(tmp_path):
    from ai_agent.persistence.session_materials import create_material_binding
    root = Path(__file__).resolve().parents[1]
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "notes.md").write_text("合成资料", encoding="utf-8")
    original = LearningState(learning_goal="原目标", plan_markdown="原计划",
                             material_binding=create_material_binding(str(materials)))
    source = save_learning_state(original, tmp_path / "source.json")
    source_bytes = source.read_bytes()
    first_bytes = None
    archive_root = tmp_path / "archives"
    for index in (1, 2):
        folder = tmp_path / f"process{index}"
        folder.mkdir()
        current = folder / "current.json"
        spec = folder / "input.json"
        spec.write_text(json.dumps({
            "source": str(source if index == 1 else tmp_path / "process1/current.json"),
            "current": str(current), "observation": str(folder / "request.json"),
            "inputs": ["  第一轮原话  " if index == 1 else "第二轮继续", "exit"],
            "response": "固定显示回复", "history_root": str(archive_root),
            "archive_history": True if index == 1 else None,
        }, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run([sys.executable, "-X", "utf8", "-B",
                                 str(root / "tests/helpers/feedback_plan_process_worker.py"), str(spec)],
                                cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        if index == 1:
            first_bytes = current.read_bytes()
    resumed = load_learning_state(tmp_path / "process2/current.json")
    refs = resumed.conversation_memory.history_archive.sources
    assert len(refs) == 2 and all(r.through_turn == 1 for r in refs)
    assert read_source(archive_root, refs[0], 1).user_text == "  第一轮原话  "
    assert read_source(archive_root, refs[1], 1).user_text == "第二轮继续"
    assert (tmp_path / "process1/current.json").read_bytes() == first_bytes
    assert source.read_bytes() == source_bytes
