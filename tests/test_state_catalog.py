"""存档目录只读入口测试：真实临时文件，不接模型、不污染学习存档。"""

import os

import pytest

import ai_agent.config as config
import ai_agent.state_catalog as catalog
from ai_agent.learning.learning_state import LearningState, LearningTask
from ai_agent.persistence.memory_state import MemorySnapshot
from ai_agent.persistence.state_store import save_learning_state, save_current_learning_state


def state(goal="理解ReAct"):
    return LearningState(
        learning_goal=goal,
        plan_markdown="不应打印的计划正文",
        tasks=(LearningTask(task_id="t1", title="阅读", latest_feedback="不应打印的反馈"),),
    )


def test_both_locations_are_sorted_by_modified_time(tmp_path):
    old = save_learning_state(state("旧目标"), tmp_path / "history.json")
    current = save_current_learning_state(state("新目标"), tmp_path / "current" / "session.json")
    os.utime(old, (1000, 1000))
    os.utime(current, (2000, 2000))
    entries = catalog.list_state_entries(tmp_path)
    assert [entry.path for entry in entries] == [current, old]
    assert [entry.kind for entry in entries] == ["当前存档", "历史/旧版"]
    assert entries[0].state.learning_goal == "新目标"
    assert entries[1].modified_at.timestamp() == 1000


def test_bad_latest_file_is_shown_not_silently_replaced(tmp_path):
    good = save_learning_state(state(), tmp_path / "good.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    os.utime(good, (1000, 1000))
    os.utime(bad, (2000, 2000))
    entries = catalog.list_state_entries(tmp_path, limit=1)
    assert len(entries) == 1
    assert entries[0].path == bad
    assert entries[0].state is None
    assert entries[0].error_type is not None


def test_only_two_known_directory_levels_are_scanned(tmp_path):
    expected = save_learning_state(state(), tmp_path / "valid.json")
    save_learning_state(state(), tmp_path / "other" / "hidden.json")
    save_learning_state(state(), tmp_path / "current" / "nested" / "hidden.json")
    (tmp_path / ".learning-state-test.tmp").write_text("{broken", encoding="utf-8")
    assert [entry.path for entry in catalog.list_state_entries(tmp_path)] == [expected]


def test_missing_directory_is_empty(tmp_path):
    assert catalog.list_state_entries(tmp_path / "missing") == []
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_is_rejected(tmp_path, limit):
    with pytest.raises(ValueError):
        catalog.list_state_entries(tmp_path, limit=limit)


def test_listing_does_not_write_and_does_not_call_models(monkeypatch, tmp_path, capsys):
    path = save_learning_state(state(), tmp_path / "history.json")
    original = path.read_bytes()
    modified = path.stat().st_mtime_ns
    monkeypatch.setattr(catalog, "resolve_project_path", lambda _: tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("目录查看不应创建模型或加载模型配置")

    monkeypatch.setattr(config, "create_llm", unexpected)
    monkeypatch.setattr(config, "get_settings", unexpected)
    catalog.main([])
    output = capsys.readouterr().out
    assert "理解ReAct" in output
    assert str(path) in output
    assert "任务数=1" in output
    assert "记忆策略=未保存" in output
    assert "不应打印的计划正文" not in output
    assert "不应打印的反馈" not in output
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == modified
    assert list(tmp_path.iterdir()) == [path]


def test_cli_empty_and_invalid_limit(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(catalog, "resolve_project_path", lambda _: tmp_path / "missing")
    catalog.main([])
    assert "暂无存档" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        catalog.main(["--limit", "0"])
    assert exc.value.code == 2


def test_corruption_is_reported_without_raw_payload(monkeypatch, tmp_path, capsys):
    (tmp_path / "bad.json").write_text('{"private": "不应泄露的原文"}', encoding="utf-8")
    monkeypatch.setattr(catalog, "resolve_project_path", lambda _: tmp_path)
    catalog.main([])
    output = capsys.readouterr().out
    assert "无法读取" in output
    assert "不应泄露的原文" not in output


def test_saved_memory_strategy_is_displayed(monkeypatch, tmp_path, capsys):
    memory = MemorySnapshot(strategy="full", max_chars=2000, window_size=6)
    stored = LearningState(
        learning_goal="继续学习", plan_markdown="计划", conversation_memory=memory,
    )
    save_learning_state(stored, tmp_path / "full.json")
    monkeypatch.setattr(catalog, "resolve_project_path", lambda _: tmp_path)
    catalog.main([])
    assert "记忆策略=full" in capsys.readouterr().out
