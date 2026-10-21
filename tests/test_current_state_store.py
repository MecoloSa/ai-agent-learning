"""当前存档的磁盘契约：允许替换，但不能破坏历史快照或失败前的存档。

只使用 tmp_path 和故障替身；不调用模型、不写真实学习记录。
"""

import pytest

import ai_agent.persistence.state_store as store
from ai_agent.learning.learning_state import LearningState


def make_state(goal="理解 ReAct"):
    return LearningState(learning_goal=goal, plan_markdown="阅读并比较终止条件。")


def test_current_save_reuses_one_file(tmp_path):
    path = tmp_path / "session" / "current.json"
    first, second = make_state(), make_state("理解工具回传")
    assert store.save_current_learning_state(first, path) == path
    assert store.save_current_learning_state(second, path) == path
    assert store.load_learning_state(path) == second
    assert list(path.parent.glob("*.json")) == [path]
    assert not list(path.parent.glob(".learning-state-*.tmp"))


@pytest.mark.parametrize("stage", ["fsync", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_current_save_preserves_previous_file(
    monkeypatch, tmp_path, stage, existing,
):
    path = tmp_path / "current.json"
    if existing:
        store.save_learning_state(make_state(), path)
    original = path.read_bytes() if existing else None

    def fail(*args, **kwargs):
        raise OSError("模拟发布失败")

    monkeypatch.setattr(store.os, stage, fail)
    with pytest.raises(OSError, match="模拟"):
        store.save_current_learning_state(make_state("新目标"), path)
    assert (path.read_bytes() if path.exists() else None) == original
    assert not list(tmp_path.glob(".learning-state-*.tmp"))


def test_replace_publishes_complete_state(monkeypatch, tmp_path):
    path = tmp_path / "current.json"
    first, second = make_state(), make_state("新目标")
    store.save_learning_state(first, path)
    real_replace = store.os.replace

    def inspect_then_replace(source, destination):
        # 发布前旧文件仍可读，新文件也已完整，不能先清空正式文件。
        assert store.load_learning_state(path) == first
        assert store.load_learning_state(source) == second
        real_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", inspect_then_replace)
    store.save_current_learning_state(second, path)
    assert store.load_learning_state(path) == second


def test_history_remains_immutable_and_independent(tmp_path):
    history, current = tmp_path / "history.json", tmp_path / "current.json"
    first, second = make_state(), make_state("新目标")
    store.save_learning_state(first, history)
    original = history.read_bytes()
    store.save_current_learning_state(first, current)
    store.save_current_learning_state(second, current)
    with pytest.raises(FileExistsError):
        store.save_learning_state(second, history)
    assert history.read_bytes() == original
    assert store.load_learning_state(current) == second
