"""学习状态快照测试。

使用 pytest 的 tmp_path 临时目录，不写入真实学习记录。
检查内容恢复、禁止覆盖、损坏拒绝以及保存失败时的状态保护。
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import ai_agent.learning.learning_commands as commands_module
import ai_agent.persistence.state_store as store_module
from ai_agent.learning.learning_state import (
    LearningState,
    LearningTask,
    UserFeedback,
    apply_user_feedback,
)
from ai_agent.persistence.state_store import (
    save_learning_state,
    load_learning_state,
)


def make_state():
    state = LearningState(
        learning_goal="理解文章观点",
        plan_markdown="阅读文章并比较观点。",
        tasks=(
            LearningTask(task_id="task-001", title="阅读文章"),
        ),
    )
    return apply_user_feedback(
        state,
        UserFeedback(
            task_id="task-001",
            text="文章只读了一半，两个观点的区别没理解。",
            activity_status="in_progress",
            understanding_status="self_reported_unclear",
        ),
    )


def test_saved_state_restores_all_fields(tmp_path):
    state = make_state()
    path = tmp_path / "nested" / "state.json"

    save_learning_state(state, path)
    restored = load_learning_state(path)

    assert restored == state
    assert restored.tasks[0].latest_feedback == (
        "文章只读了一半，两个观点的区别没理解。"
    )
    assert restored.tasks[0].feedback_source == "user_report"

    # 恢复的不是普通字典，可以继续使用现有更新函数。
    updated = apply_user_feedback(
        restored,
        UserFeedback(
            task_id="task-001",
            text="文章读完了。",
            activity_status="completed",
        ),
    )
    assert updated.tasks[0].activity_status == "completed"
    assert updated.tasks[0].understanding_status == "self_reported_unclear"


def test_existing_snapshot_is_not_overwritten(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("原有记录", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_learning_state(make_state(), path)

    assert path.read_text(encoding="utf-8") == "原有记录"


def test_missing_snapshot_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_learning_state(tmp_path / "missing.json")


@pytest.mark.parametrize("invalid_kind", [
    "broken_json",
    "unknown_version",
    "invalid_status",
    "duplicate_id",
])
def test_invalid_snapshot_is_rejected(tmp_path, invalid_kind):
    path = tmp_path / "invalid.json"
    data = {
        "schema_version": "learning-state-v1",
        "state": make_state().model_dump(mode="json"),
    }

    if invalid_kind == "broken_json":
        path.write_text("{unfinished", encoding="utf-8")
    else:
        if invalid_kind == "unknown_version":
            data["schema_version"] = "future-version"
        elif invalid_kind == "invalid_status":
            data["state"]["tasks"][0]["activity_status"] = "finished"
        else:
            data["state"]["tasks"].append(
                dict(data["state"]["tasks"][0])
            )

        path.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )

    with pytest.raises(ValidationError):
        load_learning_state(path)


def test_save_command_preserves_in_memory_state(monkeypatch, tmp_path):
    # 把命令的输出位置改到临时目录，避免污染真实快照目录。
    monkeypatch.setattr(
        commands_module,
        "resolve_project_path",
        lambda value: tmp_path / Path(value).name,
    )
    state = make_state()

    updated, message = commands_module.handle_learning_command(
        state, "/save-state",
    )

    assert updated is state
    assert "快照已保存" in message
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert load_learning_state(files[0]) == state


def test_save_failure_does_not_report_success(monkeypatch):
    def fail_save(state, path):
        raise OSError("模拟磁盘写入失败")

    monkeypatch.setattr(
        commands_module, "save_learning_state", fail_save,
    )
    state = make_state()

    updated, message = commands_module.handle_learning_command(
        state, "/save-state",
    )

    assert updated is state
    assert "未执行" in message
    assert "快照已保存" not in message

def test_snapshot_names_include_date_and_remain_unique(monkeypatch, tmp_path):
    """同一秒保存两次，日期正确、文件不重名、状态内容相同。"""
    from datetime import datetime, timezone, timedelta
    from types import SimpleNamespace

    class FixedDatetime:
        @staticmethod
        def now(tz):
            # 固定时间，避免测试恰好跨秒、跨午夜造成偶发失败。
            assert tz == timezone(timedelta(hours=8))
            return datetime(2026, 10, 10, 9, 45, 0, tzinfo=tz)

    monkeypatch.setattr(commands_module, "datetime", FixedDatetime)
    monkeypatch.setattr(
        commands_module,
        "resolve_project_path",
        lambda value: tmp_path / Path(value).name,
    )

    # 固定两个不同 UUID，明确验证唯一性来自后缀，而不是时间变化。
    identifiers = iter(["a" * 32, "b" * 32])
    monkeypatch.setattr(
        commands_module,
        "uuid4",
        lambda: SimpleNamespace(hex=next(identifiers)),
    )

    state = make_state()
    for _ in range(2):
        updated, message = commands_module.handle_learning_command(
            state, "/save-state",
        )
        assert updated is state
        assert "快照已保存" in message

    files = sorted(tmp_path.glob("*.json"))
    assert [path.name for path in files] == [
        f"261010_094500_{'a' * 32}.json",
        f"261010_094500_{'b' * 32}.json",
    ]
    assert all(load_learning_state(path) == state for path in files)

@pytest.mark.parametrize("failure_stage", ["sync", "publish"])
def test_failed_save_does_not_leave_formal_snapshot(
    monkeypatch, tmp_path, failure_stage,
):
    """模拟刷新或发布失败：不产生正式快照，也不影响旧记录。"""
    state = make_state()
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"

    save_learning_state(state, old_path)
    original_bytes = old_path.read_bytes()

    def fail_operation(*args, **kwargs):
        raise OSError(f"模拟 {failure_stage} 失败")

    # monkeypatch 临时替换系统操作，测试结束后自动恢复。
    # 不需要真的制造磁盘故障，也不调用模型。
    operation = "fsync" if failure_stage == "sync" else "link"
    monkeypatch.setattr(store_module.os, operation, fail_operation)

    with pytest.raises(OSError, match="模拟"):
        save_learning_state(state, new_path)

    print(f"\n[失败阶段] {failure_stage}")
    print(f"正式新快照存在：{new_path.exists()}")
    print(f"旧快照内容不变：{old_path.read_bytes() == original_bytes}")

    assert not new_path.exists()
    assert old_path.read_bytes() == original_bytes
    assert load_learning_state(old_path) == state
    assert not list(tmp_path.glob(".learning-state-*.tmp"))


def test_snapshot_is_complete_before_publication(monkeypatch, tmp_path):
    """在发布之前检查临时文件：它必须已经能完整恢复。"""
    state = make_state()
    path = tmp_path / "state.json"
    real_link = store_module.os.link

    def inspect_then_publish(source, destination):
        assert not Path(destination).exists()

        # 在正式文件尚不存在时，临时文件已是可恢复的完整快照。
        assert load_learning_state(Path(source)) == state
        print("\n[发布前] 正式文件不存在，临时文件已通过恢复校验")

        real_link(source, destination)

    monkeypatch.setattr(store_module.os, "link", inspect_then_publish)

    save_learning_state(state, path)

    assert load_learning_state(path) == state
    assert not list(tmp_path.glob(".learning-state-*.tmp"))
    print("[发布后] 正式快照可恢复，临时文件已清理")


def test_target_created_during_save_is_not_overwritten(
    monkeypatch, tmp_path,
):
    """模拟另一进程抢先创建目标，验证发布阶段也不会覆盖它。"""
    path = tmp_path / "state.json"
    real_link = store_module.os.link

    def competing_publish(source, destination):
        # 模拟竞争发生在临时文件已写好、正式发布尚未完成的时候。
        Path(destination).write_text("另一进程的记录", encoding="utf-8")
        real_link(source, destination)

    monkeypatch.setattr(store_module.os, "link", competing_publish)

    with pytest.raises(FileExistsError):
        save_learning_state(make_state(), path)

    assert path.read_text(encoding="utf-8") == "另一进程的记录"
    assert not list(tmp_path.glob(".learning-state-*.tmp"))
    print("\n[竞争保护] 保存被拒绝，已有目标内容未被覆盖")
