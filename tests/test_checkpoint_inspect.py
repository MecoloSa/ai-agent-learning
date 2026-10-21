"""检查点摘要查询测试。

使用真实SQLite文件及最小兼容表结构，不调用模型或执行工作流。
验证任务隔离、只读查询和错误路径不会创建数据库。
"""
import sqlite3

import pytest

from ai_agent.checkpoint_inspect import inspect_checkpoint_db


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "workflow test.sqlite"

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE checkpoints (
                thread_id TEXT,
                checkpoint_ns TEXT,
                checkpoint_id TEXT,
                parent_checkpoint_id TEXT,
                checkpoint BLOB
            )
            """
        )
        connection.executemany(
            "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
            [
                ("task-1", "", "001", None, b"private-evidence"),
                ("task-1", "", "002", "001", b"private-evidence"),
                ("task-2", "", "003", None, b"other-evidence"),
                ("task-1", "child", "004", None, b"child-evidence"),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    return path


def test_summary_is_scoped_and_does_not_change_database(database):
    before = database.read_bytes()

    result = inspect_checkpoint_db(database, "task-1")

    assert result == {
        "workflow_id": "task-1",
        "checkpoint_namespace": "",
        "checkpoint_count": 2,
        "latest_checkpoint_id": "002",
        "parent_checkpoint_id": "001",
    }

    # 动态评估：查询不应修改数据库，也不应返回资料正文。
    assert database.read_bytes() == before
    assert "private-evidence" not in str(result)


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite"

    with pytest.raises(ValueError, match="数据库不存在"):
        inspect_checkpoint_db(path, "task-1")

    assert not path.exists()


def test_unknown_workflow_is_rejected(database):
    with pytest.raises(ValueError, match="没有根命名空间检查点"):
        inspect_checkpoint_db(database, "unknown-task")


def test_wrong_database_schema_is_rejected(tmp_path):
    path = tmp_path / "other.sqlite"
    connection = sqlite3.connect(path)
    connection.close()

    with pytest.raises(ValueError, match="检查点表结构"):
        inspect_checkpoint_db(path, "task-1")