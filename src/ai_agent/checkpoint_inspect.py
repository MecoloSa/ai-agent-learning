"""工作流检查点的只读摘要查询。

只读取记录数量和检查点标识，不反序列化资料正文、不执行图。
依赖当前SqliteSaver的表结构；不将“记录存在”判断为“允许恢复”。
"""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .config import resolve_project_path


def inspect_checkpoint_db(
    db_path: str | Path,
    workflow_id: str,
) -> dict:
    """查看指定任务根命名空间中的检查点摘要。"""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        raise ValueError("workflow_id不能为空")

    path = resolve_project_path(db_path)
    if not path.is_file():
        raise ValueError("数据库不存在或不是文件")

    # mode=ro从连接层禁止写入，也不会把错误路径变成新数据库。
    uri = path.as_uri() + "?mode=ro"

    # sqlite连接自身的with主要管理事务，不保证关闭连接；
    # closing确保查询成功或失败后都释放连接。
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")

        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(checkpoints)"
            )
        }
        required = {
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "parent_checkpoint_id",
        }
        if not required.issubset(columns):
            raise ValueError("数据库缺少预期的检查点表结构")

        # 当前三节点图使用根命名空间。
        # 若以后增加子图，不应把子图记录混算成主图检查点。
        parameters = (workflow_id, "")
        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM checkpoints
            WHERE thread_id = ? AND checkpoint_ns = ?
            """,
            parameters,
        ).fetchone()[0]

        if count == 0:
            raise ValueError("该任务没有根命名空间检查点")

        # 与当前SqliteSaver读取最新记录的排序方式一致。
        latest = connection.execute(
            """
            SELECT checkpoint_id, parent_checkpoint_id
            FROM checkpoints
            WHERE thread_id = ? AND checkpoint_ns = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """,
            parameters,
        ).fetchone()

    # 只输出标识与计数，不输出分析、证据、反馈或模型消息。
    return {
        "workflow_id": workflow_id,
        "checkpoint_namespace": "",
        "checkpoint_count": count,
        "latest_checkpoint_id": latest[0],
        "parent_checkpoint_id": latest[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="只读查看工作流检查点摘要，不执行或恢复任务"
    )
    parser.add_argument("--checkpoint-db", required=True)
    parser.add_argument("--workflow-id", required=True)
    args = parser.parse_args()

    try:
        summary = inspect_checkpoint_db(
            args.checkpoint_db,
            args.workflow_id,
        )
    except ValueError as exc:
        parser.exit(1, f"查询失败：{exc}\n")
    except (OSError, sqlite3.Error) as exc:
        parser.exit(
            1,
            f"数据库读取失败：{type(exc).__name__}\n",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()