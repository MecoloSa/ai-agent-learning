"""本地图检查点存储入口。

只负责打开和关闭SQLite检查点连接，不生成计划，也不自动恢复。
数据库可能包含完整分析和资料证据，只应打开自己生成的可信文件。
"""
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from ai_agent.config import resolve_project_path


@contextmanager
def open_graph_checkpoint(
    db_path: str | Path = ".agents_data/checkpoints/workflows.sqlite",
):
    """在with语句范围内提供可用的SQLite检查点存储。"""
    path = resolve_project_path(db_path)
    if path.suffix.lower() != ".sqlite":
        raise ValueError("检查点数据库请使用.sqlite后缀")

    path.parent.mkdir(parents=True, exist_ok=True)

    # yield把连接暂时交给调用方；离开with后由底层关闭连接。
    # 不能把saver保存起来，在with结束之后继续使用。
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver
