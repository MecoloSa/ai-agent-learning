# 原文归档底座：完整新文件方案（尚未应用）

新建位置：src/ai_agent/persistence/history_archive.py。没有修改前代码，以下为完整新文件及导入。

## 必要性与边界

摘要成功后普通聊天原文会离开当前记忆，不能依靠摘要逐字还原。本步只建原文存储与精确读取底座，
不接入主交互、快照或模型工具，不宣称Agent已经具备回溯能力；不写入旧历史或从摘要反推原文。
新数据库独立使用application_id和schema_version，不改learning-state-v7。旧存档照常读取。

## 数据契约

- session_id：本归档会话标识。create_session生成，不从自然语言猜测。
- turn_id：调用方为完整问答生成的稳定UUID十六进制字符串；重试同一提交必须复用。
- turn_number：数据库在写事务中分配的会话内顺序号。
- user_text、assistant_text：原样保存传入文本，保留空白与标点。
- message_id：turn_id:user或turn_id:assistant，可供未来检索命中后准确回读。
- recorded_at：归档写入UTC时间，不冒充用户发言发生时间。

本层不判断用户事实是否正确，不执行任务或反馈操作。将来主流程需把真实显示回复和程序确认回执分开保存；
当前memory中的程序增强文本不能直接冒充原始助手回复。原输入在strip之前捕获，否则无法承诺保留外侧空白。
数据库路径由可信调用方绑定；未来工具只允许访问当前会话及显式授权来源，不能把session_id校验当作权限校验。
本层可以服务不同会话，不等于已实现模型访问控制。

## 完整新文件

```python
"""原文归档底座：完整问答原样保存；尚未接入Agent主流程或模型工具。"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from typing import Literal
from uuid import uuid4

_APPLICATION_ID = 0x41474831
_SCHEMA_VERSION = 1
MAX_READ_CHARS = 20000


class HistoryArchiveError(ValueError):
    """归档标识、格式或提交不符合契约。"""


class HistoryReadLimitError(HistoryArchiveError):
    """本次读取超过明确预算；不返回伪装成完整原文的截断文本。"""


def _identifier(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise HistoryArchiveError("归档ID必须为32位小写十六进制字符串")
    return value


def _text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HistoryArchiveError("消息必须为非空文本")
    # 校验不等于规范化：保留前后空格、换行与标点。
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class HistoryTurn:
    session_id: str
    turn_id: str
    turn_number: int
    user_text: str
    assistant_text: str
    recorded_at: str

    def message_id(self, role: Literal["user", "assistant"]) -> str:
        if role not in {"user", "assistant"}:
            raise HistoryArchiveError("未知消息角色")
        return f"{self.turn_id}:{role}"


class ConversationArchive:
    def __init__(self, path: Path):
        # 数据库路径只接受可信调用方配置，未来不能暴露为模型工具参数。
        self.path = Path(path).resolve()

    def _connect(self, mode: str):
        # ro/rw模式不创建缺失数据库；只有显式initialize使用rwc。
        connection = sqlite3.connect(
            f"{self.path.as_uri()}?mode={mode}", uri=True, timeout=5,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _check(connection) -> None:
        app = connection.execute("PRAGMA application_id").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if app != _APPLICATION_ID or version != _SCHEMA_VERSION:
            raise HistoryArchiveError("不是本应用支持的原文归档格式")

    def initialize(self) -> None:
        """仅显式初始化可以建库；不把其他数据库静默迁移成本应用数据库。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect("rwc")) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            app = connection.execute("PRAGMA application_id").fetchone()[0]
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if app == _APPLICATION_ID and version == _SCHEMA_VERSION:
                return
            if app != 0 or version != 0 or tables:
                raise HistoryArchiveError("拒绝初始化非空或未知格式数据库")
            connection.execute(
                "CREATE TABLE history_sessions (session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL)"
            )
            connection.execute(
                """CREATE TABLE history_turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES history_sessions(session_id),
                    turn_number INTEGER NOT NULL CHECK(turn_number > 0),
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    UNIQUE(session_id, turn_number)
                )"""
            )
            connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")

    def create_session(self) -> str:
        session_id = uuid4().hex
        with closing(self._connect("rw")) as connection, connection:
            self._check(connection)
            connection.execute(
                "INSERT INTO history_sessions VALUES (?, ?)", (session_id, _now()),
            )
        return session_id

    def append_turn(
        self, session_id: str, turn_id: str, user_text: str, assistant_text: str,
    ) -> HistoryTurn:
        """调用方为同一完整问答复用turn_id；相同内容重试幂等，冲突则拒绝。"""
        session_id, turn_id = _identifier(session_id), _identifier(turn_id)
        user_text, assistant_text = _text(user_text), _text(assistant_text)
        with closing(self._connect("rw")) as connection, connection:
            # 锁定本次写事务，避免并发写入者分配到同一个轮次。
            connection.execute("BEGIN IMMEDIATE")
            self._check(connection)
            old = connection.execute(
                "SELECT * FROM history_turns WHERE turn_id=?", (turn_id,),
            ).fetchone()
            if old is not None:
                if (old["session_id"], old["user_text"], old["assistant_text"]) != (
                    session_id, user_text, assistant_text,
                ):
                    raise HistoryArchiveError("同一turn_id的来源或内容发生冲突")
                return HistoryTurn(**dict(old))
            if connection.execute(
                "SELECT 1 FROM history_sessions WHERE session_id=?", (session_id,),
            ).fetchone() is None:
                raise HistoryArchiveError("归档会话不存在")
            number = connection.execute(
                "SELECT COALESCE(MAX(turn_number), 0) + 1 FROM history_turns WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
            turn = HistoryTurn(session_id, turn_id, number, user_text, assistant_text, _now())
            # 一问一答在同一行、同一事务提交，不会只写入用户消息。
            connection.execute(
                "INSERT INTO history_turns VALUES (?, ?, ?, ?, ?, ?)",
                (turn.turn_id, turn.session_id, turn.turn_number,
                 turn.user_text, turn.assistant_text, turn.recorded_at),
            )
            return turn

    def read_turn(self, session_id: str, turn_number: int, *, max_chars: int = 4000) -> HistoryTurn | None:
        """精确读取完整轮次；缺失返回None，超限明确拒绝，不截断原文。"""
        session_id = _identifier(session_id)
        if type(turn_number) is not int or turn_number < 1:
            raise HistoryArchiveError("轮次必须为正整数")
        if type(max_chars) is not int or not 1 <= max_chars <= MAX_READ_CHARS:
            raise HistoryReadLimitError("读取预算必须在1至20000字符之间")
        with closing(self._connect("ro")) as connection:
            self._check(connection)
            row = connection.execute(
                "SELECT * FROM history_turns WHERE session_id=? AND turn_number=?",
                (session_id, turn_number),
            ).fetchone()
            if row is None:
                return None
            if len(row["user_text"]) + len(row["assistant_text"]) > max_chars:
                raise HistoryReadLimitError("本轮完整原文超过读取预算")
            return HistoryTurn(**dict(row))
```

## 取舍

SQLite来自Python标准库，无新增依赖。完整问答原子提交；重复turn_id且同内容返回已有记录，冲突拒绝。
BEGIN IMMEDIATE串行分配轮次；繁忙超时或其他写入异常继续上抛，由下一步会话接线明确处理。
不宣称同时提交业务存档与归档具备跨文件事务；下一步必须决定写入顺序、失败回执与恢复来源关联。
读方法使用只读连接、固定会话与轮次，完整内容超预算则拒绝，不把截断内容冒充原话。
本层没有流式半轮恢复、模型自动检索、旧历史补全、删除/保留策略或向量索引。

## 实施顺序与后续

1. 应用本新模块，执行tests/test_history_archive.py及演示脚本。
2. 再审查主流程接线：原文捕获、稳定ID、显示回复/程序回执分离、归档失败语义、快照指针及旧版本兼容。
3. 接入绑定会话的只读历史工具，并复用原话引用案例验收。
4. 建立精确/词法检索基线后，再按实际检索困难决定向量召回。

