"""原文归档底座：完整问答原样保存，支持稳定定位与精确读取。"""
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
        return self.ensure_session(uuid4().hex)

    def ensure_session(self, session_id: str) -> str:
        """支持待归档恢复：会话ID由程序预先分配，重复确保不改变既有记录。"""
        session_id = _identifier(session_id)
        with closing(self._connect("rw")) as connection, connection:
            self._check(connection)
            connection.execute(
                "INSERT OR IGNORE INTO history_sessions VALUES (?, ?)", (session_id, _now()),
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

    def load_user_turns(self, session_id: str, *, through_turn: int) -> list[dict]:
        """为历史检索加载受限前缀；超预算整体拒绝，不悄悄漏掉早期消息。"""
        session_id = _identifier(session_id)
        if type(through_turn) is not int or through_turn < 1:
            raise HistoryArchiveError("来源上限必须为正整数")
        with closing(self._connect("ro")) as connection:
            self._check(connection)
            # 在同一读取事务中检查体量并加载，避免两次查询看到不同数据库状态。
            connection.execute("BEGIN")
            count, chars = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(length(user_text)), 0) FROM history_turns "
                "WHERE session_id=? AND turn_number<=?", (session_id, through_turn),
            ).fetchone()
            if count > 200 or chars > 100000:
                    raise HistoryReadLimitError("当前来源超过200轮或100000用户字符的检索预算")
            rows = connection.execute(
                "SELECT turn_number, user_text FROM history_turns "
                "WHERE session_id=? AND turn_number<=? ORDER BY turn_number",
                (session_id, through_turn),
            ).fetchall()
            return [dict(row) for row in rows]

