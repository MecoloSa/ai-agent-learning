"""普通问答原文归档接线；无模型调用，不接受模型指定数据库路径。"""
from __future__ import annotations

from pathlib import Path
import sqlite3
from uuid import uuid4

from ai_agent.persistence.history_archive import ConversationArchive, HistoryArchiveError
from ai_agent.persistence.memory_state import ArchiveSource, HistoryArchiveState, PendingArchiveTurn

ARCHIVE_STORAGE_ERRORS = (OSError, sqlite3.Error, HistoryArchiveError)


class HistoryPersistenceError(RuntimeError):
    """原文归档不能继续；CLI显示明确回执并非零退出。"""


class ArchiveCapture:
    def __init__(self, root: Path, state: HistoryArchiveState | None, enabled: bool | None):
        self.root = Path(root).resolve()
        requested = enabled if enabled is not None else (state.enabled if state is not None else False)
        if state is not None and state.pending is not None and not requested:
            raise HistoryPersistenceError("存在待归档原文，请先开启归档完成恢复，再关闭归档。")
        self.state = HistoryArchiveState(
            enabled=requested,
            sources=state.sources if state is not None else (),
            pending=state.pending if state is not None else None,
        )
        # 每次进入交互都新建分支，不继续向来源会话追加新的问答。
        self._active: tuple[str, str] | None = None

    def flush_pending(self):
        pending = self.state.pending
        if pending is None:
            return None
        previous = next((s for s in self.state.sources
                         if (s.archive_id, s.session_id) == (pending.archive_id, pending.session_id)), None)
        archive_path = (self.root / f"{pending.archive_id}.sqlite3").resolve()
        if not archive_path.is_relative_to(self.root):
            raise HistoryArchiveError("归档文件越出绑定目录")
        archive = ConversationArchive(archive_path)
        if previous is None:
            archive.initialize()
            archive.ensure_session(pending.session_id)
        elif not archive.path.is_file():
            raise HistoryArchiveError("已提交的原文来源缺失，拒绝重建空库")
        turn = archive.append_turn(
            pending.session_id, pending.turn_id, pending.user_text, pending.assistant_text,
        )
        source = ArchiveSource(
            archive_id=pending.archive_id, session_id=pending.session_id,
            through_turn=max(turn.turn_number, previous.through_turn if previous is not None else 0),
        )
        sources = tuple(s for s in self.state.sources
                        if (s.archive_id, s.session_id) != (source.archive_id, source.session_id))
        self.state = HistoryArchiveState(enabled=self.state.enabled, sources=(*sources, source))
        return turn

    def capture(self, user_text: str, assistant_text: str):
        if not self.state.enabled:
            return None
        if self.state.pending is not None:
            raise HistoryArchiveError("待归档消息尚未处理，拒绝覆盖")
        if self._active is None:
            self._active = (uuid4().hex, uuid4().hex)
        pending = PendingArchiveTurn(
            archive_id=self._active[0], session_id=self._active[1], turn_id=uuid4().hex,
            user_text=user_text, assistant_text=assistant_text,
        )
        # 在数据库操作之前记录稳定ID；异常后仍可导出原文并幂等补写。
        self.state = HistoryArchiveState(enabled=True, sources=self.state.sources, pending=pending)
        return self.flush_pending()