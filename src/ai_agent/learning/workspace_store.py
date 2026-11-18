"""Durable local workspace. One transaction publishes state, proposals and receipts."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from pydantic import Field

from ai_agent.learning.learning_state import LearningState, LearningTask, StateModel
from ai_agent.learning.plan_tasks import TaskChange, place_split_children
from ai_agent.persistence.memory_state import MemorySnapshot


class WorkspaceConflict(ValueError):
    pass


class PlanCandidate(StateModel):
    id: str
    base_version: int
    mode: Literal["adjust", "replace", "chat", "extract_tasks", "split_task"]
    instructions: str
    base_plan: str
    markdown: str = Field(min_length=1)
    status: Literal["pending", "adopted", "rejected"] = "pending"
    previous_id: str | None = None
    task_changes: tuple[TaskChange, ...] = ()
    task_baseline: dict[str, LearningTask] = Field(default_factory=dict)
    conversation_id: str | None = None


class WorkspaceEvent(StateModel):
    action: str
    status: str
    message: str
    input_text: str = ""
    proposal_id: str | None = None
    task_delta: tuple[dict, ...] = ()
    conversation_id: str | None = None
    title_seed: bool = False


class WorkspaceConversation(StateModel):
    id: str
    title: str = "新对话"
    memory: MemorySnapshot | None = None
    state: LearningState | None = None
    version: int = 1


def empty_learning_state() -> LearningState:
    return LearningState(learning_goal="待确定学习目标", plan_markdown="尚未制定学习计划。")


class Workspace(StateModel):
    state: LearningState
    # sequence detects competing writes, including conversation-only writes.
    sequence: int = 0
    # version advances only when formal data changes. Draft creation leaves it intact.
    version: int = 1
    proposals: tuple[PlanCandidate, ...] = ()
    events: tuple[WorkspaceEvent, ...] = ()
    conversations: tuple[WorkspaceConversation, ...] = ()
    original_deleted: bool = False
    original_title: str = "原始对话"

    def scope(self, conversation_id: str | None) -> Workspace:
        if conversation_id is None:
            if self.original_deleted:
                raise WorkspaceConflict("该对话已删除，请选择其他对话。")
            state, version = self.state, self.version
        else:
            session = next((item for item in self.conversations if item.id == conversation_id), None)
            if session is None:
                raise WorkspaceConflict("该对话不存在，请刷新对话列表。")
            state = (session.state or empty_learning_state()).model_copy(update={"conversation_memory": session.memory})
            version = session.version
        events = tuple(event for event in self.events if event.conversation_id == conversation_id)
        # 旧库的候选没有归属字段，以首次生成该候选的事件补全归属。
        owners = {event.proposal_id: event.conversation_id for event in reversed(self.events)
                  if event.proposal_id is not None}
        proposals = tuple(item for item in self.proposals
                          if owners.get(item.id, item.conversation_id) == conversation_id)
        state = _restore_split_order(state, events, proposals)
        return self.model_copy(update={"state": state, "version": version, "events": events, "proposals": proposals})


def _restore_split_order(state, events, proposals):
    splits = {item.id: item for item in proposals if item.mode == "split_task" and item.status == "adopted"}
    if not splits:
        return state
    tasks = state.tasks
    known_ids = {task.task_id for task in tasks}
    for event in events:
        candidate = splits.get(event.proposal_id)
        if event.action != "decide" or event.status != "adopted" or candidate is None:
            continue
        parents = [change.task_id for change in candidate.task_changes if change.operation == "archive"]
        expected_titles = [change.title for change in candidate.task_changes if change.operation == "add"]
        archived = [delta["task_id"] for delta in event.task_delta
                    if delta["changes"].get("is_cancelled") == {"before": False, "after": True}]
        created = [(delta["task_id"], delta["changes"]["title"]["after"]) for delta in event.task_delta
                   if delta["changes"].get("title", {}).get("before") is None and "title" in delta["changes"]]
        child_ids = tuple(task_id for task_id, _ in created)
        if (len(parents) == 1 and parents == archived and len(child_ids) >= 2
                and [title for _, title in created] == expected_titles
                and {parents[0], *child_ids} <= known_ids):
            tasks = place_split_children(tasks, parents[0], child_ids)
    return state.model_copy(update={"tasks": tasks}) if tasks != state.tasks else state


class WorkspaceStore:
    APP_ID = 0x41475731

    def __init__(self, path: Path, initial: LearningState | None = None, *, create: bool = False):
        self.path = path.resolve()
        if initial is None and not create and not self.path.is_file():
            raise ValueError("工作区数据库不存在。")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            app_id = db.execute("PRAGMA application_id").fetchone()[0]
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if app_id == self.APP_ID and version == 1:
                if initial is not None:
                    raise ValueError("工作区已存在，请直接恢复或为导入指定新的路径。")
                db.execute("CREATE TABLE IF NOT EXISTS conversation_titles (id TEXT PRIMARY KEY, title TEXT NOT NULL)")
                return
            tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if app_id or version or tables or (initial is None and not create):
                raise ValueError("工作区数据库格式不兼容。")
            if initial is None:
                # 新建默认工作区也走同一事务。再次启动直接恢复上方已有的数据。
                initial = empty_learning_state()
            db.execute("CREATE TABLE workspace (id INTEGER PRIMARY KEY CHECK(id=1), sequence INTEGER, payload TEXT)")
            db.execute("CREATE TABLE requests (id TEXT PRIMARY KEY, fingerprint TEXT, event TEXT)")
            db.execute("CREATE TABLE conversation_titles (id TEXT PRIMARY KEY, title TEXT NOT NULL)")
            db.execute("INSERT INTO workspace VALUES (1, 0, ?)", (Workspace(state=initial).model_dump_json(),))
            db.execute(f"PRAGMA application_id={self.APP_ID}")
            db.execute("PRAGMA user_version=1")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def load(self) -> Workspace:
        with closing(self._connect()) as db:
            db.execute("BEGIN")
            payload = db.execute("SELECT payload FROM workspace WHERE id=1").fetchone()[0]
            titles = dict(db.execute("SELECT id, title FROM conversation_titles"))
        workspace = Workspace.model_validate_json(payload)
        return workspace.model_copy(update={
            "original_title": titles.get("", workspace.original_title),
            "conversations": tuple(session.model_copy(update={"title": titles[session.id]})
                                   if session.id in titles else session for session in workspace.conversations),
        })

    def has_title(self, conversation_id: str | None) -> bool:
        with closing(self._connect()) as db:
            return db.execute("SELECT 1 FROM conversation_titles WHERE id=?", (conversation_id or "",)).fetchone() is not None

    def set_title(self, conversation_id: str | None, title: str, *, source_request: str | None = None) -> bool:
        # 标题单独存储，后台改名不会推进学习状态版本或使正在生成的请求冲突。
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            workspace = Workspace.model_validate_json(db.execute("SELECT payload FROM workspace WHERE id=1").fetchone()[0])
            workspace.scope(conversation_id)
            key = conversation_id or ""
            if source_request is not None:
                source = db.execute("SELECT 1 FROM requests WHERE id=?", (source_request,)).fetchone()
                if source is None:
                    return False
                return db.execute("INSERT OR IGNORE INTO conversation_titles VALUES (?, ?)", (key, title)).rowcount == 1
            db.execute("INSERT OR REPLACE INTO conversation_titles VALUES (?, ?)", (key, title))
            return True

    @staticmethod
    def _replay(db, request_id: str, fingerprint: str) -> WorkspaceEvent | None:
        row = db.execute("SELECT fingerprint, event FROM requests WHERE id=?", (request_id,)).fetchone()
        if row is None:
            return None
        if row[0] != fingerprint:
            raise WorkspaceConflict("此请求编号已用于不同内容，请重新提交。")
        return WorkspaceEvent.model_validate_json(row[1])

    def replay(self, request_id: str, fingerprint: str) -> WorkspaceEvent | None:
        with closing(self._connect()) as db:
            return self._replay(db, request_id, fingerprint)

    def rename(self, title: str, request_id: str, fingerprint: str, event: WorkspaceEvent) -> WorkspaceEvent:
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            replay = self._replay(db, request_id, fingerprint)
            if replay is not None:
                return replay
            workspace = Workspace.model_validate_json(db.execute("SELECT payload FROM workspace WHERE id=1").fetchone()[0])
            workspace.scope(event.conversation_id)
            db.execute("INSERT OR REPLACE INTO conversation_titles VALUES (?, ?)", (event.conversation_id or "", title))
            db.execute("INSERT INTO requests VALUES (?, ?, ?)", (request_id, fingerprint, event.model_dump_json()))
        return event

    def commit(self, before: Workspace, after: Workspace, request_id: str,
               fingerprint: str, event: WorkspaceEvent, *, purge: str | None = None, rebase=None) -> WorkspaceEvent:
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            replay = self._replay(db, request_id, fingerprint)
            if replay is not None:
                return replay
            if rebase is not None:
                latest = Workspace.model_validate_json(db.execute("SELECT payload FROM workspace WHERE id=1").fetchone()[0])
                if latest.sequence != before.sequence:
                    after = rebase(latest)
                    before = latest
            published = after.model_copy(update={"sequence": before.sequence + 1, "events": (*after.events, event)})
            result = db.execute(
                "UPDATE workspace SET sequence=?, payload=? WHERE id=1 AND sequence=?",
                (published.sequence, published.model_dump_json(), before.sequence),
            )
            if result.rowcount != 1:
                raise WorkspaceConflict("操作期间工作区已变化，请核对最新状态后重试。")
            if purge:
                if purge == "all":
                    db.execute("DELETE FROM conversation_titles WHERE id=?", (event.conversation_id or "",))
                sql = "DELETE FROM requests WHERE json_extract(event, '$.conversation_id') IS ?"
                if purge == "history":
                    sql += " AND json_extract(event, '$.action') IN ('chat', 'draft', 'manual_plan', 'resubmit', 'decide')"
                db.execute(sql, (event.conversation_id,))
            db.execute("INSERT INTO requests VALUES (?, ?, ?)",
                       (request_id, fingerprint, event.model_dump_json()))
        return event

    def rebind_materials(self, before: Workspace, state: LearningState) -> None:
        """A material version change invalidates candidates drafted against older evidence."""
        if state == before.state:
            return
        event = WorkspaceEvent(action="materials", status="saved", message="资料关联已更新，请重新核对旧候选。")
        self.commit(before, before.model_copy(update={"state": state, "version": before.version + 1}),
                    uuid4().hex, state.model_dump_json(), event)
