"""Turn-local PDF reuse and conversation-scoped citation snapshots."""

import json
import re
import sqlite3
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import closing
from threading import Lock

from langchain_core.tools import tool


class SharedPdfReads:
    def __init__(self, max_chars=48000):
        self._max_chars = max_chars
        self._chars = 0
        self._ready = OrderedDict()
        self._pending = {}
        self._lock = Lock()

    def read(self, path, source, start, end, fetch):
        def fingerprint(stat):
            return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

        try:
            stat = path.stat()
        except OSError:
            return fetch()
        version = fingerprint(stat)
        key = (path, source, version, start, end)
        with self._lock:
            if key in self._ready:
                self._ready.move_to_end(key)
                return self._ready[key], True
            pending = self._pending.get(key)
            owner = pending is None
            if owner:
                pending = self._pending[key] = Future()
        if not owner:
            raw, shareable = pending.result()
            return (raw, True) if shareable else fetch()
        try:
            raw, _ = fetch()
            payload = json.loads(raw)
            pages = payload.get("evidence", [])
            try:
                stable_source = fingerprint(path.stat()) == version
            except OSError:
                stable_source = False
            shareable = (stable_source and not payload.get("truncated")
                         and len(pages) == end - start + 1
                         and all(row["page"] == page and row.get("text") and not row.get("partial")
                                 for page, row in enumerate(pages, start)))
            with self._lock:
                if shareable and len(raw) <= self._max_chars:
                    self._ready[key] = raw
                    self._chars += len(raw)
                    while self._chars > self._max_chars:
                        _, removed = self._ready.popitem(last=False)
                        self._chars -= len(removed)
                pending.set_result((raw, shareable))
            return raw, False
        except Exception as error:
            pending.set_exception(error)
            raise
        finally:
            with self._lock:
                self._pending.pop(key, None)


class CitationArchive:
    def __init__(self, path, conversation_id, turn_id):
        self.path = path
        self.conversation_id = conversation_id
        self.turn_id = turn_id
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS citations (
                conversation TEXT, turn TEXT, evidence_id TEXT, source TEXT, page INTEGER,
                text TEXT, PRIMARY KEY (conversation, turn, evidence_id)
            )""")

    def save(self, evidence_id, row):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT OR IGNORE INTO citations VALUES (?, ?, ?, ?, ?, ?)", (
                self.conversation_id, self.turn_id, evidence_id, row["source"], row["page"], row["text"],
            ))

    def bind_read(self, scope):
        @tool
        def read_previous_citations(source: str, page: int, previous_turn: str | None = None) -> str:
            """回读同一会话先前实际引用的 PDF 正文。可指定原回答的轮次标识。"""
            if source not in scope or (scope[source] is not None and page not in scope[source]):
                return json.dumps({"error": "source_scope"})
            with closing(sqlite3.connect(self.path)) as db:
                rows = db.execute("""SELECT turn, evidence_id, text FROM citations
                    WHERE conversation=? AND turn<>? AND source=? AND page=?
                    AND (? IS NULL OR turn=?) ORDER BY rowid DESC LIMIT 3
                    """, (self.conversation_id, self.turn_id, source, page, previous_turn, previous_turn)).fetchall()
            return json.dumps({"historical_citations": [
                {"turn": turn, "evidence_id": key, "source": source, "page": page, "text": text}
                for turn, key, text in rows
            ]}, ensure_ascii=False)

        return read_previous_citations

    def render(self, answer, store, served, previous):
        def replace(match):
            key = match.group(1)
            if key not in served:
                if key in previous:
                    row = previous[key]
                    return f"[source={row['source']}, page={row['page']}]（历史引用）"
                return "【资料引用未核实】"
            row = store[key]
            if "page" not in row or "chunk_index" in row or row.get("partial") or not row.get("text"):
                return "【整页引用未核实】"
            self.save(key, row)
            return f"[source={row['source']}, page={row['page']}]"

        return re.sub(r"\[evidence=([^\[\]\r\n]+)\]", replace, answer)
