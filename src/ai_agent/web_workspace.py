"""Loopback-only learning workspace backed by a persistent local database."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import secrets
import sqlite3
from urllib.parse import parse_qs, urlsplit

from pydantic import ValidationError

from ai_agent.config import resolve_project_path
from ai_agent.learning.workspace_service import WorkspaceCommand, WorkspaceService, workspace_view
from ai_agent.learning.workspace_store import WorkspaceConflict, WorkspaceStore
from ai_agent.persistence.session_materials import resolve_resume_materials
from ai_agent.persistence.state_store import StateSnapshot, load_learning_state
from ai_agent.runtime.agents import LLMInvocationError

ASSETS = Path(__file__).with_name("workspace_assets")
DEFAULT_WORKSPACE = ".agents_data/workspaces/study.sqlite3"
MAX_REQUEST_BYTES = 256_000
TASK_API_PATH = "/api/integration/tasks"
logger = logging.getLogger(__name__)


def create_server(service: WorkspaceService, port: int = 0) -> ThreadingHTTPServer:
    token = secrets.token_hex(32)
    integration_token = os.getenv("YANXI_TASK_API_TOKEN", "").encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)
            self.streaming = False
            self.stream_disconnected = False

        @property
        def authority(self):
            return f"127.0.0.1:{self.server.server_port}"

        def _send(self, status, body, content_type, *, download=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if download:
                self.send_header("Content-Disposition", 'attachment; filename="learning-state.json"')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # A retry can retrieve a committed receipt even if this response was lost.
                pass

        def _json(self, status, body):
            if self.streaming:
                self._stream_event("result", {"status": status, **body})
                return
            self._send(status, json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _stream_event(self, kind, data):
            if self.stream_disconnected:
                return
            try:
                self.wfile.write((json.dumps({"type": kind, "data": data}, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            except OSError:
                # 页面断开后允许已发出的操作完成，重连可按请求 ID 读取已保存结果。
                self.stream_disconnected = True

        def _authorized(self, *, write=False):
            if self.headers.get("Host") != self.authority:
                self._json(HTTPStatus.FORBIDDEN, {"error": "请使用终端显示的本地地址。"})
                return False
            if self.path.startswith("/api/"):
                if not write and urlsplit(self.path).path == TASK_API_PATH:
                    if not integration_token:
                        self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "任务查询接口未启用。"})
                        return False
                    supplied = self.headers.get("X-Yanxi-Task-Token", "").encode("utf-8")
                    if not secrets.compare_digest(supplied, integration_token):
                        self._json(HTTPStatus.FORBIDDEN, {"error": "任务查询凭据无效。"})
                        return False
                    return True
                supplied = self.headers.get("X-Workspace-Token", "").encode("utf-8")
                valid_token = secrets.compare_digest(supplied, token.encode("ascii"))
                if not valid_token or (write and self.headers.get("Origin") != f"http://{self.authority}"):
                    self._json(HTTPStatus.FORBIDDEN, {"error": "本地页面会话已失效，请刷新。"})
                    return False
            return True

        def do_GET(self):
            if not self._authorized():
                return
            assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                      "/markdown-it-15.0.3.min.js": ("markdown-it-15.0.3.min.js", "text/javascript"),
                      "/styles.css": ("styles.css", "text/css")}
            try:
                request = urlsplit(self.path)
                query = parse_qs(request.query)
                conversation_id = query.get("conversation", [None])[0]
                if self.path in assets:
                    name, mime = assets[self.path]
                    content = (ASSETS / name).read_text(encoding="utf-8")
                    if self.path == "/":
                        content = content.replace("__WORKSPACE_TOKEN__", token)
                    self._send(HTTPStatus.OK, content.encode("utf-8"), mime + "; charset=utf-8")
                elif request.path == "/api/state":
                    self._json(HTTPStatus.OK, workspace_view(service.store.load(), conversation_id))
                elif request.path == TASK_API_PATH:
                    selected = parse_qs(request.query, keep_blank_values=True).get("conversation", [])
                    if len(selected) != 1 or not re.fullmatch(r"original|[0-9a-f]{32}", selected[0]):
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "请指定 original 或有效的对话 ID。"})
                        return
                    try:
                        scope = service.store.load().scope(None if selected[0] == "original" else selected[0])
                    except WorkspaceConflict:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "对话不存在。"})
                        return
                    # 仅交付任务摘要。路径、聊天和反馈原文不进入服务间接口。
                    fields = {"task_id", "title", "activity_status", "understanding_status", "is_cancelled"}
                    self._json(HTTPStatus.OK, {"conversation": selected[0], "version": scope.version,
                        "tasks": [task.model_dump(mode="json", include=fields) for task in scope.state.tasks]})
                elif request.path == "/api/materials":
                    try:
                        catalog = service.material_catalog(conversation_id, offset=int(query.get("offset", ["0"])[0]))
                    except ValueError as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    else:
                        self._json(HTTPStatus.OK, catalog)
                elif request.path == "/api/export":
                    payload = StateSnapshot(state=service.store.load().scope(conversation_id).state).model_dump_json(indent=2)
                    self._send(HTTPStatus.OK, payload.encode("utf-8"), "application/json; charset=utf-8", download=True)
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "找不到此页面。"})
            except WorkspaceConflict as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except (OSError, sqlite3.Error, ValidationError) as exc:
                logger.warning("工作区读取失败：%s", type(exc).__name__)
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "无法读取工作区，请查看终端记录。"})

        def do_POST(self):
            if not self._authorized(write=True):
                return
            if self.path != "/api/action":
                self._json(HTTPStatus.NOT_FOUND, {"error": "找不到此操作。"})
                return
            if self.headers.get_content_type() != "application/json":
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "请提交JSON请求。"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "请求为空或过长。"})
                    return
                command = WorkspaceCommand.model_validate_json(self.rfile.read(length))
                observer = None
                if (command.action in {"chat", "draft"}
                        and self.headers.get("Accept") == "application/x-ndjson"):
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    self.streaming = True
                    observer = self._stream_event
                    observer("phase", "已收到请求，正在准备上下文")
                event = service.execute(command, stream_observer=observer, enforce_setup=True)
                selected = command.conversation_id
                if command.action == "new_conversation":
                    selected = event.conversation_id
                elif command.action == "delete_conversation":
                    selected = None
                title_pending = service.schedule_title(command, event)
                self._json(HTTPStatus.OK, {"event": event.model_dump(mode="json"), "title_pending": title_pending,
                                          "workspace": workspace_view(service.store.load(), selected)})
            except WorkspaceConflict as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except ValidationError as exc:
                locations = [".".join(map(str, error["loc"])) for error in exc.errors()]
                self._json(HTTPStatus.BAD_REQUEST, {"error": "请核对表单字段：" + "、".join(locations)})
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except LLMInvocationError as exc:
                logger.warning("工作区模型调用失败：%s", type(exc).__name__)
                self._json(HTTPStatus.BAD_GATEWAY, {"error": "模型请求未完成，请保留要求并重试。"})
            except (OSError, sqlite3.Error) as exc:
                logger.warning("工作区操作失败：%s", type(exc).__name__)
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "未能取得保存结果。请用同一请求重试或刷新核对。"})

        def log_message(self, format, *args):
            logger.debug("工作区 HTTP 请求")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=resolve_project_path(DEFAULT_WORKSPACE),
                        help=f"工作区SQLite路径，默认 {DEFAULT_WORKSPACE}")
    parser.add_argument("--state", type=Path, help="可选：将旧学习状态JSON导入新工作区")
    parser.add_argument("--materials", help="可选：关联资料目录，省略时沿用已保存关联")
    parser.add_argument("--archive-root", type=Path, default=resolve_project_path(".agents_data/history_archives"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        initial = None
        if args.state:
            if args.workspace.resolve() == args.state.resolve():
                parser.error("工作区路径必须与导入存档不同。")
            initial = load_learning_state(args.state)
            if args.materials is not None or initial.material_binding is not None:
                initial, notice = resolve_resume_materials(initial, args.materials)
                print(notice)
        store = WorkspaceStore(args.workspace, initial, create=True)
        if initial is None:
            current = store.load()
            if args.materials is not None or current.state.material_binding is not None:
                try:
                    state, notice = resolve_resume_materials(current.state, args.materials)
                except (ValueError, OSError):
                    if args.materials is not None:
                        raise
                    print("已保存的资料关联暂不可验证。可在网页中查看或更换目录。")
                else:
                    store.rebind_materials(current, state)
                    print(notice)
        print(f"本地数据库：{store.path}")
        service = WorkspaceService(store, args.archive_root.resolve())
        with create_server(service, args.port) as server:
            print(f"学习工作区 http://127.0.0.1:{server.server_port}/", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
