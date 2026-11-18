"""Local HTTP, durable operations and terminal compatibility."""

from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import sys
from threading import Event, Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from langchain_core.messages import AIMessage
import pytest

import ai_agent.main as app
import ai_agent.web_workspace as web
from ai_agent.learning.learning_state import LearningState, LearningTask
from ai_agent.learning.plan_change_request import PlanDraftResult
from ai_agent.learning.workspace_service import WorkspaceCommand, WorkspaceService, workspace_view
from ai_agent.learning.workspace_store import WorkspaceConflict, WorkspaceStore
from ai_agent.persistence.session_materials import create_material_binding
from ai_agent.persistence.state_store import StateSnapshot, load_learning_state
from ai_agent.web_workspace import create_server

INSTRUCTIONS = "改为阅读5分钟、口头说明10分钟，提交完整候选。"
FEEDBACK = "阅读完成，但还分不清两类工具。"


@pytest.fixture
def state(tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "notes.md").write_text("阅读工具定义。\n比较通用工具和专用工具。\n口头说明差异。", encoding="utf-8")
    return LearningState(learning_goal="理解工具边界", plan_markdown="阅读30分钟",
        tasks=(LearningTask(task_id="read-001", title="阅读工具定义"),),
        material_binding=create_material_binding(str(materials)))


def command(action, version, **payload):
    return WorkspaceCommand(action=action, version=version, request_id=uuid4().hex, payload=payload)


@contextmanager
def serving(service):
    with create_server(service) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=5)


def fetch(base, path, *, token=None, origin=None, data=None, accept=None):
    headers = {"X-Workspace-Token": token} if token else {}
    if origin:
        headers["Origin"] = origin
    if accept:
        headers["Accept"] = accept
    body = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode()
    request = Request(base + path, data=body, headers=headers)
    try:
        response = urlopen(request, timeout=15)
    except HTTPError as exc:
        response = exc
    with response:
        return response.code, response.read().decode("utf-8")


def test_startup_creates_default_database_and_restores_without_json(tmp_path, monkeypatch):
    monkeypatch.setattr("ai_agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["web_workspace"])
    opened = []

    @contextmanager
    def server(service, port):
        opened.append(service)
        yield type("Server", (), {"server_port": port, "serve_forever": lambda self: None})()

    monkeypatch.setattr(web, "create_server", server)
    web.main()
    first = opened[-1]
    path = tmp_path / ".agents_data/workspaces/study.sqlite3"
    assert first.store.path == path and path.is_file()
    assert first.store.load().state.material_binding is None
    first.execute(command("create_task", 1, title="阅读工具定义"))

    web.main()
    assert opened[-1].store.load().state.tasks[0].title == "阅读工具定义"
    assert not list(tmp_path.rglob("*.json"))

    directory = tmp_path / "local materials"
    directory.mkdir()
    (directory / "notes.md").write_text("学习资料", encoding="utf-8")
    opened[-1].execute(command("bind_materials", 2, directory=str(directory)))
    directory.rename(tmp_path / "moved materials")
    web.main()
    assert workspace_view(opened[-1].store.load())["materials"]["available"] is False
    assert opened[-1].store.load().state.tasks[0].title == "阅读工具定义"

    snapshot = tmp_path / "legacy.json"
    legacy = LearningState(learning_goal="旧存档目标", plan_markdown="旧存档计划")
    snapshot.write_text(StateSnapshot(state=legacy).model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["web_workspace", "--state", str(snapshot)])
    with pytest.raises(SystemExit, match="2"):
        web.main()
    assert WorkspaceStore(path).load().state.tasks[0].title == "阅读工具定义"
    imported = tmp_path / "imported.sqlite3"
    monkeypatch.setattr(sys, "argv", ["web_workspace", "--workspace", str(imported), "--state", str(snapshot)])
    web.main()
    snapshot.unlink()
    assert WorkspaceStore(imported).load().state == legacy


def test_http_feedback_and_candidate_lifecycle_survive_restart(state, tmp_path):
    calls = []

    def draft(workspace, text, request):
        calls.append((text, request.mode))
        return workspace.state, PlanDraftResult("候选", "阅读5分钟，口头说明10分钟")

    path = tmp_path / "workspace.sqlite3"
    store = WorkspaceStore(path, state)
    service = WorkspaceService(store, tmp_path, draft_turn=draft)
    with serving(service) as base:
        status, html = fetch(base, "/")
        assert status == 200 and "调整计划" in html
        token = re.search(r'name="workspace-token" content="([0-9a-f]+)"', html)[1]
        assert fetch(base, "/app.js")[0] == 200
        assert fetch(base, "/api/state")[0] == 403

        def post(request):
            code, body = fetch(base, "/api/action", token=token, origin=base, data=request.model_dump(mode="json"))
            return code, json.loads(body)

        feedback = command("feedback", 1, task_id="read-001", text=FEEDBACK,
                           activity_status="completed", understanding_status="self_reported_unclear")
        assert fetch(base, "/api/action", token=token, origin="http://example.invalid",
                     data=feedback.model_dump())[0] == 403
        status, body = post(feedback)
        assert status == 200 and body["workspace"]["version"] == 2
        assert body["workspace"]["state"]["tasks"][0]["understanding_status"] == "self_reported_unclear"
        assert post(feedback)[0] == 200
        assert len(store.load().state.feedback_records) == 1 and calls == []
        assert post(command("feedback", 1, task_id="read-001", text="旧页面"))[0] == 409
        status, body = post(command("draft", 2, mode="adjust", instructions=INSTRUCTIONS))
        assert status == 200
        first = body["event"]["proposal_id"]
        assert body["event"]["status"] == "pending"
        assert body["workspace"]["state"]["plan_markdown"] == "阅读30分钟"
        assert "-阅读30分钟" in body["workspace"]["proposals"][0]["diff"]
        assert post(command("decide", 2, proposal_id=first, decision="reject"))[0] == 200
        assert store.load().state.context_revision == 1
        status, body = post(command("resubmit", 2, proposal_id=first))
        second = body["event"]["proposal_id"]
        assert status == 200 and first != second
        adopt = command("decide", 2, proposal_id=second, decision="adopt")
        assert post(adopt)[0] == 200 and post(adopt)[0] == 200
        assert post(command("decide", 3, proposal_id=second, decision="adopt"))[0] == 409
        assert fetch(base, "/api/export", token=token)[0] == 200

    restored = WorkspaceStore(path).load()
    assert restored.version == 3 and restored.state.context_revision == 2
    assert restored.state.plan_markdown == "阅读5分钟，口头说明10分钟"
    assert restored.state.feedback_records[0].quote == FEEDBACK
    assert [p.status for p in restored.proposals] == ["rejected", "adopted"]
    assert restored.proposals[-1].previous_id == first
    assert calls == [(INSTRUCTIONS, "adjust")]


def test_task_integration_is_read_only_and_scoped(state, tmp_path, monkeypatch):
    monkeypatch.setenv("YANXI_TASK_API_TOKEN", "test-integration-token")
    store = WorkspaceStore(tmp_path / "workspace.sqlite3", state)
    service = WorkspaceService(store, tmp_path)
    session = service.execute(command("new_conversation", 1)).conversation_id
    create = command("create_task", 1, title="第二个对话的任务")
    service.execute(create.model_copy(update={"conversation_id": session}))
    before = store.load()

    with serving(service) as base:
        def query(conversation, token="test-integration-token"):
            request = Request(f"{base}/api/integration/tasks?conversation={conversation}",
                              headers={"X-Yanxi-Task-Token": token})
            try:
                response = urlopen(request, timeout=10)
            except HTTPError as error:
                response = error
            with response:
                return response.code, json.load(response)

        code, original = query("original")
        assert code == 200 and [item["title"] for item in original["tasks"]] == ["阅读工具定义"]
        code, other = query(session)
        assert code == 200 and [item["title"] for item in other["tasks"]] == ["第二个对话的任务"]
        assert set(other["tasks"][0]) == {"task_id", "title", "activity_status", "understanding_status", "is_cancelled"}
        assert query(session, "wrong")[0] == 403
        assert query("f" * 32)[0] == 404
        assert query("")[0] == 400
        assert fetch(base, "/api/state", token="test-integration-token")[0] == 403
    assert store.load() == before


def test_web_can_bind_list_and_read_local_material(state, tmp_path, monkeypatch):
    path = tmp_path / "workspace.sqlite3"
    store = WorkspaceStore(path, create=True)
    directory = Path(state.material_binding.root_dir)
    (directory / "中文笔记.txt").write_text("本地文件验收编号 3617\n", encoding="utf-8")
    monkeypatch.setattr(app, "create_llm", lambda: object())
    monkeypatch.setattr(app, "create_llm_with_tools", lambda _: object())
    calls = []
    title_started = Event()
    allow_title = Event()
    title_finished = Event()
    original_set_title = store.set_title

    def set_title(*args, **kwargs):
        try:
            return original_set_title(*args, **kwargs)
        finally:
            if kwargs.get("source_request"):
                title_finished.set()

    class TitleModel:
        def invoke(self, messages):
            assert "编号是 3617" in messages[-1].content
            title_started.set()
            assert allow_title.wait(5)
            return AIMessage(content="验收编号总结")

    def title_model(*, role):
        assert role == "workspace_title"
        return TitleModel()

    monkeypatch.setattr("ai_agent.config.create_llm", title_model)
    monkeypatch.setattr(store, "set_title", set_title)

    def model(llm, messages, *, stage, stream_observer=None):
        calls.append(stage)
        if len(calls) == 1:
            return AIMessage(content="", tool_calls=[{"name": "read_material_lines", "id": "read",
                "args": {"relative_path": "中文笔记.txt", "start_line": 1, "end_line": 1}}])
        assert any("本地文件验收编号 3617" in str(item.content) for item in messages)
        return AIMessage(content="编号是 3617。")

    monkeypatch.setattr(app, "invoke_llm_response", model)
    with serving(WorkspaceService(store, tmp_path)) as base:
        token = re.search(r'name="workspace-token" content="([0-9a-f]+)"', fetch(base, "/")[1])[1]

        def post(action, version, **payload):
            return fetch(base, "/api/action", token=token, origin=base,
                         data=command(action, version, **payload).model_dump())

        assert fetch(base, "/api/materials")[0] == 403
        assert fetch(base, "/api/materials", token=token)[0] == 400
        assert post("bind_materials", 1, directory="relative/path")[0] == 400
        assert post("bind_materials", 1, directory=str(directory / "missing"))[0] == 400
        assert not store.load().events
        code, _ = post("bind_materials", 1, directory=str(directory))
        assert code == 200 and store.load().version == 2
        code, catalog = fetch(base, "/api/materials", token=token)
        assert code == 200
        assert {"中文笔记.txt", "notes.md"} == {
            item["relative_path"] for item in json.loads(catalog)["data"]["files"]
        }
        code, answer = post("chat", 2, text="请读取中文笔记.txt 第1行，告诉我验收编号。")
        assert code == 200 and "编号是 3617" in json.loads(answer)["event"]["message"]
        calls.clear()
        code, stream = fetch(base, "/api/action", token=token, origin=base,
            accept="application/x-ndjson", data=command("chat", store.load().version,
                text="请再次读取中文笔记.txt 第1行并总结。").model_dump())
        events = [json.loads(line) for line in stream.splitlines()]
        assert code == 200
        assert [event["data"] for event in events if event["type"] == "tool_activity"] == [
            {"id": "read", "name": "read_material_lines", "status": "running"},
            {"id": "read", "name": "read_material_lines", "status": "completed"},
        ]
        assert events[-1]["type"] == "result"
        assert events[-1]["data"]["title_pending"]
        assert title_started.wait(2) and not title_finished.is_set()
        current_version = store.load().version
        allow_title.set()
        assert title_finished.wait(2)
        assert workspace_view(store.load())["conversations"][0]["title"] == "验收编号总结"
        assert store.load().version == current_version
        assert post("rename_conversation", current_version, title="手动命名")[0] == 200
        assert workspace_view(store.load())["conversations"][0]["title"] == "手动命名"
        assert post("clear_history", current_version, confirm=True)[0] == 200
        cleared = workspace_view(WorkspaceStore(path).load())
        assert not cleared["recent_chat"]
        assert not any(event["action"] == "chat" for event in cleared["events"])
        assert cleared["conversations"][0]["title"] == "手动命名"

    assert WorkspaceStore(path).load().state.material_binding.root_dir == str(directory)
    assert store.load().state.context_revision == 1


def test_confirmed_plan_tasks_edits_extraction_and_split(tmp_path, monkeypatch):
    path = tmp_path / "workspace.db"
    store = WorkspaceStore(path, create=True)
    service = WorkspaceService(store, tmp_path)
    roles = []
    monkeypatch.setattr(app, "create_llm", lambda *, role="standard": roles.append(role) or object())
    monkeypatch.setattr(app, "create_llm_with_tools", lambda _, **kwargs: object())
    current = "阅读并口头说明工具边界。"
    answers = [
        (current, [{"operation": "add", "title": "阅读工具定义", "estimated_minutes": 5},
                   {"operation": "add", "title": "口头说明区别", "estimated_minutes": 10}]),
    ]
    requests = []

    def model(llm, messages, *, stage):
        requests.append(messages)
        plan, changes = answers.pop(0)
        return AIMessage(content="", tool_calls=[{"name": "propose_plan_change", "id": "proposal",
                         "args": {"plan_markdown": plan, "task_changes": changes}}])

    monkeypatch.setattr(app, "invoke_llm_response", model)

    def execute(action, **data):
        return service.execute(command(action, store.load().version, **data))

    first = execute("draft", mode="replace", instructions="安排阅读5分钟和口头说明10分钟。")
    assert first.status == "pending" and not store.load().state.tasks
    execute("decide", proposal_id=first.proposal_id, decision="reject")
    assert not store.load().state.tasks
    again = execute("resubmit", proposal_id=first.proposal_id)
    execute("decide", proposal_id=again.proposal_id, decision="adopt")
    reading, speaking = store.load().state.tasks
    assert reading.from_plan and speaking.from_plan and speaking.estimated_minutes == 10
    execute("edit_task", task_id=speaking.task_id, activity_status="in_progress",
            understanding_status="self_reported_partial", text="能区分定义，但举不出例子。")
    with pytest.raises(ValueError):
        execute("edit_task", task_id=speaking.task_id, from_plan=False)

    adjusted = "阅读并口头说明工具边界。口头说明延长到15分钟。"
    answers.append((adjusted, [{"operation": "update", "task_id": speaking.task_id, "estimated_minutes": 15}]))
    draft = execute("draft", mode="adjust", instructions="口头说明延长到15分钟，保留已有反馈。")
    assert any("recent_committed_task_changes" in str(message.content)
               and "self_reported_partial" in str(message.content) for message in requests[-1])
    execute("decide", proposal_id=draft.proposal_id, decision="adopt")
    restarted = WorkspaceStore(path).load().state
    assert restarted.tasks[1].task_id == speaking.task_id and restarted.tasks[1].estimated_minutes == 15
    assert restarted.tasks[1].activity_status == "in_progress"
    assert restarted.tasks[1].latest_feedback == "能区分定义，但举不出例子。"

    answers.append((adjusted, [{"operation": "add", "title": "阅读工具定义"}]))
    duplicate = execute("draft", mode="extract_tasks", instructions="提取已有正式计划里的任务。")
    assert duplicate.status == "not_submitted"
    answers.append((adjusted, [{"operation": "add", "title": "复习工具边界"}]))
    extracted = execute("draft", mode="extract_tasks", instructions="补充登记复习活动。")
    execute("decide", proposal_id=extracted.proposal_id, decision="adopt")
    assert store.load().state.tasks[-1].title == "复习工具边界"

    answers.append((adjusted, [{"operation": "archive", "task_id": speaking.task_id},
                               {"operation": "add", "title": "准备例子"},
                               {"operation": "add", "title": "用例子口头说明"}]))
    split = execute("draft", mode="split_task", task_id=speaking.task_id, instructions="拆成准备和说明两项。")
    execute("decide", proposal_id=split.proposal_id, decision="adopt")
    saved = WorkspaceStore(path).load()
    restored = saved.state
    assert restored.plan_markdown == adjusted and restored.context_revision == restarted.context_revision
    assert restored.tasks[1].is_cancelled and restored.tasks[1].latest_feedback == "能区分定义，但举不出例子。"
    assert [task.title for task in restored.tasks[2:]] == ["准备例子", "用例子口头说明", "复习工具边界"]
    assert all(task.activity_status == "not_started" for task in restored.tasks[2:4])
    # 模拟旧版把子任务追加到末尾。加载会用采用回执恢复顺序。
    legacy_tasks = (*restored.tasks[:2], restored.tasks[-1], *restored.tasks[2:4])
    legacy = saved.model_copy(update={"state": restored.model_copy(update={"tasks": legacy_tasks})})
    assert legacy.scope(None).state.tasks == restored.tasks
    assert restored.feedback_records[-1].task_id == speaking.task_id
    assert roles == ["auxiliary", "auxiliary", "workspace_tasks", "workspace_tasks", "workspace_tasks"]


def test_web_conversations_isolate_model_memory_and_preserve_original(state, tmp_path, monkeypatch):
    path = tmp_path / "workspace.sqlite3"
    store = WorkspaceStore(path, state)
    service = WorkspaceService(store, tmp_path)
    seen = []
    monkeypatch.setattr(app, "create_llm", lambda: object())
    monkeypatch.setattr(app, "create_llm_with_tools", lambda _: object())

    def reply(llm, messages, *, stage):
        seen.append([str(message.content) for message in messages])
        return AIMessage(content="已收到。")

    monkeypatch.setattr(app, "invoke_llm_response", reply)

    def chat(text, conversation_id=None):
        request = command("chat", 1, text=text).model_copy(update={"conversation_id": conversation_id})
        return service.execute(request)

    chat("原始对话独有记号")
    session = service.execute(command("new_conversation", 1)).conversation_id
    assert chat("新对话独有记号", session).conversation_id == session
    assert all("原始对话独有记号" not in message for message in seen[-1])
    chat("继续新对话", session)
    assert any("新对话独有记号" in message for message in seen[-1])
    chat("继续原始对话")
    assert all("新对话独有记号" not in message for message in seen[-1])
    saved = WorkspaceStore(path).load()
    assert saved.version == 1 and saved.state.tasks == state.tasks
    assert workspace_view(saved)["conversations"][-1] == {"id": session, "title": "新对话独有记号"}
    assert saved.conversations[0].memory is not None


def test_conversation_scope_and_removal_leave_other_conversations_intact(state, tmp_path):
    path = tmp_path / "workspace.db"
    store = WorkspaceStore(path, state)
    service = WorkspaceService(store, tmp_path)
    session = service.execute(command("new_conversation", 1)).conversation_id

    def scoped(action, version, **data):
        request = command(action, version, **data).model_copy(update={"conversation_id": session})
        return service.execute(request)

    assert not workspace_view(store.load(), session)["state"]["tasks"]
    original = service.execute(command("manual_plan", 1, plan_markdown="原始对话的候选"))
    with pytest.raises(ValueError, match="找不到指定候选"):
        scoped("decide", 1, proposal_id=original.proposal_id, decision="adopt")
    proposal = scoped("manual_plan", 1, plan_markdown="# 第二对话计划")
    scoped("decide", 1, proposal_id=proposal.proposal_id, decision="adopt")
    scoped("create_task", 2, title="第二对话任务")
    scoped("clear_history", 3, confirm=True)
    reopened = WorkspaceStore(path).load()
    assert reopened.scope(session).state.plan_markdown == "# 第二对话计划"
    assert reopened.scope(session).state.tasks[0].title == "第二对话任务"
    assert not reopened.scope(session).proposals and len(reopened.scope(None).proposals) == 1
    assert {event.action for event in reopened.scope(session).events} == {"new_conversation", "create_task", "clear_history"}
    assert reopened.scope(None).state == state
    scoped("delete_conversation", 4, confirm=True)
    with pytest.raises(WorkspaceConflict):
        WorkspaceStore(path).load().scope(session)
    assert store.load().state == state
    service.execute(command("delete_conversation", 1, confirm=True))
    fresh = workspace_view(store.load())
    assert fresh["conversation_id"] and len(fresh["conversations"]) == 1
    assert fresh["state"]["tasks"] == []


def test_web_material_removal_preserves_bytes_and_rejects_outside_path(state, tmp_path, monkeypatch):
    store = WorkspaceStore(tmp_path / "workspace.sqlite3", state)
    service = WorkspaceService(store, tmp_path)
    source = Path(state.material_binding.root_dir) / "notes.md"
    original = source.read_bytes()
    for path in ("../outside.md", str(source)):
        with pytest.raises(ValueError):
            service.execute(command("remove_material", 1, relative_path=path, confirm=True))
    with monkeypatch.context() as scoped:
        def fail_commit(*args, **kwargs):
            raise sqlite3.OperationalError("test")
        scoped.setattr(store, "commit", fail_commit)
        with pytest.raises(sqlite3.OperationalError):
            service.execute(command("remove_material", 1, relative_path="notes.md", confirm=True))
    assert source.read_bytes() == original
    event = service.execute(command("remove_material", 1, relative_path="notes.md", confirm=True))
    saved = list(source.parent.glob(".yanxi-trash/*/notes.md.removed"))
    assert event.status == "saved" and not source.exists() and saved[0].read_bytes() == original
    assert store.load().state.material_binding is None


def test_stale_candidate_and_cancelled_task_progress_are_rejected(state, tmp_path):
    store = WorkspaceStore(tmp_path / "workspace.db", state)
    service = WorkspaceService(store, tmp_path, draft_turn=lambda w, *_: (w.state, PlanDraftResult("候选", "新计划")))
    draft = service.execute(command("draft", 1, mode="replace", instructions=INSTRUCTIONS))
    service.execute(command("cancel_task", 1, task_id="read-001", text="本次暂停"))
    assert workspace_view(store.load())["proposals"][0]["status"] == "stale"
    with pytest.raises(WorkspaceConflict):
        service.execute(command("decide", 2, proposal_id=draft.proposal_id, decision="adopt"))
    with pytest.raises(ValueError, match="任务已取消"):
        service.execute(command("feedback", 2, task_id="read-001", text=FEEDBACK, activity_status="completed"))
    assert store.load().state.plan_markdown == state.plan_markdown


def test_manual_json_plan_is_validated_and_requires_adoption(state, tmp_path):
    store = WorkspaceStore(tmp_path / "workspace.db", state)
    service = WorkspaceService(store, tmp_path, draft_turn=lambda *_: pytest.fail("manual edit must not call a model"))
    with pytest.raises(ValueError):
        service.execute(command("manual_plan", 1, plan_markdown="新计划", context_revision=99))
    with pytest.raises(ValueError, match="更新标记"):
        service.execute(command("manual_plan", 1, plan_markdown=app.GOAL_START_TAG + "越权"))
    with pytest.raises(ValueError, match="相同"):
        service.execute(command("manual_plan", 1, plan_markdown=state.plan_markdown))
    assert not store.load().events

    draft = service.execute(command("manual_plan", 1, plan_markdown="阅读5分钟，口头说明10分钟"))
    assert draft.status == "pending" and store.load().state == state
    restored = WorkspaceService(WorkspaceStore(store.path), tmp_path)
    restored.execute(command("decide", 1, proposal_id=draft.proposal_id, decision="adopt"))
    assert store.load().state.context_revision == 2
    assert store.load().state.plan_markdown == "阅读5分钟，口头说明10分钟"
    assert store.load().state.tasks == state.tasks


def test_concurrent_change_during_generation_does_not_publish_stale_proposal(state, tmp_path):
    store = WorkspaceStore(tmp_path / "workspace.db", state)

    def draft(workspace, *_):
        service.execute(command("feedback", 1, task_id="read-001", text=FEEDBACK))
        return workspace.state, PlanDraftResult("候选", "过时计划")

    service = WorkspaceService(store, tmp_path, draft_turn=draft)
    with pytest.raises(WorkspaceConflict, match="操作期间"):
        service.execute(command("draft", 1, mode="adjust", instructions=INSTRUCTIONS))
    saved = store.load()
    assert not saved.proposals and len(saved.state.feedback_records) == 1
    assert saved.state.plan_markdown == state.plan_markdown


def test_concurrent_chat_in_same_scope_does_not_overwrite_history(state, tmp_path):
    store = WorkspaceStore(tmp_path / "workspace.db", state)
    inner = command("chat", 1, text="先保存的追问")

    def draft(workspace, text, _):
        if text == "较早开始的提问":
            service.execute(inner)
        return workspace.state, PlanDraftResult("回答", None)

    service = WorkspaceService(store, tmp_path, draft_turn=draft)
    with pytest.raises(WorkspaceConflict, match="当前对话已变化"):
        service.execute(command("chat", 1, text="较早开始的提问"))
    assert [event.input_text for event in store.load().events] == ["先保存的追问"]
    assert store.load().version == 1


def test_failed_receipt_write_rolls_back_feedback_and_can_retry(state, tmp_path):
    store = WorkspaceStore(tmp_path / "workspace.db", state)
    service = WorkspaceService(store, tmp_path)
    request = command("feedback", 1, task_id="read-001", text=FEEDBACK)
    with sqlite3.connect(store.path) as db:
        db.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON requests BEGIN SELECT RAISE(ABORT,'test'); END")
    with pytest.raises(sqlite3.Error):
        service.execute(request)
    assert store.load().state == state and not store.load().events
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TRIGGER fail_receipt")
    assert service.execute(request).status == "saved"
    assert len(store.load().state.feedback_records) == 1


@pytest.mark.parametrize(("submit", "bound"), [(False, True), (True, True), (True, False)])
def test_web_model_turn_uses_real_tool_loop_and_only_plan_write(state, tmp_path, monkeypatch, submit, bound):
    store = WorkspaceStore(tmp_path / "workspace.db", state if bound else None, create=True)
    service = WorkspaceService(store, tmp_path)
    tools = []
    monkeypatch.setattr(app, "create_llm", lambda *, role: object())
    monkeypatch.setattr(app, "create_llm_with_tools",
                        lambda items, *, role: tools.extend(t.name for t in items) or object())

    def model(llm, messages, *, stage):
        if submit:
            return AIMessage(content="", tool_calls=[{"name": "propose_plan_change", "id": "candidate",
                "args": {"plan_markdown": "阅读5分钟，口头说明10分钟", "task_changes": []}}])
        return AIMessage(content="请说明你希望保留哪项活动。")

    monkeypatch.setattr(app, "invoke_llm_response", model)
    event = service.execute(command("draft", 1, mode="replace", instructions=INSTRUCTIONS))
    material_tools = {"read_material_lines", "read_pdf_pages", "list_material_files", "retrieve_learning_context"}
    if bound:
        assert {"read_material_lines", "read_pdf_pages"} <= set(tools)
    else:
        assert not material_tools & set(tools)
    assert "propose_plan_change" in tools
    assert not {"propose_learning_feedback", "propose_task_creation", "propose_task_cancellation"} & set(tools)
    assert event.status == ("pending" if submit else "not_submitted")
    assert store.load().state.context_revision == 1
    if not submit:
        assert "保留哪项活动" in event.message and not store.load().proposals


def test_cli_explicit_entry_blocks_legacy_text_and_accepts_real_candidate(state, tmp_path, monkeypatch, capsys):
    inputs = iter(["/plan-adjust", "/plan-adjust " + INSTRUCTIONS, "/plan-replace " + INSTRUCTIONS, "y", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    roles = []
    monkeypatch.setattr(app, "create_llm", lambda *, role="standard": roles.append(role) or object())
    monkeypatch.setattr(app, "create_llm_with_tools",
                        lambda _, *, role="standard": roles.append(role) or object())
    answers = iter([
        AIMessage(content=app.PLAN_START_TAG + "未提交的文本" + app.PLAN_END_TAG),
        AIMessage(content="", tool_calls=[{"name": "propose_plan_change", "id": "candidate",
            "args": {"plan_markdown": "阅读5分钟，口头说明10分钟"}}]),
    ])
    monkeypatch.setattr(app, "invoke_llm_response", lambda *a, **k: next(answers))
    path = tmp_path / "current.json"
    app.run_interactive_session("", state.plan_markdown, "", state.learning_goal, "window",
                                state.material_binding.root_dir, initial_state=state, current_state_path=path)
    saved = load_learning_state(path)
    assert roles == ["standard", "standard", "auxiliary", "auxiliary"]
    assert saved.context_revision == 2 and saved.plan_markdown == "阅读5分钟，口头说明10分钟"
    assert "本轮未生成可确认的计划" in capsys.readouterr().out
