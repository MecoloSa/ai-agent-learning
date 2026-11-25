import asyncio
import json
import pytest

from langchain_core.messages import AIMessage

import ai_agent.main as app
from ai_agent.runtime import task_delegation
from ai_agent.runtime.source_delivery import TurnSourceLedger


def test_first_request_can_answer_without_starting_delegation(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("A direct reply must not inspect the materials or start another agent.")

    monkeypatch.setattr(task_delegation, "iter_material_files", forbidden)
    tool = task_delegation.build_learning_delegation_tool(str(tmp_path), "用取快递作类比")
    calls = []

    def respond(model, messages, **kwargs):
        calls.append(messages)
        return AIMessage(content="动作是取快递，观察是获得取件结果。")

    monkeypatch.setattr(app, "invoke_llm_response", respond)
    result = app._invoke_interactive_turn(
        request_messages=[], user_input="用取快递作类比", plain_llm=object(),
        tool_llm=object(), retrieval_tool=None, read_tool=None, extra_tools=(tool,),
    )
    assert result.startswith("动作是取快递")
    assert len(calls) == 1


@pytest.mark.parametrize("main_plans", [False, True])
@pytest.mark.parametrize("thinking", [None, False, True])
def test_delegation_returns_worker_evidence_without_second_synthesis(monkeypatch, tmp_path, main_plans, thinking):
    (tmp_path / "lesson.md").write_text("工具执行返回一个结果。", encoding="utf8")
    requests = []

    class Model:
        def __init__(self, role):
            self.role = role

        def bind_tools(self, tools, **options):
            assert all(not item.name.startswith("propose_") for item in tools)
            assert options == ({"extra_body": {"enable_thinking": thinking}} if thinking is not None else {})
            return self

        async def ainvoke(self, messages):
            requests.append(self.role)
            if self.role == "standard":
                data = json.loads(messages[-1].content)
                assert data["question"] == "读资料并解释"
                assert data["delegation_goal"] == "核对教材说法"
                assert data["authorized_scope"] == {"lesson.md": None}
                return AIMessage(content=json.dumps({"tasks": [
                    {"id": "read", "instruction": "核对工具返回什么", "sources": ["lesson.md"]},
                ]}))
            if requests.count("auxiliary") == 1:
                return AIMessage(content="", tool_calls=[{"id": "read-1", "name": "read_material_lines",
                    "args": {"relative_path": "lesson.md", "start_line": 1, "end_line": 1}}])
            row = json.loads(messages[-1].content)["evidence"][0]
            return AIMessage(content="", tool_calls=[{"id": "report-1", "name": "submit_report",
                "args": {"findings": [{"statement": "工具会返回结果", "evidence_ids": [row["evidence_id"]]}],
                         "gaps": []}}])

    monkeypatch.setattr(task_delegation, "create_llm", lambda *, role: Model(role))
    tool = task_delegation.build_learning_delegation_tool(
        str(tmp_path), "读资料并解释", main_plans=main_plans, worker_thinking=thinking,
    )
    args = {"tasks": [{"id": "read", "instruction": "核对工具返回什么", "sources": ["lesson.md"]}]} if main_plans else {
        "objective": "核对教材说法",
    }
    if main_plans:
        invalid = json.loads(tool.invoke({"tasks": [{"id": "read", "instruction": "读取资料", "sources": ["unknown.md"]}]}))
        assert invalid["error"]["available_sources"] == ["lesson.md"]
    raw = tool.invoke(args)
    result = json.loads(raw)
    assert result["ok"], result
    expected = ["auxiliary", "auxiliary"] if main_plans else ["standard", "auxiliary", "auxiliary"]
    assert requests == expected
    assert result["data"]["reports"]["read"]["status"] == "done"
    assert "excerpt_note" not in result["data"]
    assert len(result["data"]["evidence_catalog"]) == 1
    assert "工具执行返回一个结果。" not in raw
    ledger = TurnSourceLedger()
    ledger.observe(tool.name, raw, executed=True)
    assert "lesson.md" in ledger.render()
    rejected = json.loads(tool.invoke(args))
    assert rejected["error"]["code"] == "delegation_already_started"
    assert requests == expected


def test_routing_hint_is_only_in_initial_model_request(monkeypatch, tmp_path):
    requests = []
    seen = []
    shared_messages = []
    tool = task_delegation.build_learning_delegation_tool(str(tmp_path), "需要协作核对")

    def respond(model, messages, **kwargs):
        requests.append(list(messages))
        if len(requests) == 1:
            return AIMessage(content="", tool_calls=[{"id": "delegate-1", "name": tool.name,
                "args": {"objective": "核对资料"}}])
        seen.append(json.loads(messages[-1].content)["error"]["code"])
        return AIMessage(content="没有可读资料。")

    monkeypatch.setattr(app, "invoke_llm_response", respond)
    answer = app._invoke_interactive_turn(
        request_messages=shared_messages, user_input="需要协作核对", plain_llm=object(),
        tool_llm=object(), retrieval_tool=None, extra_tools=(tool,),
    )
    assert answer == "没有可读资料。"
    assert seen == ["no_readable_materials"]
    hint = task_delegation.DELEGATION_ROUTING_HINT
    assert [sum(hint in str(message.content) for message in batch) for batch in requests] == [1, 0]
    assert all(hint not in str(message.content) for message in shared_messages)


def test_delegate_deadline_keeps_evidence_already_read(monkeypatch, tmp_path):
    (tmp_path / "lesson.md").write_text("已有正文。", encoding="utf8")
    requests = []

    class Model:
        def bind(self, *, extra_body):
            assert extra_body == {"enable_thinking": False}
            return self

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            requests.append(messages)
            if len(requests) == 1:
                return AIMessage(content=json.dumps({"tasks": [
                    {"id": "read", "instruction": "核对资料", "sources": ["lesson.md"]},
                ]}))
            if len(requests) == 2:
                return AIMessage(content="", tool_calls=[{"id": "read-1", "name": "read_material_lines",
                    "args": {"relative_path": "lesson.md", "start_line": 1, "end_line": 1}}])
            await asyncio.sleep(30)

    monkeypatch.setattr(task_delegation, "create_llm", lambda **_: Model())
    monkeypatch.setattr(task_delegation, "DELEGATION_TIMEOUT_SECONDS", 4)
    tool = task_delegation.build_learning_delegation_tool(str(tmp_path), "读资料")
    result = json.loads(tool.invoke({"objective": "核对资料"}))
    assert result["ok"], result
    assert result["data"]["reports"]["read"]["status"] == "unavailable"
    assert len(result["data"]["evidence_catalog"]) == 1
    evidence = next(iter(result["data"]["evidence_catalog"].values()))
    assert "已有正文。" in evidence["excerpt"]
    assert evidence["excerpt_truncated"] is False
    assert "已读原文的节选" in result["data"]["excerpt_note"]


def test_fast_report_reuses_complete_tool_history_and_preserves_scope(monkeypatch, tmp_path):
    (tmp_path / "lesson.md").write_text("第一条已读证据。\n第二条已读证据。", encoding="utf8")
    heads = []

    class Model:
        def __init__(self, role):
            self.role = role
            self.fast = False

        def bind_tools(self, tools):
            return self

        def bind(self, *, extra_body):
            assert extra_body == {"enable_thinking": False}
            self.fast = True
            return self

        async def ainvoke(self, messages):
            if self.role == "standard":
                return AIMessage(content=json.dumps({"tasks": [
                    {"id": "read", "instruction": "对照两行", "sources": ["lesson.md"]},
                ]}))
            if self.fast:
                data = json.loads(messages[-1].content)
                assert data["authorized_scope"] == {"lesson.md": None}
                assert data["assignment"]["instruction"] == "对照两行"
                results = [item["result"]["evidence"][0] for item in data["tool_history"] if "result" in item]
                assert [item["text"] for item in results] == ["第一条已读证据。", "第二条已读证据。"]
                assert data["pending_tool_call_ids"] == []
                return AIMessage(content=json.dumps({"findings": [{
                    "statement": "已核对两行", "evidence_ids": [item["evidence_id"] for item in results],
                }], "gaps": []}))
            heads.append(1)
            line = len(heads)
            if line <= 2:
                return AIMessage(content="", tool_calls=[{"id": f"read-{line}", "name": "read_material_lines",
                    "args": {"relative_path": "lesson.md", "start_line": line, "end_line": line}}])
            await asyncio.sleep(30)

    monkeypatch.setattr(task_delegation, "create_llm", lambda *, role: Model(role))
    monkeypatch.setattr(task_delegation, "WORKER_TIMEOUT_SECONDS", 2.5)
    monkeypatch.setattr(task_delegation, "WORKER_REPORT_RESERVE_SECONDS", 1.3)
    tool = task_delegation.build_learning_delegation_tool(str(tmp_path), "核对两行")
    receipt = json.loads(tool.invoke({"objective": "读完两行"}))
    assert receipt["data"]["reports"]["read"]["status"] == "done"
    assert len(receipt["data"]["evidence_catalog"]) == 2


def test_partial_excerpts_use_confirmed_ids_and_respect_limits():
    from ai_agent.runtime.delegated_reading import attach_partial_excerpts

    tasks = [{"id": f"task-{i}", "instruction": "阅读 tool_call_id"}
             for i in range(4)]
    items = {
        f"{i}:{page}": {"evidence_id": f"{i}:{page}", "source": f"{i}.pdf", "page": page,
                         "text": "tool_call_id " + "文" * 2000}
        for i in range(4) for page in range(3)
    }
    items["unconfirmed"] = {"evidence_id": "unconfirmed", "source": "outside.pdf", "page": 1,
                            "text": "tool_call_id " * 50}
    items["duplicate-page"] = {**items["0:0"], "evidence_id": "duplicate-page"}
    results = {task["id"]: {
        "status": "done" if i == 3 else "unavailable",
        "evidence_ids": [f"{i}:{j}" for j in range(3)] + (["duplicate-page"] if i == 0 else []),
    } for i, task in enumerate(tasks)}
    catalog = {key: {"page": item["page"]} for key, item in items.items() if key != "0:2"}

    attach_partial_excerpts(catalog, tasks, results, items)

    excerpts = {key: evidence["excerpt"] for key, evidence in catalog.items() if "excerpt" in evidence}
    assert set(excerpts) == {"0:0", "0:1", "1:0", "1:1", "2:0"}
    assert sum(map(len, excerpts.values())) == 6000
    assert all(catalog[key]["excerpt_truncated"] for key in excerpts)
    assert all(items[key]["text"].startswith(text) for key, text in excerpts.items())
