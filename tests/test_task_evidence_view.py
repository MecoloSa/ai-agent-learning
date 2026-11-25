import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.helpers.delegated_reading import project_task_history
from ai_agent.runtime import delegated_reading as probe
from tests.helpers.langgraph_reading_compare import BOOKS


def add_receipt(history, key, text, *, tool="read_pdf", page=1):
    call_id = f"call-{len(history)}"
    history.extend([
        AIMessage(content="", tool_calls=[{"id": call_id, "name": tool, "args": {}}]),
        ToolMessage(tool_call_id=call_id, content=json.dumps({"evidence": [
            {"evidence_id": key, "page": page, "text": text},
        ]})),
    ])


def rows(messages):
    return [json.loads(item.content)["evidence"][0] for item in messages if isinstance(item, ToolMessage)]


def test_continuous_view_keeps_multiple_pages_and_exact_call_pairs():
    history = [HumanMessage(content="使用三处证据比较")]
    add_receipt(history, "p36", "消息定义", page=36)
    add_receipt(history, "p41", "工具调用示例", page=41)
    add_receipt(history, "p44", "回填代码", page=44)
    add_receipt(history, "p41", "工具调用示例", page=41)
    view = project_task_history(history)
    assert [row.get("text") for row in rows(view)] == ["消息定义", "工具调用示例", "回填代码", None]
    assert rows(view)[-1]["text_status"] == "shown_in_another_receipt"
    assert [item.tool_call_id for item in view if isinstance(item, ToolMessage)] == [
        item.tool_call_id for item in history if isinstance(item, ToolMessage)
    ]
    assert rows(history)[-1]["text"] == "工具调用示例"


def test_long_view_offloads_older_body_and_recalls_exact_original_even_if_large():
    history = [HumanMessage(content="先读长篇，再回查较早原句")]
    for index in range(12):
        add_receipt(history, f"p{index}", f"{index:02}", page=index)
    old = rows(project_task_history(history, evidence_chars=4))
    assert [row["evidence_id"] for row in old if "text" in row] == ["p10", "p11"]
    assert old[0]["text_status"] == "available_via_recall"
    add_receipt(history, "p0", "00", tool="recall_evidence", page=0)
    recalled = rows(project_task_history(history, evidence_chars=1))
    assert recalled[-1]["text"] == "00"
    assert recalled[0]["text_status"] == "shown_in_another_receipt"


def test_different_text_on_same_page_remains_independent():
    history = []
    add_receipt(history, "p41-fragment", "片段", page=41)
    add_receipt(history, "p41-full", "完整正文", page=41)
    assert [row["text"] for row in rows(project_task_history(history))] == ["片段", "完整正文"]


def test_search_view_only_removes_locations_already_present_in_evidence():
    location = {"source": "book.pdf", "page": 41, "chunk_index": 3}
    evidence = {"evidence_id": "chunk-3", "text": "供核对的原文", **location}
    complete = {"locations": [location], "evidence": [evidence], "omitted_blocks": 2}
    incomplete = {**complete, "locations": [location, {**location, "page": 42}]}
    messages = [ToolMessage(tool_call_id=str(i), content=json.dumps(item))
                for i, item in enumerate((complete, incomplete))]

    result = project_task_history(messages)
    assert json.loads(result[0].content) == {"evidence": [evidence], "omitted_blocks": 2}
    assert json.loads(result[1].content)["locations"] == incomplete["locations"]
    assert json.loads(messages[0].content) == complete
    assert [message.tool_call_id for message in result] == ["0", "1"]


def test_worker_recovers_offloaded_text_through_the_real_recall_tool(tmp_path, monkeypatch):
    book = BOOKS[0]
    quote = "原始正文超过试验窗口"
    monkeypatch.setattr(probe, "read_pdf_pages", lambda *args: {
        "pages": [{"page": 44, "text": quote}], "truncated": False,
    })

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [{
                "id": "A", "instruction": "回读第44页原话", "sources": [book], "depends_on": [],
            }]}))
        if stage == "deliver":
            return AIMessage(content=quote)
        received = rows(messages)
        if not received:
            return AIMessage(content="", tool_calls=[{
                "id": "read", "name": "read_pdf",
                "args": {"relative_path": book, "start_page": 44, "end_page": 44},
            }])
        key = received[0]["evidence_id"]
        if len(received) == 1:
            assert received[0]["text_status"] == "available_via_recall"
            return AIMessage(content="", tool_calls=[{
                "id": "recall", "name": "recall_evidence", "args": {"evidence_id": key},
            }])
        assert received[-1]["text"] == quote
        return AIMessage(content=json.dumps({
            "findings": [{"statement": quote, "evidence_ids": [key]}], "gaps": [],
        }))

    result = asyncio.run(probe.build_graph(
        tmp_path, {book: {44}}, ask, [], roomy=True, task_evidence_chars=4,
    ).ainvoke({"question": "逐字引用第44页原话"}))
    assert result["results"]["A"]["status"] == "done"
    assert result["answer"] == quote
