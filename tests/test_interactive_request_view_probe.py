import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.helpers.interactive_request_view import project


def test_view_keeps_raw_evidence_and_tool_pairing():
    messages = [HumanMessage(content="核对资料")]
    for index, (name, data) in enumerate([
        ("inspect_document_structure", {"ok": True, "data": {"mode": "sections", "inspection_complete": True, "sections": []}}),
        ("read_material_lines", {"ok": False, "error": {"total_lines": 21}}),
        ("inspect_document_structure", {"ok": True, "data": {"mode": "whole_document", "text": "必须原样保留的正文"}}),
        ("retrieve_learning_context", {"ok": True, "data": {"text": "检索原文"}}),
    ]):
        key = str(index)
        messages.extend([AIMessage(content="", tool_calls=[{"id": key, "name": name, "args": {}}]),
                         ToolMessage(content=json.dumps(data, ensure_ascii=False), tool_call_id=key)])
    raw = [message.content for message in messages]
    view = project(messages)
    assert [message.content for message in messages] == raw
    assert [message.tool_call_id for message in view if isinstance(message, ToolMessage)] == ["0", "1", "2", "3"]
    assert "inspection_complete" not in view[2].content
    assert "未返回正文" in view[2].content
    assert json.loads(view[4].content)["returned_body_text"] is False
    assert [message.content for message in view[5:]] == raw[5:]
