"""失败回归：定位成功后空响应，不应让尚有预算的回读永久失去机会。"""
from dataclasses import replace
import json
import pytest

from langchain_core.messages import AIMessage, ToolMessage
import ai_agent.main as app
from ai_agent.persistence.history_readback import HistoryReadAccess
from ai_agent.persistence.history_session import ArchiveCapture


def test_empty_response_after_locator_can_continue_to_read_with_remaining_budget(tmp_path, monkeypatch):
    quote = "今天在地铁上，能听讲解但不方便看图。"
    capture = ArchiveCapture(tmp_path, None, True)
    capture.capture(quote, "了解。")
    access = HistoryReadAccess(tmp_path)
    access.start_turn(capture.state)
    plain, tool = object(), object()
    settings = replace(app.get_settings(), interactive_max_retrieval_calls=3)
    monkeypatch.setattr(app, "get_settings", lambda: settings)
    empty_sent = False
    def response(llm, messages, *, stage):
        nonlocal empty_sent
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            name, args = "search_conversation_turns", {"query": "地铁"}
        elif len(results) == 1 and not empty_sent:
            empty_sent = True
            return AIMessage(content="")
        elif len(results) == 1:
            if llm is plain:
                return AIMessage(content="当前恢复阶段不能继续读取，无法提供原话。")
            name, args = "read_conversation_turn", {"source_index": 1, "turn_number": 1}
        else:
            data = json.loads(results[-1].content)["data"]
            return AIMessage(content=data["user_text"] + data["citation"])
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(len(results)), "type": "tool_call"}])
    monkeypatch.setattr(app, "invoke_llm_response", response)
    trace = {}
    answer = app._invoke_interactive_turn(
        request_messages=[], user_input="请找回地铁学习的完整原话并引用，不修改记录。",
        plain_llm=plain, tool_llm=tool, retrieval_tool=None, extra_tools=access.tools, audit_trace=trace,
    )
    assert [r["tool_name"] for r in trace["tool_results"]] == ["search_conversation_turns", "read_conversation_turn"]
    assert quote in answer


@pytest.mark.parametrize("limit", [1, 3])
def test_continuation_is_bounded_and_does_not_increase_tool_budget(tmp_path, monkeypatch, limit):
    capture = ArchiveCapture(tmp_path, None, True)
    capture.capture("地铁上不方便看图。", "了解。")
    access = HistoryReadAccess(tmp_path)
    access.start_turn(capture.state)
    plain, tool = object(), object()
    monkeypatch.setattr(app, "get_settings", lambda: replace(settings, interactive_max_retrieval_calls=limit))
    settings = __import__('ai_agent.config', fromlist=['get_settings']).get_settings()
    calls = []
    def response(llm, messages, *, stage):
        calls.append((llm, stage))
        if len(calls) == 1:
            return AIMessage(content="", tool_calls=[{"name": "search_conversation_turns", "args": {"query": "地铁"}, "id": "search", "type": "tool_call"}])
        if stage == "交互回答-空响应恢复":
            return AIMessage(content="本轮未取得原话，不能引用。")
        return AIMessage(content="")
    monkeypatch.setattr(app, "invoke_llm_response", response)
    trace = {}
    answer = app._invoke_interactive_turn(request_messages=[], user_input="查找地铁原话并引用。",
        plain_llm=plain, tool_llm=tool, retrieval_tool=None, extra_tools=access.tools, audit_trace=trace)
    assert len(trace["tool_results"]) == 1
    assert len(calls) == (4 if limit == 3 else 3)
    assert answer == "本轮未取得原话，不能引用。"
