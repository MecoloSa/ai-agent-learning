from dataclasses import replace
import json
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
import ai_agent.main as app


def test_tool_stage_public_answer_is_delivered_with_final(monkeypatch):
    @tool
    def inspect_note() -> str:
        """读取实验资料。"""
        return "工具实际结果"
    responses = iter([AIMessage(content="当前计划分为阅读和口头说明。", tool_calls=[{
        "id": "r", "name": "inspect_note", "args": {}, "type": "tool_call"}]),
        AIMessage(content="原话已经核实。")])
    monkeypatch.setattr(app, "invoke_llm_response", lambda *a, **k: next(responses))
    result = app._invoke_interactive_turn(request_messages=[], user_input="说明当前计划并核实资料",
        plain_llm=object(), tool_llm=object(), retrieval_tool=None, extra_tools=(inspect_note,))
    assert "当前计划分为阅读和口头说明。" in result
    assert "【最终回答】\n原话已经核实。" in result
    assert "非最终结论" in result


def test_complete_final_does_not_duplicate_fragment(monkeypatch):
    @tool
    def inspect_note() -> str:
        """读取实验资料。"""
        return "结果"
    responses = iter([AIMessage(content="计划分两步。", tool_calls=[{
        "id": "r", "name": "inspect_note", "args": {}, "type": "tool_call"}]),
        AIMessage(content="计划分两步。补充内容。")])
    monkeypatch.setattr(app, "invoke_llm_response", lambda *a, **k: next(responses))
    result = app._invoke_interactive_turn(request_messages=[], user_input="查询资料",
        plain_llm=object(), tool_llm=object(), retrieval_tool=None, extra_tools=(inspect_note,))
    assert result == "计划分两步。补充内容。"


def test_actual_missing_sections_are_restored_without_model_call(monkeypatch):
    case = json.loads((Path(__file__).parent / "fixtures/evaluation/public_fragment_regression.json").read_text(encoding="utf-8"))
    responses = iter([AIMessage(**item) for item in case["responses"]])
    class ReplayTool:
        def __init__(self, name):
            self.name = name
        def invoke(self, args):
            return next(record["result"] for record in case["tool_results"]
                if record["tool_name"] == self.name and record["args"] == args)
    names = dict.fromkeys(record["tool_name"] for record in case["tool_results"])
    monkeypatch.setattr(app, "invoke_llm_response", lambda *a, **k: next(responses))
    answer = app._invoke_interactive_turn(request_messages=[], user_input=case["user_input"],
        plain_llm=object(), tool_llm=object(), retrieval_tool=None,
        extra_tools=tuple(ReplayTool(name) for name in names))
    assert "阶段1：核心区别聚焦" not in case["responses"][-1]["content"]
    assert "阶段1：核心区别聚焦" in answer
    assert "self_reported_unclear" in answer
    assert "今天最多20分钟，只阅读和口头说明。" in answer
    assert "【最终回答】" in answer
