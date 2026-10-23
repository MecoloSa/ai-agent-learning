"""真实MCP会话到同步工具接口的回归；不请求LLM或修改学习存档。"""
import importlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import ai_agent.main as main
from ai_agent.resources.tools import build_read_material_lines_tool


@pytest.fixture
def open_reader():
    try:
        module = importlib.import_module("ai_agent.integrations.mcp_reader")
    except ModuleNotFoundError as exc:
        if exc.name != "ai_agent.integrations.mcp_reader":
            raise
        pytest.skip("MCP同步适配方案尚未应用")
    return module.open_mcp_reader


def test_same_session_results_errors_and_close(open_reader, tmp_path):
    (tmp_path / "notes.md").write_text("第一行\n第二行\n", encoding="utf-8")
    args = {"relative_path": "notes.md", "start_line": 1, "end_line": 2}
    expected = build_read_material_lines_tool(str(tmp_path)).invoke(args)
    with open_reader(str(tmp_path)) as tool:
        assert json.loads(tool.invoke(args)) == json.loads(expected)
        with pytest.raises(RuntimeError, match="MCP资料工具返回错误"):
            tool.invoke({**args, "relative_path": "../outside.md"})
        assert json.loads(tool.invoke(args)) == json.loads(expected)
        print("[适配] 正常读取、失败回传、失败后再读均通过")
    with pytest.raises(RuntimeError, match="会话已关闭"):
        tool.invoke(args)
    print("[适配] with退出后调用被拒绝")


def test_existing_turn_loop_accepts_mcp_tool(open_reader, tmp_path, monkeypatch):
    (tmp_path / "notes.md").write_text("正文\n", encoding="utf-8")
    args = {"relative_path": "notes.md", "start_line": 1, "end_line": 1}
    responses = iter([
        message for n in range(2) for message in [
            AIMessage(content="", tool_calls=[{"name": "read_material_lines", "args": args, "id": f"read-{n}"}]),
            AIMessage(content="读取完成"),
        ]
    ])
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(interactive_max_retrieval_calls=1))
    monkeypatch.setattr(main, "invoke_llm_response", lambda *a, **kw: next(responses))
    with open_reader(str(tmp_path)) as tool:
        for _ in range(2):
            trace = {}
            main._invoke_interactive_turn(
                request_messages=[], user_input="请读取资料",
                plain_llm=object(), tool_llm=object(), retrieval_tool=None,
                read_tool=tool, audit_trace=trace,
            )
            assert len(trace["tool_results"]) == 1
            assert trace["tool_results"][0]["executed"] is True
            assert json.loads(trace["tool_results"][0]["result"])["ok"] is True
    print("[适配] 既有轮次循环两轮均实际读取；每轮预算=1")
