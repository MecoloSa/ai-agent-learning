"""MCP启动错误与业务异常分开处理，不访问模型或真实用户存档。"""
from contextlib import contextmanager

import pytest

import ai_agent.main as main
import ai_agent.integrations.mcp_reader as reader


pytestmark = pytest.mark.skipif(
    not hasattr(main, "MCPStartupError"),
    reason="MCP启动错误边界方案尚未应用",
)


def test_startup_failure_never_enters_chat_or_falls_back(monkeypatch):
    @contextmanager
    def broken(_):
        raise RuntimeError("PRIVATE-STARTUP-DETAIL")
        yield

    monkeypatch.setattr(reader, "open_mcp_reader", broken)
    monkeypatch.setattr(main, "_run_interactive_session", lambda *a, **kw: pytest.fail("不能进入聊天或本地回退"))
    with pytest.raises(main.MCPStartupError) as captured:
        main.run_interactive_session("", "计划", "", "目标", None, "materials", read_transport="mcp")
    assert "PRIVATE-STARTUP-DETAIL" not in str(captured.value)


def test_cli_startup_error_is_nonzero_and_safe(monkeypatch, capsys):
    def broken_main():
        raise main.MCPStartupError("MCP读取服务未能启动")

    monkeypatch.setattr(main, "_main", broken_main)
    with pytest.raises(SystemExit) as captured:
        main.main()
    assert captured.value.code == 1
    output = capsys.readouterr().out
    assert "未切换为本地读取" in output


def test_cli_does_not_reclassify_other_business_errors(monkeypatch):
    def broken_main():
        raise ValueError("业务参数错误")

    monkeypatch.setattr(main, "_main", broken_main)
    with pytest.raises(ValueError, match="业务参数错误"):
        main.main()
