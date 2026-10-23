"""交互传输选择与资源释放；模型循环隔离，MCP协议另有真实子进程测试。"""
from contextlib import contextmanager
import inspect
import sys

import pytest

import ai_agent.main as main
import ai_agent.integrations.mcp_reader as mcp_reader
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.state_store import save_learning_state


pytestmark = pytest.mark.skipif(
    "read_transport" not in inspect.signature(main.run_interactive_session).parameters,
    reason="主交互传输接线尚未应用",
)


@pytest.mark.parametrize("transport", ["local", "mcp"])
@pytest.mark.parametrize("fail", [False, True])
def test_transport_lifetime_and_return(monkeypatch, transport, fail):
    events = []
    adapter = object()

    @contextmanager
    def opened(directory):
        assert directory == "materials"
        events.append("open")
        try:
            yield adapter
        finally:
            events.append("close")

    def core(*args, **kwargs):
        assert kwargs["read_tool_override"] is (adapter if transport == "mcp" else None)
        if fail:
            raise ValueError("测试业务异常")
        return "计划", "目标"

    monkeypatch.setattr(mcp_reader, "open_mcp_reader", opened)
    monkeypatch.setattr(main, "_run_interactive_session", core)
    args = ("", "计划", "", "目标", "window", "materials")
    if fail:
        with pytest.raises(ValueError, match="测试业务异常"):
            main.run_interactive_session(*args, read_transport=transport)
    else:
        assert main.run_interactive_session(*args, read_transport=transport) == ("计划", "目标")
    assert events == (["open", "close"] if transport == "mcp" else [])


def test_unknown_transport_fails_before_core(monkeypatch):
    monkeypatch.setattr(main, "_run_interactive_session", lambda *a, **kw: pytest.fail("不能进入业务"))
    with pytest.raises(ValueError):
        main.run_interactive_session("", "计划", "", "目标", None, "materials", read_transport="other")


def test_resume_cli_forwards_transport_without_initial_generation(monkeypatch, tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "notes.md").write_text("演示资料", encoding="utf-8")
    state = LearningState(learning_goal="理解演示", plan_markdown="阅读资料")
    snapshot = save_learning_state(state, tmp_path / "source.json")
    before = snapshot.read_bytes()
    received = []
    monkeypatch.setattr(sys, "argv", [
        "agent", "--resume-state", str(snapshot), "--interactive",
        "--materials", str(materials), "--read-transport", "mcp",
    ])
    monkeypatch.setattr(main, "new_current_state_path", lambda: tmp_path / "current.json")
    monkeypatch.setattr(main, "_run_initial_workflow", lambda *a: pytest.fail("不能重新生成"))
    monkeypatch.setattr(main, "run_interactive_session", lambda **kw: received.append(kw))
    main.main()
    assert received[0]["read_transport"] == "mcp"
    assert received[0]["plan"] == state.plan_markdown
    assert snapshot.read_bytes() == before
