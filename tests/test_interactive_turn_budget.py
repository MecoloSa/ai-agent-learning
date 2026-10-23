"""复用同一工具跨轮耗尽预算，验证调用计数和重复保护不会串轮。"""
from types import SimpleNamespace

from langchain_core.messages import AIMessage

import ai_agent.main as main


def test_tool_budget_and_duplicate_guard_reset_for_each_user_turn(monkeypatch):
    executed = []

    class Tool:
        name = "probe_material"

        def invoke(self, args):
            executed.append(dict(args))
            return "本轮资料"

    responses = iter([
        AIMessage(content="", tool_calls=[{
            "name": "probe_material", "args": {"topic": "循环"}, "id": "call-a",
        }]),
        AIMessage(content="第一轮回答"),
        AIMessage(content="", tool_calls=[{
            "name": "probe_material", "args": {"topic": "循环"}, "id": "call-b",
        }]),
        AIMessage(content="第二轮回答"),
    ])
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(interactive_max_retrieval_calls=1))
    monkeypatch.setattr(main, "invoke_llm_response", lambda *a, **kw: next(responses))
    tool = Tool()
    traces = []
    for question in ("请解释循环", "请再核对一次循环"):
        trace = {}
        main._invoke_interactive_turn(
            request_messages=[], user_input=question,
            plain_llm=object(), tool_llm=object(), retrieval_tool=tool,
            audit_trace=trace,
        )
        traces.append(trace)

    # 两轮都用尽一次额度；同工具同参数在新轮次仍应真实执行。
    assert executed == [{"topic": "循环"}, {"topic": "循环"}]
    for trace in traces:
        assert trace["finished"] is True
        assert len(trace["tool_results"]) == 1
        assert trace["tool_results"][0]["executed"] is True
