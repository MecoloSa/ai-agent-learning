"""显式检索调用的离线回归测试。

确认正常返回可完成调用要求，无命中不冒充找到证据；
漏调用、错误工具和调用异常不能通过确认。
"""

from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import ai_agent.main as main_module
from ai_agent.runtime.execution_guard import resolve_execution_requirement


@pytest.mark.parametrize(
    "name, expected",
    [
        ("retrieve_learning_context", "returned"),
        ("read_pdf_pages", "succeeded"),
        ("read_material_lines", "succeeded"),
    ],
)
def test_confirmation_policy(name, expected):
    requirement = resolve_execution_requirement(
        f"请实际调用 {name}，核对资料。"
    )
    assert requirement.confirmation == expected


@pytest.mark.parametrize(
    "scenario",
    ["returned_empty", "raises", "never_calls", "wrong_tool"],
)
def test_explicit_retrieval_confirmation(monkeypatch, scenario):
    settings = replace(
        main_module.get_settings(),
        interactive_max_retrieval_calls=1,
    )
    monkeypatch.setattr(
        main_module, "get_settings", lambda: settings
    )

    tool_calls = []
    model_calls = []

    class FakeRetrievalTool:
        name = "retrieve_learning_context"

        def invoke(self, args):
            tool_calls.append(args)
            if scenario == "raises":
                raise OSError("模拟检索失败")
            return "没有达到相关性阈值的新增资料。"

    def fake_model(llm, messages, *, stage):
        model_calls.append(stage)

        if scenario == "never_calls":
            return AIMessage(content="我已经查询过了。")

        if len(model_calls) == 1:
            name = (
                "read_pdf_pages"
                if scenario == "wrong_tool"
                else "retrieve_learning_context"
            )
            return AIMessage(
                content="",
                tool_calls=[{
                    "id": "retrieve-1",
                    "name": name,
                    "args": {"queries": ["问题A", "问题B"]},
                    "type": "tool_call",
                }],
            )

        observations = [
            message
            for message in messages
            if isinstance(message, ToolMessage)
        ]
        assert observations
        assert "没有达到相关性阈值" in observations[0].content
        return AIMessage(content="本轮检索未找到可用证据。")

    monkeypatch.setattr(
        main_module, "invoke_llm_response", fake_model
    )

    result = main_module._invoke_interactive_turn(
        request_messages=[],
        user_input=(
            "请实际调用 retrieve_learning_context，"
            "在同一次调用中查询两个问题。"
        ),
        plain_llm=object(),
        tool_llm=object(),
        retrieval_tool=FakeRetrievalTool(),
    )

    if scenario == "returned_empty":
        assert result == "本轮检索未找到可用证据。"
        assert tool_calls == [{"queries": ["问题A", "问题B"]}]
        assert len(model_calls) == 2

    elif scenario == "raises":
        assert len(tool_calls) == 1
        assert len(model_calls) == 1
        assert "不能确认" in result

    else:
        assert tool_calls == []
        assert "未实际执行指定工具" in result
