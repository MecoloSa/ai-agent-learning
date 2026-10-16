"""交互空回答恢复测试。

模拟工具执行后模型返回空内容，验证：
保留工具结果、最多补生成一次、不重复执行工具，以及失败时明确提示。
所有模型响应均为模拟，不访问在线服务。
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import ai_agent.main as main_module
from ai_agent.tools import build_list_material_files_tool


@pytest.mark.parametrize("call_limit", [1, 2])
@pytest.mark.parametrize("explicit_request", [True, False])
@pytest.mark.parametrize("recovery", ["success", "empty", "error"])
def test_empty_answer_recovers_without_replaying_tool(
    monkeypatch, tmp_path, call_limit, explicit_request, recovery,
):
    # call_limit=1覆盖预算耗尽后的强制收束出口；
    # call_limit=2覆盖模型自主结束的普通出口。
    settings = replace(
        main_module.get_settings(),
        interactive_max_retrieval_calls=call_limit,
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    (tmp_path / "notes.md").write_text("测试正文", encoding="utf-8")
    real_tool = build_list_material_files_tool(str(tmp_path))

    tool_count = 0

    def counted_invoke(args):
        nonlocal tool_count
        tool_count += 1
        return real_tool.invoke(args)

    # 分发函数只需要name和invoke；用轻量对象记录真实执行次数。
    counted_tool = SimpleNamespace(
        name=real_tool.name,
        invoke=counted_invoke,
    )

    plain_llm = object()
    tool_llm = object()
    model_count = 0

    def fake_invoke(llm, messages, *, stage):
        nonlocal model_count
        model_count += 1

        if model_count == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "list_material_files",
                    "args": {"offset": 0, "limit": 20},
                    "id": "list-empty-001",
                    "type": "tool_call",
                }],
            )

        if model_count == 2:
            return AIMessage(content="")

        assert model_count == 3
        assert llm is plain_llm
        assert stage == "交互回答-空响应恢复"

        results = [
            message for message in messages
            if isinstance(message, ToolMessage)
        ]
        assert len(results) == 1
        assert results[0].tool_call_id == "list-empty-001"
        assert "notes.md" in results[0].content

        if recovery == "error":
            raise main_module.LLMInvocationError("模拟恢复失败")

        if recovery == "empty":
            return AIMessage(content=" \n ")

        return AIMessage(content="当前资料包含notes.md，类型为md。")

    monkeypatch.setattr(main_module, "invoke_llm_response", fake_invoke)

    user_input = (
        "请实际调用 list_material_files，列出资料。"
        if explicit_request
        else "请查看当前资料文件清单。"
    )

    result = main_module._invoke_interactive_turn(
        request_messages=[],
        user_input=user_input,
        plain_llm=plain_llm,
        tool_llm=tool_llm,
        retrieval_tool=None,
        extra_tools=(counted_tool,),
    )

    assert tool_count == 1
    assert model_count == 3
    assert result.strip()

    if recovery == "success":
        assert result == "当前资料包含notes.md，类型为md。"
    else:
        assert "自动恢复仍未成功" in result


def test_plain_chat_empty_answer_also_recovers(monkeypatch):
    """没有工具的普通问答，也不能直接输出空回答。"""
    calls = 0

    def fake_invoke(llm, messages, *, stage):
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(content="")
        assert stage == "交互回答-空响应恢复"
        return AIMessage(content="这是补生成的解释。")

    monkeypatch.setattr(main_module, "invoke_llm_response", fake_invoke)

    result = main_module._invoke_interactive_turn(
        request_messages=[],
        user_input="解释活动进度与理解情况的区别。",
        plain_llm=object(),
        tool_llm=None,
        retrieval_tool=None,
    )

    assert calls == 2
    assert result == "这是补生成的解释。"