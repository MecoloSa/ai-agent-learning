"""按行读取工具与交互分发测试。

使用真实临时文件和假模型响应，验证工具执行及ToolMessage回传。
不访问在线服务、不执行Python资料、不写入真实学习状态。
"""

import json
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import ai_agent.main as main_module
from ai_agent.resources.tools import build_read_material_lines_tool


def test_reader_tool_returns_actual_lines(tmp_path):
    (tmp_path / "notes.md").write_text(
        "第一行\n第二行\n第三行\n", encoding="utf-8",
    )
    tool = build_read_material_lines_tool(str(tmp_path))

    result = json.loads(tool.invoke({
        "relative_path": "notes.md",
        "start_line": 2,
        "end_line": 3,
    }))

    assert result["ok"] is True
    assert result["data"]["citation"] == "[source=notes.md, lines=2-3]"
    assert result["data"]["lines"][0]["text"] == "第二行"


@pytest.mark.parametrize("invalid_start", [True, "1", 1.5])
def test_tool_schema_rejects_non_integer_lines(tmp_path, invalid_start):
    tool = build_read_material_lines_tool(str(tmp_path))

    with pytest.raises(ValueError):
        tool.invoke({
            "relative_path": "notes.md",
            "start_line": invalid_start,
            "end_line": 2,
        })


@pytest.mark.parametrize(
    "relative_path, expected_ok",
    [("notes.md", True), ("../outside.md", False)],
)
def test_interactive_reader_result_reaches_model(
    monkeypatch, tmp_path, relative_path, expected_ok,
):
    root = tmp_path / "materials"
    root.mkdir()
    (root / "notes.md").write_text(
        "第一行\n第二行\n第三行\n", encoding="utf-8",
    )
    (tmp_path / "outside.md").write_text(
        "目录外内容不得返回", encoding="utf-8",
    )

    settings = replace(
        main_module.get_settings(),
        interactive_max_retrieval_calls=2,
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    calls = []

    def fake_invoke(llm, messages, *, stage):
        calls.append(list(messages))

        if len(calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_material_lines",
                    "args": {
                        "relative_path": relative_path,
                        "start_line": 2,
                        "end_line": 3,
                    },
                    "id": "read-001",
                    "type": "tool_call",
                }],
            )

        results = [
            message for message in messages
            if isinstance(message, ToolMessage)
        ]
        assert len(results) == 1
        assert results[0].tool_call_id == "read-001"

        data = json.loads(results[0].content)
        assert data["ok"] is expected_ok
        assert "目录外内容不得返回" not in results[0].content

        if expected_ok:
            assert data["data"]["lines"][0]["text"] == "第二行"
        else:
            assert "error" in data

        return AIMessage(content="依据实际工具结果回答。")

    monkeypatch.setattr(main_module, "invoke_llm_response", fake_invoke)

    result = main_module._invoke_interactive_turn(
        request_messages=[],
        user_input="请读取指定文件第2到3行。",
        plain_llm=object(),
        tool_llm=object(),
        retrieval_tool=None,
        read_tool=build_read_material_lines_tool(str(root)),
    )

    assert result == "依据实际工具结果回答。"
    assert len(calls) == 2

@pytest.mark.parametrize(
    "scenario, expected_calls",
    [
        ("never_calls", 2),
        ("corrected", 3),
        ("read_failed", 2),
    ],
)
def test_explicit_read_requires_successful_execution(
    monkeypatch, tmp_path, scenario, expected_calls,
):
    """验证漏调用、纠正成功、读取失败三个分支，不访问在线模型。"""
    (tmp_path / "notes.md").write_text(
        "第一行\n第二行\n第三行\n",
        encoding="utf-8",
    )

    settings = replace(
        main_module.get_settings(),
        interactive_max_retrieval_calls=2,
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    call_count = 0

    def fake_invoke(llm, messages, *, stage):
        nonlocal call_count
        call_count += 1

        if scenario == "never_calls":
            # 故意模拟本次线上问题：没有工具调用，却声称已经读取。
            return AIMessage(content="工具已成功读取，文件共有10行。")

        should_call = (
            scenario == "corrected" and call_count == 2
        ) or (
            scenario == "read_failed" and call_count == 1
        )

        if should_call:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_material_lines",
                    "args": {
                        "relative_path": (
                            "missing.md"
                            if scenario == "read_failed"
                            else "notes.md"
                        ),
                        "start_line": 3,
                        "end_line": 20,
                    },
                    "id": "read-check-001",
                    "type": "tool_call",
                }],
            )

        if scenario == "corrected" and call_count == 1:
            return AIMessage(content="没有读取就直接回答。")

        results = [
            message for message in messages
            if isinstance(message, ToolMessage)
        ]
        assert len(results) == 1
        payload = json.loads(results[0].content)

        if scenario == "corrected":
            assert payload["ok"] is True

            # 请求到20行，但真实文件只有3行：
            # 必须以实际返回范围为准，不能补出不存在的行。
            assert payload["data"]["returned_start_line"] == 3
            assert payload["data"]["returned_end_line"] == 3
            assert payload["data"]["lines"][0]["text"] == "第三行"
            return AIMessage(content="实际只返回第3行：第三行。")

        assert payload["ok"] is False
        # 即使模型在工具失败后仍声称成功，也不能向用户放行。
        return AIMessage(content="工具已成功读取，文件共有10行。")

    monkeypatch.setattr(
        main_module, "invoke_llm_response", fake_invoke,
    )

    result = main_module._invoke_interactive_turn(
        request_messages=[],
        user_input=(
            "请实际调用 read_material_lines，"
            "读取 notes.md 第3到20行。"
        ),
        plain_llm=object(),
        tool_llm=object(),
        retrieval_tool=None,
        read_tool=build_read_material_lines_tool(str(tmp_path)),
    )

    assert call_count == expected_calls

    if scenario == "corrected":
        assert result == "实际只返回第3行：第三行。"
    else:
        assert "不能确认" in result
        assert "文件共有10行" not in result
