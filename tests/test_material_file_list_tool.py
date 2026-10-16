"""交互资料清单测试。

验证分页、类型过滤、参数边界及真实工具结果回传。
只使用临时目录与假模型，不访问在线服务。
"""

import json
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import ai_agent.main as main_module
from ai_agent.execution_guard import tool_result_succeeded
from ai_agent.tools import build_list_material_files_tool


def test_file_list_filters_and_paginates(tmp_path):
    for name in ["a.md", "b.pdf", "c.py", "d.txt", "secret.env"]:
        (tmp_path / name).write_text("正文不应返回", encoding="utf-8")

    tool = build_list_material_files_tool(str(tmp_path))

    first = json.loads(tool.invoke({"offset": 0, "limit": 2}))
    data = first["data"]

    assert first["ok"] is True
    assert data["total_files"] == 4
    assert data["returned_count"] == 2
    assert data["has_more"] is True
    assert data["next_offset"] == 2
    assert [
        item["relative_path"] for item in data["files"]
    ] == ["a.md", "b.pdf"]
    assert "正文不应返回" not in json.dumps(first, ensure_ascii=False)

    second = json.loads(tool.invoke({
        "offset": data["next_offset"],
        "limit": 2,
    }))
    assert second["data"]["has_more"] is False
    assert second["data"]["next_offset"] is None
    assert [
        item["relative_path"] for item in second["data"]["files"]
    ] == ["c.py", "d.txt"]


def test_empty_directory_is_successful_listing(tmp_path):
    tool = build_list_material_files_tool(str(tmp_path))
    result = tool.invoke({})

    assert tool_result_succeeded("list_material_files", result)
    data = json.loads(result)["data"]
    assert data["files"] == []
    assert data["total_files"] == 0
    assert data["has_more"] is False


@pytest.mark.parametrize(
    "args",
    [
        {"offset": -1},
        {"offset": True},
        {"offset": "0"},
        {"limit": 0},
        {"limit": 21},
        {"limit": 1.5},
    ],
)
def test_file_list_rejects_invalid_parameters(tmp_path, args):
    tool = build_list_material_files_tool(str(tmp_path))

    with pytest.raises(ValueError):
        tool.invoke(args)


def test_file_list_rejects_missing_root(tmp_path):
    tool = build_list_material_files_tool(str(tmp_path / "missing"))

    with pytest.raises(ValueError):
        tool.invoke({})


def test_file_list_result_reaches_interactive_model(monkeypatch, tmp_path):
    (tmp_path / "notes.md").write_text("测试正文", encoding="utf-8")
    list_tool = build_list_material_files_tool(str(tmp_path))

    settings = replace(
        main_module.get_settings(),
        interactive_max_retrieval_calls=2,
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    calls = 0

    def fake_invoke(llm, messages, *, stage):
        nonlocal calls
        calls += 1

        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "list_material_files",
                    "args": {"offset": 0, "limit": 20},
                    "id": "list-001",
                    "type": "tool_call",
                }],
            )

        results = [
            message for message in messages
            if isinstance(message, ToolMessage)
        ]
        assert len(results) == 1
        assert results[0].tool_call_id == "list-001"

        data = json.loads(results[0].content)["data"]
        assert data["files"][0]["relative_path"] == "notes.md"
        return AIMessage(content="当前资料包含notes.md。")

    monkeypatch.setattr(main_module, "invoke_llm_response", fake_invoke)

    result = main_module._invoke_interactive_turn(
        request_messages=[],
        user_input="请实际调用 list_material_files，列出资料。",
        plain_llm=object(),
        tool_llm=object(),
        retrieval_tool=None,
        extra_tools=(list_tool,),
    )

    assert calls == 2
    assert result == "当前资料包含notes.md。"