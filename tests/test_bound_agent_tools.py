"""分析 Agent 工具目录绑定测试。

验证模型参数不能替换根目录，正常访问仍然可用；
全部使用临时文件，不调用 LLM 或 Embedding。
"""

import pytest
import json
from pydantic import ValidationError

import ai_agent.resources.tools as tools_module


@pytest.fixture
def bound_tools(tmp_path, monkeypatch):
    root = tmp_path / "materials"
    root.mkdir()
    (root / "notes.md").write_text("阅读与理解", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("外部内容", encoding="utf-8")

    monkeypatch.setattr(tools_module, "needs_rag", lambda _: False)
    tools = {
        item.name: item
        for item in tools_module.build_agent_tools(str(root))
    }
    return root, tools


def test_normal_access_uses_bound_directory(bound_tools):
    _, tools = bound_tools

    assert "notes.md" in tools["list_learning_files"].invoke({})
    result = json.loads(tools["read_learning_file"].invoke({
        "relative_path": "notes.md",
    }))
    assert result["ok"] is True
    assert result["data"]["text"] == "阅读与理解"
    assert result["data"]["citation"] == "[source=notes.md]"
    assert result["data"]["truncated"] is False
    assert "chunk" not in result["data"]
    assert "notes.md:1:" in tools["search_learning_files"].invoke({
        "keyword": "理解",
    })


@pytest.mark.parametrize("name,args", [
    ("list_learning_files", {}),
    ("read_learning_file", {"relative_path": "notes.md"}),
    ("search_learning_files", {"keyword": "理解"}),
])
def test_model_cannot_supply_another_root(bound_tools, name, args):
    root, tools = bound_tools
    tool = tools[name]

    # 统一支持 Pydantic 类与 JSON Schema 字典两种声明方式。
    schema = tool.get_input_jsonschema()
    assert "root_dir" not in schema.get("properties", {})

    with pytest.raises(ValidationError):
        tool.invoke({**args, "root_dir": str(root.parent)})


def test_read_rejects_parent_and_absolute_paths(bound_tools):
    root, tools = bound_tools
    read_tool = tools["read_learning_file"]

    for path in ("../outside.txt", str(root.parent / "outside.txt")):
        with pytest.raises(ValueError):
            read_tool.invoke({"relative_path": path})


@pytest.mark.parametrize("max_chars", [0, -1, True, 1.5, "10"])
def test_read_rejects_invalid_length(bound_tools, max_chars):
    _, tools = bound_tools

    with pytest.raises(ValidationError):
        tools["read_learning_file"].invoke({
            "relative_path": "notes.md",
            "max_chars": max_chars,
        })


def test_external_file_link_is_not_listed_or_searched(bound_tools):
    root, tools = bound_tools
    link = root / "linked.txt"
    try:
        link.symlink_to(root.parent / "outside.txt")
    except (OSError, NotImplementedError):
        pytest.skip("当前环境不允许创建文件符号链接")

    assert "linked.txt" not in tools["list_learning_files"].invoke({})
    assert "外部内容" not in tools["search_learning_files"].invoke({
        "keyword": "外部内容",
    })
    with pytest.raises(ValueError):
        tools["read_learning_file"].invoke({
            "relative_path": "linked.txt",
        })

def test_list_rejects_unknown_fields_before_scanning(
    bound_tools, monkeypatch
):
    """非法参数应在文件扫描前失败，而不是执行后才发现。"""
    _, tools = bound_tools

    def forbidden_scan(_):
        raise AssertionError("非法参数不应触发文件扫描")

    monkeypatch.setattr(
        tools_module, "iter_material_files", forbidden_scan
    )

    with pytest.raises(ValidationError):
        tools["list_learning_files"].invoke({"unexpected": 1})


def test_list_supports_tool_call_and_rejects_extra_arguments(bound_tools):
    """验证真实模型使用的 ToolCall 外层结构，不只测试普通参数字典。"""
    root, tools = bound_tools
    listing = tools["list_learning_files"]

    result = listing.invoke({
        "name": listing.name,
        "id": "list-ok",
        "type": "tool_call",
        "args": {},
    })
    assert result.tool_call_id == "list-ok"
    assert "notes.md" in result.content

    with pytest.raises(ValidationError):
        listing.invoke({
            "name": listing.name,
            "id": "list-invalid",
            "type": "tool_call",
            "args": {"root_dir": str(root.parent)},
        })

def test_direct_read_reports_truncation_without_inventing_chunks(bound_tools):
    root, tools = bound_tools
    (root / "long.txt").write_text("abcdefghij", encoding="utf-8")

    result = json.loads(tools["read_learning_file"].invoke({
        "relative_path": "long.txt",
        "max_chars": 3,
    }))
    data = result["data"]

    assert data["text"] == "abc"
    assert data["returned_chars"] == 3
    assert data["truncated"] is True
    assert data["scope"] == "prefix"
    assert data["citation"] == "[source=long.txt]"
    assert "chunk" not in data
