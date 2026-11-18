"""分析阶段长源码访问：只创建并读取资料，不执行源码、不调用模型。"""
import json

import pytest

import ai_agent.resources.tools as tools
from ai_agent.runtime.agents import _build_analysis_budget_status


def long_source(tmp_path):
    prefix = "# 与目标无关的开头说明\n" * (tools.MAX_CHARS_PER_FILE // 8 + 1)
    body = "def target_rule():\n    required_count = 2\n    total_count = 3\n    return required_count, total_count\n"
    (tmp_path / "rules.py").write_text(prefix + body, encoding="utf-8")
    return prefix.count("\n") + 1


def test_analysis_can_search_then_read_complete_function_beyond_prefix(tmp_path, monkeypatch):
    start = long_source(tmp_path)
    monkeypatch.setattr(tools, "needs_rag", lambda _: False)
    blocks, facts = [], []
    bound = {t.name: t for t in tools.build_agent_tools(str(tmp_path), evidence_blocks=blocks, acquisition_facts=facts)}
    head = json.loads(bound["read_learning_file"].invoke({"relative_path": "rules.py"}))["data"]
    assert head["truncated"] is True and "target_rule" not in head["text"]
    located = bound["search_learning_files"].invoke({"keyword": "target_rule"})
    assert f"rules.py:{start}:" in located
    assert "required_count = 2" not in located
    # 只有定位信息不能代替函数正文；分析工具应复用现有的有界行读取。
    assert "read_material_lines" in bound
    blocks.clear()
    facts.clear()
    result = bound["read_material_lines"].invoke({"relative_path": "rules.py", "start_line": start, "end_line": start + 3})
    assert blocks == [result] and facts == []
    data = json.loads(result)["data"]
    assert "required_count = 2" in "\n".join(row["text"] for row in data["lines"])
    assert data["citation"] == f"[source=rules.py, lines={start}-{start + 3}]"
    status = _build_analysis_budget_status(
        ["read_material_lines"], {"read_material_lines": 7}, 7, 12, 2,
    )
    assert status["remaining_by_tool"]["read_material_lines"] == 5
    assert "root_dir" not in bound["read_material_lines"].args
    assert "acquisition_facts" not in bound["read_material_lines"].args


def test_line_reader_publishes_truncation_and_preserves_old_default_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "needs_rag", lambda _: False)
    (tmp_path / "notes.md").write_text("甲" * 9000, encoding="utf-8")
    blocks, facts = [], []
    reader = tools.build_read_material_lines_tool(str(tmp_path), evidence_blocks=blocks, acquisition_facts=facts)
    result = reader.invoke({"relative_path": "notes.md", "start_line": 1, "end_line": 1})
    data = json.loads(result)["data"]
    assert data["truncated"] is True and data["lines"][0]["partial"] is True
    assert blocks == [result]
    assert facts == [{
        "tool": "read_material_lines", "status": "truncated", "source": "notes.md",
        "scope": "lines", "returned_start_line": 1, "returned_end_line": 1,
        "returned_chars": len(data["lines"][0]["text"]),
    }]
    plain = tools.build_read_material_lines_tool(str(tmp_path))
    assert plain.invoke({"relative_path": "notes.md", "start_line": 1, "end_line": 1}) == result


def test_analysis_line_reader_still_rejects_escape_before_collecting(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "needs_rag", lambda _: False)
    blocks, facts = [], []
    bound = {t.name: t for t in tools.build_agent_tools(str(tmp_path), evidence_blocks=blocks, acquisition_facts=facts)}
    assert "read_material_lines" in bound
    with pytest.raises(ValueError, match="以外"):
        bound["read_material_lines"].invoke({"relative_path": "../outside.md", "start_line": 1, "end_line": 1})
    assert blocks == [] and facts == []
