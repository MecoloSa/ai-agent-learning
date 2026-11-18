"""真实工具的列表/搜索上限必须可见，不强制模型继续检索。"""
from dataclasses import replace
import json

import ai_agent.resources.tools as tools
from ai_agent.runtime.context_handoff import pack_material_context, unpack_material_context


def factory(tmp_path, monkeypatch, **limits):
    settings = replace(tools.get_settings(), **limits)
    monkeypatch.setattr(tools, "get_settings", lambda: settings)
    facts = []
    result = {t.name: t for t in tools.build_agent_tools(str(tmp_path), acquisition_facts=facts)}
    return result, facts


def test_hidden_file_list_reports_total_and_limit(tmp_path, monkeypatch):
    for name in ("a.md", "b.md", "z.md"):
        (tmp_path / name).write_text("测试资料", encoding="utf-8")
    bound, facts = factory(tmp_path, monkeypatch, tool_file_list_limit=2)
    result = bound["list_learning_files"].invoke({})
    assert "z.md" not in result
    assert "仅返回2/3个文件" in result
    assert facts == [{"tool": "list_learning_files", "status": "truncated", "returned_files": 2, "total_files": 3}]
    assert unpack_material_context(pack_material_context("", facts)) == ("", facts)


def test_complete_list_does_not_add_notice_or_fact(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("甲", encoding="utf-8")
    bound, facts = factory(tmp_path, monkeypatch, tool_file_list_limit=2)
    assert bound["list_learning_files"].invoke({}) == "a.md (3 bytes)"
    assert facts == []


def test_search_limit_is_not_exhaustive_claim(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("标记甲\n标记乙\n标记丙", encoding="utf-8")
    bound, facts = factory(tmp_path, monkeypatch, tool_search_result_limit=2)
    result = bound["search_learning_files"].invoke({"keyword": "标记"})
    assert "标记丙" not in result
    assert "未确认是否还有其他命中" in result
    assert facts == [{"tool": "search_learning_files", "status": "limit_reached", "returned_hits": 2, "limit": 2}]


def test_no_hits_adds_no_limit_fact(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("甲", encoding="utf-8")
    bound, facts = factory(tmp_path, monkeypatch, tool_search_result_limit=2)
    assert bound["search_learning_files"].invoke({"keyword": "乙"}) == "未找到关键词：乙"
    assert facts == []
