"""首次pipeline资料预算：不截断、不暗中切换模式，模型请求前检查。"""
from dataclasses import replace

import pytest
import ai_agent.runtime.agents as agents
import ai_agent.config as config


def prepare(monkeypatch, context, limit):
    settings = replace(agents.get_settings(), analysis_initial_context_max_chars=limit)
    monkeypatch.setattr(agents, "get_settings", lambda: settings)
    monkeypatch.setattr(agents, "_collect_python_sources", lambda **kwargs: "")
    monkeypatch.setattr(agents, "needs_rag", lambda _: False)
    monkeypatch.setattr(agents, "collect_material_context", lambda _: context)


@pytest.mark.parametrize("length", [9, 10])
def test_within_budget_keeps_all_context(monkeypatch, length):
    context = "甲" * length
    prepare(monkeypatch, context, 10)
    monkeypatch.setattr(agents, "create_llm", lambda: object())
    messages = []

    def invoke(llm, supplied, *, stage):
        messages.extend(supplied)
        return "分析完成"

    monkeypatch.setattr(agents, "_invoke_llm", invoke)
    _, returned = agents.analyse_materials("unused", "学习目标")
    assert returned == context and context in messages[1].content


@pytest.mark.parametrize("orchestrator", ["legacy", "langgraph"])
def test_over_budget_stops_both_workflows_without_model_or_plan(tmp_path, monkeypatch, orchestrator):
    (tmp_path / "notes.md").write_text("合法资料", encoding="utf-8")
    prepare(monkeypatch, "甲" * 11, 10)

    def forbidden(*args, **kwargs):
        raise AssertionError("超过首次资料预算后不应请求模型、进入规划或切换模式")

    monkeypatch.setattr(agents, "create_llm", forbidden)
    monkeypatch.setattr(agents, "create_study_plan", forbidden)
    monkeypatch.setattr(agents, "analyse_materials_agentic", forbidden)
    with pytest.raises(agents.AnalysisExecutionError) as caught:
        agents.run_learning_assistant(str(tmp_path), "学习目标", 1, "pipeline", orchestrator=orchestrator)
    assert caught.value.reason == "analysis_context_budget_exceeded"


def test_final_assembled_context_is_checked_including_python_headers(monkeypatch):
    prepare(monkeypatch, "概览", 10)
    monkeypatch.setattr(agents, "_collect_python_sources", lambda **kwargs: "[source=a.py]\nvalue=3")
    monkeypatch.setattr(agents, "create_llm", lambda: pytest.fail("不应请求模型"))
    with pytest.raises(agents.AnalysisExecutionError) as caught:
        agents.analyse_materials("unused", "学习目标")
    assert caught.value.reason == "analysis_context_budget_exceeded"


def test_explicit_larger_budget_preserves_full_read_baseline(monkeypatch):
    prepare(monkeypatch, "甲" * 100, 101)
    monkeypatch.setattr(agents, "create_llm", lambda: object())
    monkeypatch.setattr(agents, "_invoke_llm", lambda *args, **kwargs: "分析完成")
    assert agents.analyse_materials("unused", "目标")[1] == "甲" * 100


def test_rag_assembled_context_is_also_checked_before_chat(monkeypatch):
    prepare(monkeypatch, "unused", 80)
    monkeypatch.setattr(agents, "needs_rag", lambda _: True)

    class FakeRAG:
        def __init__(self, *args, **kwargs):
            pass

        def build_overview_text(self, **kwargs):
            return "甲" * 100

        def retrieve_many(self, *args, **kwargs):
            return []

        def expand_neighbors(self, hits, **kwargs):
            return hits

    monkeypatch.setattr(agents, "LearningRAG", FakeRAG)
    monkeypatch.setattr(agents, "create_llm", lambda: pytest.fail("不应请求模型"))
    with pytest.raises(agents.AnalysisExecutionError) as caught:
        agents.analyse_materials("unused", "目标")
    assert caught.value.reason == "analysis_context_budget_exceeded"


def test_guard_does_not_apply_to_agent_tool_loop(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage

    (tmp_path / "notes.md").write_text("甲" * 100, encoding="utf-8")
    prepare(monkeypatch, "unused", 1)
    monkeypatch.setattr(agents, "create_llm_with_tools", lambda _: object())
    monkeypatch.setattr(agents, "_invoke_llm_raw", lambda *args, **kwargs: AIMessage(content="当前证据不足，需后续核对。"))
    analysis, context = agents.analyse_materials_agentic(str(tmp_path), "目标")
    assert analysis and context == ""


def test_config_and_snapshot_include_budget(monkeypatch):
    from ai_agent.evaluation.interactive_metrics import build_eval_config_snapshot

    monkeypatch.setenv("ANALYSIS_INITIAL_CONTEXT_MAX_CHARS", "12345")
    config.get_settings.cache_clear()
    try:
        settings = config.get_settings()
        assert settings.analysis_initial_context_max_chars == 12345
        assert build_eval_config_snapshot(settings)["settings"]["analysis_initial_context_max_chars"] == 12345
    finally:
        config.get_settings.cache_clear()


def test_zero_budget_is_rejected_instead_of_disabling_guard(monkeypatch):
    monkeypatch.setenv("ANALYSIS_INITIAL_CONTEXT_MAX_CHARS", "0")
    config.get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="ANALYSIS_INITIAL_CONTEXT_MAX_CHARS"):
            config.get_settings()
    finally:
        config.get_settings.cache_clear()
