"""生产循环实验的脚本起点、工具接线、视图隔离与失败记录。"""
import json

from langchain_core.messages import AIMessage
import pytest

from tests.helpers import receipt_agent_compare as experiment


@pytest.fixture
def cases():
    return json.loads(experiment.CASES.read_text(encoding="utf-8"))


@pytest.mark.parametrize("arm", ["raw", "compact"])
@pytest.mark.parametrize("index", [0, 1])
def test_runs_production_loop_with_real_tools(cases, arm, index, tmp_path, monkeypatch):
    responses = []
    if index == 0:
        responses.append(AIMessage(content="", tool_calls=[{"name": "read_material_lines",
            "args": {"relative_path": "markers.md", "start_line": 1, "end_line": 200},
            "id": "continue", "type": "tool_call"}]))
    responses.append(AIMessage(content="已依据原文回答。[source=markers.md]"))
    iterator = iter(responses)

    class Fake:
        def invoke(self, messages):
            return next(iterator)

    monkeypatch.setattr(experiment.agents, "create_llm_with_tools", lambda tools: Fake())
    original_prepare = experiment.agents.prepare_model_messages
    row = experiment.run_chain(cases[index], arm, 1, tmp_path / "materials")
    assert row["status"] == "completed"
    assert row["source_unchanged"] and row["materials_unchanged"]
    assert len(row["tool_executions"]) == 2
    assert len(row["attempts"]) == (2 if index == 0 else 1)
    assert row["requests"][0]["source"] == "scripted_seed_no_model_request"
    assert experiment.agents.prepare_model_messages is original_prepare
    for request in row["requests"]:
        raw, view = request["working_messages"], request["request_messages"]
        if arm == "raw":
            assert raw == view
        else:
            assert len(raw) == len(view)
    tool_result = row["requests"][1]["request_messages"][-1]["content"]
    if arm == "raw" or index == 0:
        # 截断首批回执现在两组都保留原始JSON。
        assert tool_result.startswith("{")
        if index == 0:
            assert json.loads(tool_result)["data"]["truncated"] is True
    else:
        assert tool_result.startswith("[source=")
    assert len(row["tool_schemas"]) >= 5


def test_empty_response_is_chain_failure_and_restores_hooks(cases, tmp_path, monkeypatch):
    class Fake:
        def invoke(self, messages):
            return AIMessage(content="")
    monkeypatch.setattr(experiment.agents, "create_llm_with_tools", lambda tools: Fake())
    before = experiment.agents._invoke_llm_raw
    row = experiment.run_chain(cases[0], "compact", 1, tmp_path / "materials")
    assert row["status"] == "failed"
    assert len(row["model_responses"]) == 0
    assert len(row["attempts"]) == 2
    assert experiment.agents._invoke_llm_raw is before


def test_no_seed_uses_model_for_first_decision(cases, tmp_path, monkeypatch):
    case = {**cases[0], "seed_calls": []}
    class Fake:
        def invoke(self, messages):
            return AIMessage(content="没有取得资料，暂无法确认。")
    monkeypatch.setattr(experiment.agents, "create_llm_with_tools", lambda tools: Fake())
    row = experiment.run_chain(case, "compact", 1, tmp_path / "materials")
    assert row["status"] == "completed"
    assert len(row["attempts"]) == 1
    assert row["requests"][0]["source"] == "model_request"
    assert row["tool_executions"] == []
