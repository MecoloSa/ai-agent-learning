"""真实执行分支的离线接线测试；不请求在线模型，不评价生成效果。"""
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from ai_agent.evaluation import feedback_plan_runner as runner
from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases


@pytest.fixture
def cases():
    return load_feedback_plan_cases(Path(__file__).parent / "fixtures/evaluation/feedback_plan_cases.jsonl")


@pytest.fixture
def dependencies(monkeypatch):
    calls = []
    def plain(**kwargs):
        calls.append("plain")
        return object()
    def tools(*args, **kwargs):
        calls.append("tools")
        return object()
    def needs_rag(_):
        calls.append("needs_rag")
        return False
    monkeypatch.setattr(runner.app, "create_llm", plain)
    monkeypatch.setattr(runner.app, "create_llm_with_tools", tools)
    monkeypatch.setattr(runner.app, "needs_rag", needs_rag)
    return calls


def test_live_branch_uses_original_turn_function_and_keeps_audit(cases, dependencies, monkeypatch, tmp_path):
    def model(*args, **kwargs):
        return AIMessage(content="只讨论：阅读15分钟，口述15分钟。")
    monkeypatch.setattr(runner.app, "invoke_llm_response", model)
    original = runner.app._invoke_interactive_turn
    result = runner.run_case(cases[0], None, tmp_path, online=True)
    assert result["status"] == "completed"
    assert result["execution_mode"] == "online_model"
    assert result["stub_response_count"] == 0
    assert {"plain", "tools", "needs_rag"} <= set(dependencies)
    turn = result["turns"][0]
    assert turn["status"] == "completed"
    assert turn["audit_trace"]["finished"] is True
    assert turn["response"] == turn["audit_trace"]["final_answer"]
    assert turn["audit_trace"]["model_responses"][0]["content"] == turn["response"]
    assert runner.app._invoke_interactive_turn is original
    assert all(c["value"] is None for c in result["checks"] if c["category"] != "control")


def test_live_candidate_turn_number_and_decline_still_follow_business_flow(cases, dependencies, monkeypatch, tmp_path):
    monkeypatch.setattr(runner.app, "invoke_llm_response", lambda *a, **k: AIMessage(
        content="===NEW_PLAN_START===\n阅读15分钟，口述15分钟。\n===NEW_PLAN_END==="))
    result = runner.run_case(cases[1], None, tmp_path, online=True)
    assert result["status"] == "completed"
    assert result["stub_response_count"] == 0
    assert result["plan_changes"][0]["turn"] == 1
    assert result["plan_changes"][0]["decision"] == "cancelled"
    assert result["business_state_unchanged"] is True


def test_partial_trace_survives_turn_exception(cases, dependencies, monkeypatch, tmp_path):
    def broken(**kwargs):
        kwargs["audit_trace"]["tool_results"] = [{"executed": True, "result": "部分证据"}]
        raise RuntimeError("异常正文不应写入结果摘要")
    monkeypatch.setattr(runner.app, "_invoke_interactive_turn", broken)
    result = runner.run_case(cases[0], None, tmp_path, online=True)
    assert result["status"] == "failed"
    turn = result["turns"][0]
    assert turn["status"] == "failed"
    assert turn["error_type"] == "RuntimeError"
    assert turn["response"] is None
    assert turn["audit_trace"]["tool_results"][0]["result"] == "部分证据"
    assert "异常正文" not in json.dumps(result, ensure_ascii=False)


def test_each_turn_gets_fresh_audit_dictionary(cases, dependencies, monkeypatch, tmp_path):
    traces = []
    def execute(**kwargs):
        trace = kwargs["audit_trace"]
        assert trace == {}
        traces.append(trace)
        trace["finished"] = True
        return "只讨论，不修改。"
    monkeypatch.setattr(runner.app, "_invoke_interactive_turn", execute)
    case = cases[0].model_copy(update={"inputs": ["问题一", "问题二", "exit"]})
    result = runner.run_case(case, None, tmp_path, online=True)
    assert result["status"] == "completed"
    assert [t["turn"] for t in result["turns"]] == [1, 2]
    assert traces[0] is not traces[1]


def test_live_missing_candidate_is_not_filled_in(cases, dependencies, monkeypatch, tmp_path):
    monkeypatch.setattr(runner.app, "invoke_llm_response", lambda *a, **k: AIMessage(content="没有候选。"))
    result = runner.run_case(cases[1], None, tmp_path, online=True)
    assert result["status"] == "script_mismatch"
    assert result["plan_changes"] == []
    assert len(result["turns"]) == 1


@pytest.mark.parametrize("responses,online", [([], True), (["固定回复"], True), (None, False), ([], "true")])
def test_invalid_mode_arguments_fail_before_artifact_creation(cases, tmp_path, responses, online):
    with pytest.raises(ValueError):
        runner.run_case(cases[0], responses, tmp_path, online=online)
    assert list(tmp_path.iterdir()) == []


def test_non_string_return_is_a_failed_turn(cases, dependencies, monkeypatch, tmp_path):
    monkeypatch.setattr(runner.app, "_invoke_interactive_turn", lambda **kwargs: None)
    result = runner.run_case(cases[0], None, tmp_path, online=True)
    assert result["status"] == "failed"
    assert result["turns"][0]["error_type"] == "TypeError"
