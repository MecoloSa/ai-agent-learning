"""配对实验本身的控制变量和真实规划接线；不请求在线模型。"""
import json
from pathlib import Path
import pytest

from langchain_core.messages import AIMessage

import ai_agent.evaluation.acquisition_pair as experiment


CASES = Path("tests/fixtures/evaluation/acquisition_pair/cases.json")


def prepared():
    root = CASES.parent / "materials"
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    return root, [experiment.prepare_case(case, root) for case in cases]


def test_case_limits_are_generated_by_real_tools():
    _, cases = prepared()
    assert cases[0]["acquisition_facts"] == []
    assert cases[1]["acquisition_facts"][0]["executed_queries"] == 1
    assert cases[1]["acquisition_facts"][0]["requested_queries"] == 2
    assert cases[2]["acquisition_facts"][0]["selected_blocks"] == 1
    assert cases[2]["acquisition_facts"][0]["omitted_blocks"] == 1
    assert "source=stop_protective.md" not in cases[2]["evidence"]


def test_only_facts_change_and_tool_path_is_captured(monkeypatch):
    root, cases = prepared()

    class FakeLLM:
        def __init__(self, use_tool):
            self.use_tool = use_tool

        def invoke(self, messages):
            if self.use_tool:
                return AIMessage(content="", tool_calls=[{
                    "id": "query", "name": "retrieve_learning_context",
                    "args": {"queries": ["人物态度的定义"]}, "type": "tool_call",
                }], usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30})
            return AIMessage(content="任务1：阅读资料。任务2：口头说明。", usage_metadata={"input_tokens": 40, "output_tokens": 10, "total_tokens": 50})

    monkeypatch.setattr(experiment.agents, "create_llm_with_tools", lambda _: FakeLLM(True))
    monkeypatch.setattr(experiment.agents, "create_llm", lambda: FakeLLM(False))
    rows = [experiment.run_one(cases[1], root, arm, 1) for arm in ("without_facts", "with_facts")]
    assert all(row["status"] == "completed" for row in rows)
    assert rows[0]["evidence_sha256"] == rows[1]["evidence_sha256"]
    assert len(rows[0]["retrieval_calls"]) == len(rows[1]["retrieval_calls"]) == 1
    assert len(rows[0]["model_responses"]) == len(rows[1]["model_responses"]) == 2
    before, after = [r["request_messages"][0]["messages"] for r in rows]
    assert before[0] == after[0]
    marker = "\n\n初始资料获取范围与限制"
    assert after[1]["content"].split(marker)[0] == before[1]["content"]
    assert cases[1]["criterion"] not in before[1]["content"]


def test_preflight_does_not_invoke_chat(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("预检不能调用模型")

    monkeypatch.setattr(experiment, "run_one", forbidden)
    assert experiment.main(["--cases", str(CASES), "--output-dir", str(tmp_path)]) == 0
    assert not (tmp_path / "results.jsonl").exists()
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["mode"] == "preflight_only"


def report_rows():
    case = {"case_id": "one", "evidence": "原文", "acquisition_facts": []}
    rows = [{
        "case_id": "one", "repeat": 1, "arm": arm, "status": "completed",
        "source_unchanged": True,
        "request_messages": [{"messages": [{"role": "system", "content": "规则"}, {"role": "human", "content": "任务与证据"}]}],
        "plan_trace": {"initial_evidence": "原文", "tool_observations": []},
        "model_responses": [{}], "latency_ms": 10, "retrieval_calls": [],
        "usage": {"usage_complete": True, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    } for arm in ("without_facts", "with_facts")]
    return case, rows


def test_report_rejects_missing_or_duplicate_pairs():
    from ai_agent.evaluation.acquisition_pair_report import summarize

    case, rows = report_rows()
    for invalid in (rows[:1], rows + rows[:1]):
        with pytest.raises(ValueError, match="缺失、重复"):
            summarize([case], invalid, 1)


def test_tool_behavior_difference_is_an_outcome_not_a_control_failure():
    from ai_agent.evaluation.acquisition_pair_report import summarize

    case, rows = report_rows()
    rows[1]["plan_trace"]["tool_observations"] = [{"content": "补查资料"}]
    rows[1]["retrieval_calls"] = [{"queries": [["q", "检索问题"]]}]
    result = summarize([case], rows, 1)
    assert result["all_control_checks_passed"] is True
    assert result["paired_checks"][0]["same_returned_tool_text"] is False


def test_system_message_difference_invalidates_control():
    from ai_agent.evaluation.acquisition_pair_report import summarize

    case, rows = report_rows()
    rows[1]["request_messages"][0]["messages"][0]["content"] = "额外规则"
    assert summarize([case], rows, 1)["all_control_checks_passed"] is False
