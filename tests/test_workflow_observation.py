"""节点观察测试。

用假业务函数验证顺序、状态摘要、失败定位与异常保留。
不连接模型，不保存真实资料，不把观察轨迹当成checkpoint。
"""
import json

import pytest

from ai_agent.runtime import agents
from ai_agent.runtime.plan_validation import PlanExecutionError


@pytest.fixture
def fake_steps(monkeypatch):
    calls = []

    def validate(material_dir):
        calls.append("validate_materials")
        return None

    def analyse(material_dir, goal):
        calls.append("analyse")
        return "分析正文-secret", "资料正文-secret"

    def plan(*args, **kwargs):
        calls.append("plan")
        return "计划正文-secret"

    monkeypatch.setattr(agents, "_validate_material_dir", validate)
    monkeypatch.setattr(agents, "analyse_materials", analyse)
    monkeypatch.setattr(agents, "analyse_materials_agentic", analyse)
    monkeypatch.setattr(agents, "create_study_plan", plan)

    return calls


def run(trace, mode="pipeline"):
    return agents.run_learning_assistant(
        "private-material-path",
        "private-learning-goal",
        1,
        mode,
        orchestrator="langgraph",
        workflow_trace=trace,
    )


@pytest.mark.parametrize("mode", ["pipeline", "agent"])
def test_records_normal_flow_without_body(fake_steps, mode):
    trace = {}
    result = run(trace, mode)
    record = trace["latest_run"]

    assert result == (
        "分析正文-secret",
        "计划正文-secret",
        "资料正文-secret",
    )
    assert fake_steps == ["validate_materials", "analyse", "plan"]
    assert record["status"] == "completed"
    assert [event["node"] for event in record["events"]] == fake_steps
    assert all(
        event["status"] == "returned"
        for event in record["events"]
    )

    analyse_event = record["events"][1]
    plan_event = record["events"][2]

    assert analyse_event["input"]["analysis_chars"] == 0
    assert analyse_event["output"]["analysis_chars"] == len(
        "分析正文-secret"
    )
    assert plan_event["input"]["material_context_chars"] == len(
        "资料正文-secret"
    )
    assert plan_event["output"]["plan_chars"] == len(
        "计划正文-secret"
    )
    assert all(
        event["elapsed_ms"] >= 0
        for event in record["events"]
    )

    serialized = json.dumps(record, ensure_ascii=False)
    assert "secret" not in serialized
    assert "private-material-path" not in serialized
    assert "private-learning-goal" not in serialized

    # pytest默认捕获输出；使用-s可直观看到纯摘要轨迹。
    print(json.dumps(record, ensure_ascii=False, indent=2))


def test_directory_rejection_is_not_node_failure(
    fake_steps, monkeypatch
):
    monkeypatch.setattr(
        agents, "_validate_material_dir",
        lambda _: "目录无效",
    )
    trace = {}

    assert run(trace) == ("目录无效", "", "")
    record = trace["latest_run"]

    assert record["status"] == "rejected"
    assert len(record["events"]) == 1
    assert record["events"][0]["status"] == "returned"
    assert record["events"][0]["output"]["materials_valid"] is False
    assert fake_steps == []


def test_analysis_failure_keeps_original_exception(
    fake_steps, monkeypatch
):
    original = agents.AnalysisExecutionError(
        "model_round_limit", "private-error-detail"
    )

    def fail(*args):
        fake_steps.append("analyse")
        raise original

    monkeypatch.setattr(agents, "analyse_materials", fail)
    trace = {}

    with pytest.raises(agents.AnalysisExecutionError) as caught:
        run(trace)

    assert caught.value is original
    assert fake_steps == ["validate_materials", "analyse"]

    record = trace["latest_run"]
    assert record["status"] == "failed"
    assert record["failed_node"] == "analyse"

    event = record["events"][-1]
    assert event["status"] == "failed"
    assert event["reason"] == "model_round_limit"
    assert "output" not in event
    assert "private-error-detail" not in json.dumps(record)


def test_plan_failure_is_located_at_plan(
    fake_steps, monkeypatch
):
    def fail(*args, **kwargs):
        fake_steps.append("plan")
        raise PlanExecutionError(
            "invalid_plan_output", "模拟空计划"
        )

    monkeypatch.setattr(agents, "create_study_plan", fail)
    trace = {}

    with pytest.raises(PlanExecutionError):
        run(trace)

    record = trace["latest_run"]
    assert record["status"] == "failed"
    assert record["failed_node"] == "plan"
    assert record["events"][-1]["reason"] == "invalid_plan_output"
    assert fake_steps == ["validate_materials", "analyse", "plan"]


def test_reused_receiver_gets_a_new_run(fake_steps):
    trace = {}
    run(trace)
    previous = trace["latest_run"]

    run(trace)
    current = trace["latest_run"]

    assert current["run_id"] != previous["run_id"]
    assert len(previous["events"]) == 3
    assert len(current["events"]) == 3
