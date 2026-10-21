"""评测入口的节点轨迹接线与JSONL保存测试。

使用真实图控制流、假分析和假规划，不调用模型。
验证成功、失败记录可保存，且原legacy替身接口保持兼容。
"""
import json
import sys

import pytest

from ai_agent.runtime import agents
from ai_agent.evaluation import runner
from ai_agent.runtime.plan_validation import PlanExecutionError


@pytest.fixture
def fake_steps(monkeypatch):
    monkeypatch.setattr(
        agents, "_validate_material_dir", lambda _: None
    )
    monkeypatch.setattr(
        agents, "analyse_materials",
        lambda *args: ("分析", "证据"),
    )
    monkeypatch.setattr(
        agents, "analyse_materials_agentic",
        lambda *args: ("分析", "证据"),
    )
    monkeypatch.setattr(
        agents, "create_study_plan",
        lambda *args, **kwargs: "计划",
    )


def case():
    return runner.EvalCase(
        "workflow-case", "materials", "理解ReAct", 1
    )


def test_graph_trace_survives_jsonl_roundtrip(
    fake_steps, tmp_path
):
    records = [
        runner.run_case(
            case(), "pipeline", orchestrator="langgraph"
        )
        for _ in range(2)
    ]

    output = tmp_path / "results.jsonl"
    runner.write_record(records, output)
    rows = [
        json.loads(line)
        for line in output.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    for row in rows:
        assert row["success"] is True
        assert row["orchestrator"] == "langgraph"
        assert row["workflow_trace"]["status"] == "completed"
        assert [
            event["node"]
            for event in row["workflow_trace"]["events"]
        ] == ["validate_materials", "analyse", "plan"]

    # 每个案例独立记录，不能复用上一案例的图ID。
    assert rows[0]["workflow_trace"]["run_id"] != (
        rows[1]["workflow_trace"]["run_id"]
    )

    # 使用-s可看到保存后重新读取的轨迹。
    print(json.dumps(
        rows[0]["workflow_trace"],
        ensure_ascii=False,
        indent=2,
    ))


@pytest.mark.parametrize("node", ["analyse", "plan"])
def test_failed_node_is_saved(fake_steps, monkeypatch, tmp_path, node):
    def fail(*args, **kwargs):
        if node == "analyse":
            raise agents.AnalysisExecutionError(
                "model_round_limit", "模拟分析失败"
            )
        raise PlanExecutionError(
            "invalid_plan_output", "模拟空计划"
        )

    monkeypatch.setattr(
        agents,
        "analyse_materials" if node == "analyse"
        else "create_study_plan",
        fail,
    )

    record = runner.run_case(
        case(), "pipeline", orchestrator="langgraph"
    )
    output = tmp_path / "failure.jsonl"
    runner.write_record([record], output)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["success"] is False
    assert saved["plan"] == ""
    assert saved["workflow_trace"]["status"] == "failed"
    assert saved["workflow_trace"]["failed_node"] == node
    assert saved["workflow_trace"]["events"][-1]["status"] == "failed"

    if node == "analyse":
        assert [
            event["node"]
            for event in saved["workflow_trace"]["events"]
        ] == ["validate_materials", "analyse"]


def test_directory_rejection_is_saved(fake_steps, monkeypatch):
    monkeypatch.setattr(
        agents, "_validate_material_dir",
        lambda _: "目录无效",
    )

    record = runner.run_case(
        case(), "pipeline", orchestrator="langgraph"
    )

    assert record.success is False
    assert record.error is None
    assert record.workflow_trace["status"] == "rejected"
    assert len(record.workflow_trace["events"]) == 1


def test_custom_execute_is_not_mislabeled_as_graph():
    calls = []

    def fake_execute(*args):
        calls.append(args)
        return "分析", "计划", "证据"

    record = runner.run_case(
        case(), "pipeline", execute=fake_execute
    )
    assert record.orchestrator == "legacy"
    assert record.workflow_trace == {}
    assert len(calls) == 1

    with pytest.raises(ValueError, match="仅支持 legacy"):
        runner.run_case(
            case(),
            "pipeline",
            execute=fake_execute,
            orchestrator="langgraph",
        )

    assert len(calls) == 1


def test_cli_passes_orchestrator(monkeypatch, tmp_path):
    captured = []
    output = tmp_path / "cli.jsonl"

    monkeypatch.setattr(
        runner, "load_cases", lambda _: [case()]
    )

    def fake_run_case(case, mode, *, orchestrator):
        captured.append(orchestrator)
        return runner.RunRecord(
            run_id="test-run",
            case_id=case.case_id,
            mode=mode,
            success=True,
            latency_ms=0,
            citations=[],
            analysis="分析",
            plan="计划",
            orchestrator=orchestrator,
        )

    monkeypatch.setattr(runner, "run_case", fake_run_case)
    monkeypatch.setattr(sys, "argv", [
        "runner",
        "--cases", "unused.jsonl",
        "--output", str(output),
        "--modes", "pipeline",
        "--orchestrator", "langgraph",
    ])

    runner.main()

    assert captured == ["langgraph"]
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["orchestrator"] == "langgraph"
