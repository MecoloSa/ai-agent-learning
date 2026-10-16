import pytest
import json
import ai_agent.evaluation.runner as runner_module
from ai_agent.evaluation.runner import EvalCase, run_case, write_record


def test_run_case_records_success():
    def fake_execute(material_dir, goal, days, mode):
        return (
            "资料分析。[source=a.md, chunk=0]",
            "第1天计划。[source=a.md, chunk=0]",
            "证据",
        )

    result = run_case(
        EvalCase(
            case_id="case-001",
            material_dir="materials",
            learning_goal="学习 Agent",
            days=1,
        ),
        "pipeline",
        execute=fake_execute,
    )

    assert result.success is True
    assert result.error is None
    assert result.citations == ["[source=a.md, chunk=0]"]


def test_run_case_records_error_without_interrupting_batch():
    def fake_execute(material_dir, goal, days, mode):
        raise TimeoutError("模型超时")

    result = run_case(
        EvalCase(
            case_id="case-002",
            material_dir="materials",
            learning_goal="学习 Agent",
            days=1,
        ),
        "agent",
        execute=fake_execute,
    )

    assert result.success is False
    assert result.error == "TimeoutError: 模型超时"

def test_write_record_preserves_existing_results(tmp_path):
    output = tmp_path / "results.jsonl"
    output.write_text("old-results\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_record([], output)

    # 验证实际内容仍然存在，不仅检查是否抛出异常。
    assert output.read_text(encoding="utf-8") == "old-results\n"

def test_runner_saves_plan_evidence_trace(monkeypatch, tmp_path):
    def fake_assistant(material_dir, goal, days, mode, *, plan_evidence_trace):
        plan_evidence_trace.update({
            "trace_version": "plan-evidence-v1",
            "initial_evidence": "[source=a.md, chunk=0]\n正文",
            "tool_observations": [],
            "requests": [],
        })
        return "分析", "计划", "原始上下文"

    monkeypatch.setattr(
        runner_module, "run_learning_assistant", fake_assistant
    )
    record = run_case(
        EvalCase("trace-case", "materials", "学习Agent", 1),
        "pipeline",
    )
    output = tmp_path / "trace.jsonl"
    write_record([record], output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["plan_evidence_trace"]["initial_evidence"] == (
        "[source=a.md, chunk=0]\n正文"
    )
    assert saved["success"] is True