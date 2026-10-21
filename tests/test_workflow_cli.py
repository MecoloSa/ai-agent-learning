"""工作流检查点命令行接线测试。

使用真实临时资料和SQLite数据库，分析与规划使用替身。
验证参数拒绝、失败不导出、恢复只执行规划及连接关闭后的重开。
跨进程恢复由test_sqlite_checkpoint.py单独覆盖。
"""
import json
import sys
from types import SimpleNamespace

import pytest

from ai_agent.runtime import agents
import ai_agent.main as main_module
from ai_agent.runtime.plan_validation import PlanExecutionError


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "book.md").write_text(
        "ReAct学习资料", encoding="utf-8"
    )

    calls = {"analyse": 0, "plan": 0}

    def analyse(*args):
        calls["analyse"] += 1
        return "分析结果", "资料证据"

    def plan(*args, **kwargs):
        calls["plan"] += 1
        if calls["plan"] == 1:
            raise PlanExecutionError(
                "invalid_plan_output", "模拟首次规划失败"
            )
        return "学习计划"

    monkeypatch.setattr(
        agents, "_validate_material_dir", lambda _: None
    )
    monkeypatch.setattr(agents, "analyse_materials", analyse)
    monkeypatch.setattr(agents, "create_study_plan", plan)
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(log_level="WARNING"),
    )

    from ai_agent.runtime import workflow_audit

    audit_dir = tmp_path / "workflow_traces"
    monkeypatch.setattr(
        workflow_audit,
        "DEFAULT_AUDIT_DIR",
        audit_dir,
    )

    return SimpleNamespace(
        materials=materials,
        db=tmp_path / "workflow.sqlite",
        output=tmp_path / "plan.md",
        calls=calls,
        audit_dir=audit_dir,
    )


def arguments(scenario):
    return [
        "ai_agent",
        "--materials", str(scenario.materials),
        "--goal", "理解ReAct",
        "--days", "1",
        "--mode", "pipeline",
        "--orchestrator", "langgraph",
        "--checkpoint-db", str(scenario.db),
        "--workflow-id", "cli-test",
        "--output", str(scenario.output),
    ]


def test_cli_failed_plan_then_resume_exports(
    scenario, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", arguments(scenario))

    with pytest.raises(SystemExit) as caught:
        main_module.main()

    assert caught.value.code == 1
    assert scenario.db.is_file()
    assert not scenario.output.exists()
    assert scenario.calls == {"analyse": 1, "plan": 1}

    # 第二次main会重新打开数据库，不复用第一次的连接。
    monkeypatch.setattr(
        sys, "argv",
        arguments(scenario) + ["--resume-workflow"],
    )
    main_module.main()

    assert scenario.calls == {"analyse": 1, "plan": 2}
    assert "学习计划" in scenario.output.read_text(
        encoding="utf-8"
    )

    output = capsys.readouterr().out
    assert "状态=completed 本次节点数=1" in output

    attempts = [
        json.loads(path.read_text(encoding="utf-8"))["attempt"]
        for path in scenario.audit_dir.glob("*.json")
    ]
    assert len(attempts) == 2

    by_kind = {
        attempt["execution_kind"]: attempt
        for attempt in attempts
    }
    assert by_kind["new"]["status"] == "failed"
    assert by_kind["resume"]["status"] == "completed"
    assert [
        event["node"]
        for event in by_kind["resume"]["events"]
    ] == ["plan"]

    # 同一工作流，不同执行尝试。
    assert (
        by_kind["new"]["thread_id"]
        == by_kind["resume"]["thread_id"]
    )
    assert (
        by_kind["new"]["run_id"]
        != by_kind["resume"]["run_id"]
    )


def test_resume_missing_database_does_not_create_it(
    scenario, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys, "argv",
        arguments(scenario) + ["--resume-workflow"],
    )

    with pytest.raises(SystemExit) as caught:
        main_module.main()

    assert caught.value.code == 1
    assert not scenario.db.exists()
    assert not scenario.output.exists()
    assert scenario.calls == {"analyse": 0, "plan": 0}
    assert "恢复数据库不存在" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [
    ["--checkpoint-db", "unused.sqlite"],
    ["--workflow-id", "only-id"],
    ["--resume-workflow"],
    [
        "--checkpoint-db", "unused.sqlite",
        "--workflow-id", "task",
        "--orchestrator", "legacy",
    ],
    [
        "--checkpoint-db", "unused.sqlite",
        "--workflow-id", "task",
        "--orchestrator", "langgraph",
        "--interactive",
    ],
    [
        "--checkpoint-db", "unused.sqlite",
        "--workflow-id", " ",
        "--orchestrator", "langgraph",
    ],
])
def test_invalid_options_rejected_before_settings(
    monkeypatch, extra
):
    def forbidden():
        pytest.fail("参数错误应在读取运行设置前拒绝")

    monkeypatch.setattr(
        main_module, "get_settings", forbidden
    )
    monkeypatch.setattr(
        sys, "argv",
        ["ai_agent", "--goal", "理解ReAct", *extra],
    )

    with pytest.raises(SystemExit) as caught:
        main_module.main()

    assert caught.value.code == 2


def test_state_recovery_cannot_mix_with_workflow(monkeypatch):
    def forbidden():
        pytest.fail("不能进入状态加载或模型执行")

    monkeypatch.setattr(
        main_module, "get_settings", forbidden
    )
    monkeypatch.setattr(sys, "argv", [
        "ai_agent",
        "--resume-state", "unused-state.json",
        "--checkpoint-db", "unused.sqlite",
        "--workflow-id", "task",
        "--orchestrator", "langgraph",
    ])

    with pytest.raises(SystemExit) as caught:
        main_module.main()

    assert caught.value.code == 2

def test_audit_whitelist_and_no_overwrite(tmp_path):
    """摘要不夹带正文，重复保存不能覆盖旧运行记录。"""
    from uuid import uuid4
    from ai_agent.runtime.workflow_audit import save_workflow_attempt

    trace = {
        "run_id": uuid4().hex,
        "thread_id": "task-1",
        "execution_kind": "new",
        "status": "completed",
        "analysis": "不应保存的正文",
        "events": [{
            "node": "analyse",
            "status": "returned",
            "elapsed_ms": 10,
            "raw_messages": "不应保存的模型消息",
            "output": {
                "analysis_chars": 20,
                "material_context_chars": 30,
                "plan_chars": 0,
                "materials_valid": True,
                "full_text": "不应保存的资料",
            },
        }],
    }

    path = save_workflow_attempt(
        trace,
        tmp_path / "workflow.sqlite",
        directory=tmp_path / "audit",
    )
    original = path.read_bytes()
    content = original.decode("utf-8")

    assert "不应保存" not in content
    assert '"analysis_chars": 20' in content

    with pytest.raises(FileExistsError):
        save_workflow_attempt(
            trace,
            tmp_path / "workflow.sqlite",
            directory=tmp_path / "audit",
        )

    assert path.read_bytes() == original


def test_audit_failure_does_not_change_business_result(
    scenario, monkeypatch, capsys
):
    """存档失败既不能掩盖规划失败，也不能阻止成功结果导出。"""
    from ai_agent.runtime import workflow_audit

    def fail_save(*args, **kwargs):
        raise OSError("模拟轨迹存储失败")

    monkeypatch.setattr(
        workflow_audit,
        "save_workflow_attempt",
        fail_save,
    )

    monkeypatch.setattr(sys, "argv", arguments(scenario))
    with pytest.raises(SystemExit) as caught:
        main_module.main()

    assert caught.value.code == 1
    assert not scenario.output.exists()

    monkeypatch.setattr(
        sys, "argv",
        arguments(scenario) + ["--resume-workflow"],
    )
    main_module.main()

    assert scenario.output.exists()
    assert scenario.calls == {"analyse": 1, "plan": 2}
    assert "轨迹] 保存失败" in capsys.readouterr().out
