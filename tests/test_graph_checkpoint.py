"""内存checkpoint的最小恢复测试。

真实图配合假业务，验证规划失败后不重跑分析。
不调用模型，不支持进程重启，不把观察轨迹当作检查点。
"""
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from ai_agent.runtime import agents
from ai_agent.persistence import checkpoint_guard
from ai_agent.runtime.orchestration import run_learning_graph
from ai_agent.runtime.plan_validation import PlanExecutionError


@pytest.fixture
def fake_steps(monkeypatch):
    calls = {"validate": 0, "analyse": 0, "plan": 0}

    # 本文件只测恢复控制流；不存在的假目录使用固定指纹。
    # 真实文件增改删由独立测试覆盖。
    monkeypatch.setattr(
        checkpoint_guard,
        "material_fingerprint",
        lambda _: "materials-v1:offline",
    )

    def validate(*args):
        calls["validate"] += 1
        return None

    def analyse(*args):
        calls["analyse"] += 1
        return "分析结果", "资料证据"

    def plan(analysis, context, *args, **kwargs):
        calls["plan"] += 1
        assert (analysis, context) == ("分析结果", "资料证据")
        if calls["plan"] == 1:
            raise PlanExecutionError(
                "invalid_plan_output", "模拟第一次规划失败"
            )
        return "学习计划"

    monkeypatch.setattr(
        agents, "_validate_material_dir", validate
    )
    monkeypatch.setattr(agents, "analyse_materials", analyse)
    monkeypatch.setattr(
        agents, "analyse_materials_agentic", analyse
    )
    monkeypatch.setattr(agents, "create_study_plan", plan)
    return calls


def run(saver, trace=None, **changes):
    arguments = {
        "material_dir": "frozen-materials",
        "learning_goal": "理解ReAct",
        "days": 1,
        "mode": "pipeline",
    }
    arguments.update(changes)
    return run_learning_graph(
        **arguments,
        checkpointer=saver,
        thread_id="checkpoint-test",
        workflow_trace=trace,
    )


def start_failed(saver, trace=None):
    with pytest.raises(PlanExecutionError):
        run(saver, trace)


def test_resume_only_reexecutes_plan(fake_steps):
    saver = InMemorySaver()
    trace = {}

    start_failed(saver, trace)
    first = trace["latest_run"]

    result = run(saver, trace, resume=True)
    resumed = trace["latest_run"]

    assert result == ("分析结果", "学习计划", "资料证据")
    assert fake_steps == {
        "validate": 1,
        "analyse": 1,
        "plan": 2,
    }
    assert first["status"] == "failed"
    assert first["failed_node"] == "plan"
    assert resumed["status"] == "completed"
    assert resumed["execution_kind"] == "resume"
    assert resumed["thread_id"] == first["thread_id"]
    assert resumed["run_id"] != first["run_id"]
    assert [
        event["node"] for event in resumed["events"]
    ] == ["plan"]

    # 展示每次尝试的摘要，不打印保存的资料正文。
    print(json.dumps({
        "first_status": first["status"],
        "first_nodes": [
            event["node"] for event in first["events"]
        ],
        "resume_status": resumed["status"],
        "resume_nodes": [
            event["node"] for event in resumed["events"]
        ],
        "call_counts": fake_steps,
    }, ensure_ascii=False, indent=2))


@pytest.mark.parametrize("change", [
    {"material_dir": "other-materials"},
    {"learning_goal": "另一个目标"},
    {"days": 2},
    {"mode": "agent"},
])
def test_changed_task_is_rejected_before_business(
    fake_steps, change
):
    saver = InMemorySaver()
    start_failed(saver)

    with pytest.raises(ValueError, match="恢复参数"):
        run(saver, resume=True, **change)

    assert fake_steps == {
        "validate": 1,
        "analyse": 1,
        "plan": 1,
    }


def test_existing_thread_cannot_be_started_again(fake_steps):
    saver = InMemorySaver()
    start_failed(saver)

    with pytest.raises(ValueError, match="已有任务"):
        run(saver)

    assert fake_steps["analyse"] == 1
    assert fake_steps["plan"] == 1


def test_completed_task_cannot_be_resumed(fake_steps):
    saver = InMemorySaver()
    start_failed(saver)
    run(saver, resume=True)

    with pytest.raises(ValueError, match="没有可重试"):
        run(saver, resume=True)

    assert fake_steps["plan"] == 2


def test_new_saver_has_no_previous_checkpoint(fake_steps):
    first_saver = InMemorySaver()
    start_failed(first_saver)

    # 新对象没有旧对象的内存数据，也说明它不支持进程重启恢复。
    empty_saver = InMemorySaver()
    with pytest.raises(ValueError, match="没有可重试"):
        run(empty_saver, resume=True)

    assert fake_steps["plan"] == 1


def test_analysis_failure_is_not_plan_recovery(
    fake_steps, monkeypatch
):
    def fail(*args):
        fake_steps["analyse"] += 1
        raise agents.AnalysisExecutionError(
            "model_round_limit", "模拟分析失败"
        )

    monkeypatch.setattr(agents, "analyse_materials", fail)
    saver = InMemorySaver()

    with pytest.raises(agents.AnalysisExecutionError):
        run(saver)

    with pytest.raises(ValueError, match="没有可重试"):
        run(saver, resume=True)

    assert fake_steps["analyse"] == 1
    assert fake_steps["plan"] == 0

@pytest.mark.parametrize("saved_revision", [None, "", 123])
def test_missing_or_invalid_workflow_revision_is_rejected(
    saved_revision,
):
    # 无版本的历史记录不能被默认为“兼容当前程序”。
    with pytest.raises(ValueError, match="缺少工作流兼容版本"):
        checkpoint_guard.require_workflow_revision(
            saved_revision
        )


def test_current_workflow_revision_is_accepted():
    checkpoint_guard.require_workflow_revision(
        checkpoint_guard.WORKFLOW_REVISION
    )


def test_incompatible_revision_blocks_resume_without_overwrite(
    fake_steps, monkeypatch
):
    saver = InMemorySaver()
    start_failed(saver)

    original_revision = checkpoint_guard.WORKFLOW_REVISION
    trace = {}

    # 临时模拟程序发生不兼容升级，不修改源码中的真实版本。
    # context结束后，monkeypatch会自动恢复原值。
    with monkeypatch.context() as patch:
        patch.setattr(
            checkpoint_guard,
            "WORKFLOW_REVISION",
            original_revision + "-incompatible",
        )

        with pytest.raises(
            ValueError, match="工作流兼容版本与检查点不一致"
        ):
            run(saver, trace, resume=True)

    rejected = trace["latest_run"]

    # 动态评估：拒绝发生在进入图节点之前，不追加模型业务调用。
    assert rejected["status"] == "failed"
    assert rejected["events"] == []
    assert fake_steps == {
        "validate": 1,
        "analyse": 1,
        "plan": 1,
    }

    # 回到兼容版本后仍能恢复，证明刚才的拒绝没有覆盖旧状态。
    result = run(saver, resume=True)

    assert result == ("分析结果", "学习计划", "资料证据")
    assert fake_steps == {
        "validate": 1,
        "analyse": 1,
        "plan": 2,
    }

    print(json.dumps({
        "incompatible_resume_status": rejected["status"],
        "incompatible_resume_nodes": [],
        "compatible_resume_succeeded": True,
        "calls_after_compatible_resume": fake_steps,
    }, ensure_ascii=False, indent=2))
