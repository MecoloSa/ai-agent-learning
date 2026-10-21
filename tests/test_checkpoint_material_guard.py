"""检查点资料版本保护测试。

业务模型使用替身，资料使用真实临时文件。
验证资料变化拒绝恢复，以及阶段执行中变化不能提交正常结果。
"""
import json
import os
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from ai_agent.runtime import agents
from ai_agent.persistence import checkpoint_guard
from ai_agent.runtime.orchestration import run_learning_graph
from ai_agent.runtime.plan_validation import PlanExecutionError


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    root = tmp_path / "materials"
    root.mkdir()
    document = root / "book.md"
    document.write_text("AAAA", encoding="utf-8")

    calls = {"analyse": 0, "plan": 0}
    flags = {"change_during": None}

    def analyse(*args):
        calls["analyse"] += 1
        if flags["change_during"] == "analyse":
            document.write_text("BBBB", encoding="utf-8")
        return "分析", "证据"

    def plan(*args, **kwargs):
        calls["plan"] += 1

        if flags["change_during"] == "plan":
            document.write_text("BBBB", encoding="utf-8")
            return "计划"

        if calls["plan"] == 1:
            raise PlanExecutionError(
                "invalid_plan_output", "模拟首次规划失败"
            )
        return "计划"

    monkeypatch.setattr(
        agents, "_validate_material_dir", lambda _: None
    )
    monkeypatch.setattr(agents, "analyse_materials", analyse)
    monkeypatch.setattr(agents, "create_study_plan", plan)

    saver = InMemorySaver()

    def run(**kwargs):
        return run_learning_graph(
            str(root),
            "理解资料",
            1,
            "pipeline",
            checkpointer=saver,
            thread_id="material-version-test",
            **kwargs,
        )

    return SimpleNamespace(
        root=root,
        document=document,
        calls=calls,
        flags=flags,
        run=run,
    )


@pytest.mark.parametrize(
    "operation",
    ["unchanged", "content", "add", "delete", "rename"],
)
def test_resume_checks_actual_material_version(scenario, operation):
    with pytest.raises(PlanExecutionError):
        scenario.run()

    if operation == "content":
        original_stat = scenario.document.stat()
        scenario.document.write_text("BBBB", encoding="utf-8")
        # 内容长度相同，并恢复修改时间，证明不能只依赖大小和mtime。
        os.utime(
            scenario.document,
            ns=(
                original_stat.st_atime_ns,
                original_stat.st_mtime_ns,
            ),
        )
    elif operation == "add":
        (scenario.root / "new.txt").write_text(
            "新资料", encoding="utf-8"
        )
    elif operation == "delete":
        scenario.document.unlink()
    elif operation == "rename":
        scenario.document.rename(scenario.root / "renamed.md")

    trace = {}

    if operation == "unchanged":
        assert scenario.run(
            resume=True, workflow_trace=trace
        ) == ("分析", "计划", "证据")
        assert scenario.calls == {"analyse": 1, "plan": 2}
    else:
        with pytest.raises(ValueError, match="资料版本"):
            scenario.run(resume=True, workflow_trace=trace)

        # 恢复前就拒绝，不执行额外分析或规划。
        assert scenario.calls == {"analyse": 1, "plan": 1}
        assert trace["latest_run"]["events"] == []
        assert trace["latest_run"]["status"] == "failed"

    print(json.dumps({
        "operation": operation,
        "status": trace["latest_run"]["status"],
        "node_count": len(trace["latest_run"]["events"]),
        "calls": scenario.calls,
    }, ensure_ascii=False))


@pytest.mark.parametrize("node", ["analyse", "plan"])
def test_change_during_node_is_not_committed(scenario, node):
    scenario.flags["change_during"] = node
    trace = {}

    with pytest.raises(ValueError, match="资料版本"):
        scenario.run(workflow_trace=trace)

    assert trace["latest_run"]["status"] == "failed"
    assert trace["latest_run"]["failed_node"] == node
    assert scenario.calls["analyse"] == 1
    assert scenario.calls["plan"] == (
        0 if node == "analyse" else 1
    )


def test_old_checkpoint_without_fingerprint_is_rejected(scenario):
    with pytest.raises(ValueError, match="缺少资料指纹"):
        checkpoint_guard.require_material_fingerprint(
            str(scenario.root), None
        )
