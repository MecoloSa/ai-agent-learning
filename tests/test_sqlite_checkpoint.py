"""SQLite检查点跨进程恢复测试。

使用真实临时资料和数据库，模型业务使用替身，不请求在线服务。
验证进程退出后可恢复，以及资料变化仍会阻止旧分析被复用。
"""
import json
import subprocess
import sys

import pytest


# 这段脚本分别在两个独立Python进程中执行。
# 两次调用共享的只有磁盘上的资料和SQLite数据库，不共享内存。
WORKER = r"""
import json
import sys
from unittest.mock import patch

from ai_agent.runtime import agents
from ai_agent.persistence.checkpoint_store import open_graph_checkpoint
from ai_agent.runtime.orchestration import run_learning_graph
from ai_agent.runtime.plan_validation import PlanExecutionError

db_path, material_dir, phase = sys.argv[1:]
calls = {"analyse": 0, "plan": 0}
trace = {}


def analyse(*args):
    calls["analyse"] += 1
    return "分析结果", "资料证据"


def plan(analysis, context, *args, **kwargs):
    calls["plan"] += 1
    # 恢复时必须读到首次保存的分析和证据，而不是空状态。
    assert (analysis, context) == ("分析结果", "资料证据")
    if phase == "first":
        raise PlanExecutionError(
            "invalid_plan_output", "模拟规划失败"
        )
    return "学习计划"


with patch.object(
    agents, "_validate_material_dir", return_value=None
), patch.object(
    agents, "analyse_materials", analyse
), patch.object(
    agents, "create_study_plan", plan
):
    with open_graph_checkpoint(db_path) as saver:
        try:
            run_learning_graph(
                material_dir,
                "固定目标",
                1,
                "pipeline",
                checkpointer=saver,
                thread_id="sqlite-test",
                resume=phase == "resume",
                workflow_trace=trace,
            )
            status = "completed"
        except PlanExecutionError:
            status = "plan_failed"
        except ValueError:
            status = "blocked"

print(json.dumps({
    "status": status,
    "calls": calls,
    "nodes": [
        event["node"]
        for event in trace["latest_run"]["events"]
    ],
}, ensure_ascii=False))
"""


def run_worker(db_path, material_dir, phase):
    result = subprocess.run(
        [
            sys.executable,
            "-X", "utf8",
            "-B",
            "-c", WORKER,
            str(db_path),
            str(material_dir),
            phase,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
    )
    # 子进程意外崩溃不能被当成“正确拒绝恢复”。
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("changed", [False, True])
def test_sqlite_resume_across_processes(tmp_path, changed):
    material_dir = tmp_path / "materials"
    material_dir.mkdir()
    document = material_dir / "book.md"
    document.write_text("AAAA", encoding="utf-8")

    # 数据库不放进资料目录，避免存储文件与学习资料混在一起。
    db_path = tmp_path / "checkpoints" / "workflow.sqlite"

    first = run_worker(db_path, material_dir, "first")
    assert first["status"] == "plan_failed"
    assert first["calls"] == {"analyse": 1, "plan": 1}
    assert db_path.is_file()

    if changed:
        document.write_text("BBBB", encoding="utf-8")

    second = run_worker(db_path, material_dir, "resume")

    if changed:
        assert second["status"] == "blocked"
        assert second["calls"] == {"analyse": 0, "plan": 0}
        assert second["nodes"] == []
    else:
        assert second["status"] == "completed"
        assert second["calls"] == {"analyse": 0, "plan": 1}
        assert second["nodes"] == ["plan"]

    # 动态评估：恢复进程不应再次分析；资料变化时连规划也不能执行。
    print(json.dumps({
        "material_changed": changed,
        "first_process": first,
        "second_process": second,
    }, ensure_ascii=False))
