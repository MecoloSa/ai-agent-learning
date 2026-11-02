"""采用/拒绝后，用另一Python进程验证真实CLI恢复，不评价模型语义。"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai_agent.persistence.session_materials import create_material_binding
from ai_agent.persistence.state_store import load_learning_state, save_learning_state
from ai_agent.learning.learning_state import LearningState


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/evaluation"
PROMPT = (
    "请依据已有学习状态，为还没理解的部分调整本次后续安排，总计不超过20分钟。"
    "只给完整计划候选供确认，不改变学习目标，不更改或重建已登记任务，"
    "不恢复已取消任务，不新增或改写反馈。"
)
RESUME_PROMPT = "只复述当前安排和已确认记录，不修改状态"
CANDIDATE = (
    "# 后续学习安排\n"
    "活动1：对比两类停止的触发条件，5分钟。\n"
    "活动2：口头解释区别并举例，10分钟。\n总计15分钟。"
)


def run_worker(directory, source, inputs, response):
    directory.mkdir()
    current = directory / "current.json"
    observation = directory / "request.json"
    spec = directory / "input.json"
    spec.write_text(json.dumps({
        "source": str(source), "current": str(current),
        "inputs": inputs, "response": response, "observation": str(observation),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # 每次subprocess.run均创建独立解释器；只通过磁盘文件传递状态。
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B",
         str(ROOT / "tests/helpers/feedback_plan_process_worker.py"), str(spec)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=45,
    )
    (directory / "console.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert current.is_file(), "必须实际落盘，不能只检查进程退出码"
    return current, json.loads(observation.read_text(encoding="utf-8"))


@pytest.mark.parametrize("decision", ["y", "n"], ids=["adopt", "decline"])
def test_plan_and_records_survive_independent_cli_processes(tmp_path, decision):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "notes.md").write_text("跨进程回归使用的合成资料。", encoding="utf-8")
    original = load_learning_state(FIXTURES / "feedback_plan/adoption_state.json")
    initial = LearningState.model_validate({
        **original.model_dump(),
        "material_binding": create_material_binding(str(materials)).model_dump(),
    })
    source = save_learning_state(initial, tmp_path / "source.json")
    initial_bytes = source.read_bytes()
    adopted = decision == "y"
    first_path, first_requests = run_worker(
        tmp_path / "first", source, [PROMPT, "v", decision, "exit"],
        f"===NEW_PLAN_START===\n{CANDIDATE}\n===NEW_PLAN_END===",
    )
    first = load_learning_state(first_path)
    assert len(first_requests) == 1
    assert first.plan_markdown == (CANDIDATE if adopted else initial.plan_markdown)
    assert first.context_revision == (2 if adopted else 1)
    assert first.tasks_need_review is adopted
    for field in ("learning_goal", "initial_plan_markdown", "initial_learning_goal",
                  "tasks", "feedback_records", "material_binding"):
        assert getattr(first, field) == getattr(initial, field)
    assert first.conversation_memory.strategy == "window"
    first_bytes = first_path.read_bytes()

    second_path, requests = run_worker(
        tmp_path / "second", first_path, [RESUME_PROMPT, "exit"],
        "这里只读取当前记录，不产生候选或反馈。",
    )
    assert len(requests) == 1
    assert requests[0]["user_input"] == RESUME_PROMPT
    messages = requests[0]["request_messages"]
    assert first.plan_markdown in messages[0]["content"]
    views = [json.loads(m["content"]) for m in messages
             if m["type"] == "human" and isinstance(m["content"], str)
             and m["content"].startswith('{"context_type": "current_learning_state"')]
    assert len(views) == 1
    assert views[0]["tasks"] == [t.model_dump(mode="json") for t in first.tasks]
    assert views[0]["context_revision"] == first.context_revision
    # 恢复的不只是业务状态：上一问答和程序确认结论也要进入本轮请求。
    assert any(m["type"] == "human" and m["content"] == PROMPT for m in messages)
    assert any(m["type"] == "ai" and ("confirmed" if adopted else "cancelled")
               in m["content"] for m in messages)
    second = load_learning_state(second_path)
    assert second.model_dump(exclude={"conversation_memory"}) == first.model_dump(
        exclude={"conversation_memory"},
    )
    assert second.conversation_memory.strategy == "window"
    assert len(second.conversation_memory.messages) == len(first.conversation_memory.messages) + 2
    assert first_path.read_bytes() == first_bytes
    assert source.read_bytes() == initial_bytes
