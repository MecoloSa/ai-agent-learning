"""恢复问答入口与证据边界测试。

真实读写临时快照、校验临时目录；替换模型和问答执行，不访问网络。
边界提示是给模型的约束，测试其存在不等于已证明模型语义遵守。
"""
import builtins
import json
import sys

import pytest

import ai_agent.main as main_module
import ai_agent.persistence.memory as memory_module
from ai_agent.learning.learning_state import LearningState, LearningTask
from ai_agent.persistence.state_store import save_learning_state, load_learning_state


def make_state():
    return LearningState(
        learning_goal="理解ReAct",
        plan_markdown="比较正常结束与保护性停止。",
        initial_plan_markdown="最初计划",
        context_revision=4,
        tasks=(LearningTask(
            task_id="task-001", title="理解终止条件",
            understanding_status="self_reported_unclear",
            latest_feedback="读完了，仍不理解。", feedback_source="user_report",
        ),),
    )


def unexpected(*args, **kwargs):
    raise AssertionError("恢复不能进入初始生成或意外创建模型")


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
def test_resume_chat_passes_full_state_and_skips_generation(
    monkeypatch, tmp_path, strategy,
):
    state = make_state()
    snapshot = save_learning_state(state, tmp_path / "state.json")
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "notes.md").write_text("本次资料", encoding="utf-8")
    received = []
    monkeypatch.setattr(sys, "argv", [
        "agent", "--resume-state", str(snapshot), "--interactive",
        "--materials", str(materials), "--memory-strategy", strategy,
    ])
    monkeypatch.setattr(main_module, "_run_initial_workflow", unexpected)
    monkeypatch.setattr(main_module, "run_state_recovery_session", unexpected)
    monkeypatch.setattr(main_module, "create_llm", unexpected)
    saved = []
    monkeypatch.setattr(
        main_module, "save_current_learning_state",
        lambda state, _path: (lambda updated: (saved.append(updated) or tmp_path / "associated.json"))(state),
    )
    monkeypatch.setattr(
        main_module, "run_interactive_session",
        lambda **kwargs: received.append(kwargs),
    )

    main_module.main()

    assert len(received) == 1
    args = received[0]
    assert args["initial_state"].tasks == state.tasks
    assert args["initial_state"].context_revision == state.context_revision
    assert args["initial_state"].material_binding.root_dir == str(materials.resolve())
    assert saved == [args["initial_state"]]
    assert args["plan"] == state.plan_markdown
    assert args["learning_goal"] == state.learning_goal
    assert args["analysis"] == args["material_context"] == ""
    assert args["material_dir"] == str(materials.resolve())
    assert args["strategy"] == strategy
    assert load_learning_state(snapshot) == state


def test_legacy_resume_without_materials_falls_back(monkeypatch, tmp_path):
    state = make_state()
    snapshot = save_learning_state(state, tmp_path / "state.json")
    monkeypatch.setattr(sys, "argv", [
        "agent", "--resume-state", str(snapshot), "--interactive",
    ])
    received = []
    monkeypatch.setattr(main_module, "run_interactive_session", unexpected)
    monkeypatch.setattr(main_module, "create_llm", unexpected)
    monkeypatch.setattr(main_module, "run_state_recovery_session", received.append)
    main_module.main()
    assert received == [state]


@pytest.mark.parametrize("kind", ["missing", "empty", "unsupported"])
def test_bad_materials_fall_back_without_chat(monkeypatch, tmp_path, kind):
    state = make_state()
    snapshot = save_learning_state(state, tmp_path / "state.json")
    root = tmp_path / "materials"
    if kind != "missing":
        root.mkdir()
    if kind == "unsupported":
        (root / "image.png").write_bytes(b"not-a-real-image")
    monkeypatch.setattr(sys, "argv", [
        "agent", "--resume-state", str(snapshot), "--interactive",
        "--materials", str(root),
    ])
    monkeypatch.setattr(main_module, "run_interactive_session", unexpected)
    received = []
    monkeypatch.setattr(main_module, "run_state_recovery_session", received.append)
    monkeypatch.setattr(main_module, "_run_initial_workflow", unexpected)
    monkeypatch.setattr(main_module, "create_llm", unexpected)
    main_module.main()
    assert received == [state]
    assert load_learning_state(snapshot) == state


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
@pytest.mark.parametrize("answer", ["y", "n"])
def test_resume_boundary_survives_plan_confirmation(
    monkeypatch, tmp_path, strategy, answer,
):
    state = make_state()
    requests, saved = [], []
    inputs = iter(["请修改计划", answer, "解释当前计划", "exit"])

    def fake_turn(**kwargs):
        requests.append(kwargs["request_messages"])
        if len(requests) == 1:
            return "===NEW_PLAN_START===新计划===NEW_PLAN_END==="
        return "只解释，不修改。"

    def fake_save(updated):
        saved.append(updated)
        return tmp_path / "not-written.json"

    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(main_module, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main_module, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(main_module, "_invoke_interactive_turn", fake_turn)
    monkeypatch.setattr(main_module, "save_current_learning_state", lambda state, _path: (fake_save)(state))
    monkeypatch.setattr(
        memory_module.ConversationMemory, "_maybe_compress", lambda self: None
    )

    main_module.run_interactive_session(
        analysis="", plan=state.plan_markdown, material_context="",
        learning_goal=state.learning_goal, strategy=strategy,
        material_dir=str(tmp_path), initial_state=state,
    )

    assert len(requests) == 2
    for request in requests:
        assert main_module.RESUMED_SESSION_RULES in str(request[0].content)
    assert len(saved) == 2
    assert all(item.context_revision == (5 if answer == "y" else 4) for item in saved)
    assert all(item.tasks == state.tasks for item in saved)
    states = [
        json.loads(str(message.content)) for message in requests[1]
        if message.type == "human" and str(message.content).startswith("{")
    ]
    states = [item for item in states if item.get("context_type") == "current_learning_state"]
    assert len(states) == 1
    assert states[-1]["context_revision"] == (5 if answer == "y" else 4)
    assert states[-1]["tasks"] == [task.model_dump() for task in state.tasks]
