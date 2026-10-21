"""资料关联与自动恢复测试。

临时目录真实读写，模型和问答使用替身；不访问网络、不改真实快照。
检查目录重定位、字节变化、旧格式兼容及状态保真，不把指纹一致当作答案正确。
"""
import builtins
import json
import sys

import pytest
from pydantic import ValidationError

import ai_agent.main as main_module
import ai_agent.persistence.memory as memory_module
from ai_agent.learning.learning_state import (
    LearningState, LearningTask, MaterialBinding, UserFeedback,
    apply_user_feedback, update_learning_context,
)
from ai_agent.persistence.session_materials import create_material_binding, resolve_resume_materials
from ai_agent.persistence.state_store import save_learning_state, load_learning_state


def make_state():
    return LearningState(
        learning_goal="理解ReAct", plan_markdown="比较终止条件。",
        initial_plan_markdown="最初计划", context_revision=2,
        tasks_need_review=True,
        tasks=(LearningTask(
            task_id="confirm-demo-001", title="阅读终止条件",
            activity_status="completed", understanding_status="self_reported_unclear",
            latest_feedback="读完但不理解。", feedback_source="user_report",
        ),),
    )


def make_materials(tmp_path, name="materials", text="版本A"):
    root = tmp_path / name
    root.mkdir()
    (root / "notes.md").write_text(text, encoding="utf-8")
    return root


def test_binding_roundtrip_and_no_revision_change(tmp_path):
    original = make_state()
    root = make_materials(tmp_path)
    associated, notice = resolve_resume_materials(original, str(root))
    assert "首次关联" in notice
    assert associated.material_binding.root_dir == str(root.resolve())
    assert associated.context_revision == original.context_revision
    assert associated.tasks == original.tasks
    assert original.material_binding is None
    path = save_learning_state(associated, tmp_path / "state.json")
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "learning-state-v5"
    assert load_learning_state(path) == associated
    same, notice = resolve_resume_materials(associated)
    assert same is associated
    assert "检查通过" in notice


def test_binding_survives_feedback_plan_and_registration(tmp_path):
    state, _ = resolve_resume_materials(make_state(), str(make_materials(tmp_path)))
    updated = apply_user_feedback(state, UserFeedback(
        task_id="confirm-demo-001", text="理解了。", understanding_status="self_reported_clear",
    ))
    updated = update_learning_context(updated, plan_markdown="新的补学计划。")
    updated, _ = main_module.handle_learning_command(
        updated, '/task {"task_id":"second","title":"观点比较"}',
    )
    assert updated.material_binding == state.material_binding
    assert updated.initial_plan_markdown == state.initial_plan_markdown


def test_changed_materials_detected_without_reset(tmp_path):
    root = make_materials(tmp_path)
    original, _ = resolve_resume_materials(make_state(), str(root))
    (root / "notes.md").write_text("版本B", encoding="utf-8")
    updated, notice = resolve_resume_materials(original)
    assert "资料变化" in notice
    assert updated.material_binding.fingerprint != original.material_binding.fingerprint
    assert updated.tasks == original.tasks
    assert updated.context_revision == original.context_revision


def test_relocated_identical_materials_keep_fingerprint(tmp_path):
    first = make_materials(tmp_path, "first")
    second = make_materials(tmp_path, "second")
    state, _ = resolve_resume_materials(make_state(), str(first))
    updated, notice = resolve_resume_materials(state, str(second))
    assert "位置已更新" in notice
    assert updated.material_binding.fingerprint == state.material_binding.fingerprint
    assert updated.material_binding.root_dir == str(second.resolve())


@pytest.mark.parametrize("version", ["learning-state-v1", "learning-state-v2"])
def test_old_snapshots_have_no_invented_binding(tmp_path, version):
    data = make_state().model_dump(mode="json")
    data.pop("material_binding", None)
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": version, "state": data}), encoding="utf-8")
    assert load_learning_state(path).material_binding is None


def test_relative_binding_path_rejected():
    with pytest.raises(ValidationError):
        MaterialBinding(root_dir="relative", fingerprint="materials-v1:" + "a" * 64)


@pytest.mark.parametrize("state_only", [False, True])
def test_bound_snapshot_auto_chat_or_explicit_state_only(monkeypatch, tmp_path, state_only):
    state, _ = resolve_resume_materials(make_state(), str(make_materials(tmp_path)))
    path = save_learning_state(state, tmp_path / "state.json")
    args = ["agent", "--resume-state", str(path)] + (["--state-only"] if state_only else [])
    monkeypatch.setattr(sys, "argv", args)
    chats, records = [], []
    monkeypatch.setattr(main_module, "run_interactive_session", lambda **kw: chats.append(kw))
    monkeypatch.setattr(main_module, "run_state_recovery_session", records.append)
    monkeypatch.setattr(main_module, "_run_initial_workflow", lambda *a: pytest.fail("不能重新生成"))
    monkeypatch.setattr(main_module, "save_current_learning_state", lambda *a: pytest.fail("无变化不保存"))
    main_module.main()
    assert len(chats) == (0 if state_only else 1)
    assert records == ([state] if state_only else [])
    if chats:
        assert chats[0]["initial_state"] == state


def test_missing_bound_directory_falls_back_without_model(monkeypatch, tmp_path):
    root = make_materials(tmp_path)
    state, _ = resolve_resume_materials(make_state(), str(root))
    path = save_learning_state(state, tmp_path / "state.json")
    root.rename(tmp_path / "moved")
    records = []
    monkeypatch.setattr(sys, "argv", ["agent", "--resume-state", str(path)])
    monkeypatch.setattr(main_module, "run_state_recovery_session", records.append)
    monkeypatch.setattr(main_module, "run_interactive_session", lambda **kw: pytest.fail("不能进入问答"))
    monkeypatch.setattr(main_module, "create_llm", lambda **kw: pytest.fail("不能创建模型"))
    main_module.main()
    assert records == [state]
    assert load_learning_state(path) == state


def test_interactive_save_includes_binding(monkeypatch, tmp_path):
    binding = create_material_binding(str(make_materials(tmp_path)))
    inputs = iter(["/save-state", "exit"])
    path = tmp_path / "saved.json"
    import ai_agent.learning.learning_commands as commands
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(main_module, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main_module, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(commands, "resolve_project_path", lambda _: path)
    monkeypatch.setattr(memory_module.ConversationMemory, "_maybe_compress", lambda self: None)
    main_module.run_interactive_session(
        analysis="", plan="计划", material_context="", learning_goal="目标",
        strategy="window", material_dir=binding.root_dir, material_binding=binding,
    )
    assert load_learning_state(path).material_binding == binding


@pytest.mark.parametrize("extra", [[], ["--interactive"], ["--materials", "somewhere"]])
def test_state_only_rejects_conflicting_parameters(monkeypatch, extra):
    # 第一项缺少--resume-state；其余两项与仅管理状态模式冲突。
    args = ["agent", "--state-only"]
    if extra:
        args.extend(["--resume-state", "not-read.json", *extra])
    else:
        args.extend(["--goal", "新目标"])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as exc:
        main_module.main()
    assert exc.value.code == 2
