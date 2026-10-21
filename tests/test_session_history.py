"""沿用full/window/summary的持久化与恢复测试。

使用假摘要模型和临时快照，验证真实记忆对象的导出/恢复，
而不是另建固定轮数队列。不访问网络，也不写入真实用户快照。
"""
import builtins
import json
import sys

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

import ai_agent.main as main_module
import ai_agent.persistence.memory as memory_module
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.memory import ConversationMemory
from ai_agent.persistence.memory_state import MemorySnapshot
from ai_agent.persistence.session_materials import resolve_resume_materials
from ai_agent.persistence.state_store import save_learning_state, load_learning_state


def make_state():
    return LearningState(
        learning_goal="原目标", plan_markdown="原计划",
        context_revision=2, tasks_need_review=True,
    )


class FakeSummary:
    def invoke(self, messages):
        return AIMessage(content="讨论代号是松针-618，候选按程序结果处理。")


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
def test_policy_state_roundtrip_preserves_visible_context(monkeypatch, strategy):
    monkeypatch.setattr(memory_module, "create_llm", lambda: FakeSummary())
    memory = ConversationMemory(
        system_prompt="旧system", strategy=strategy,
        max_chars=1 if strategy == "summary" else 10000,
        window_size=4,
    )
    for index in range(8):
        memory.add_user(f"问题{index}")
        memory.add_ai(f"回答{index}")
    memory.update_system_message("当前system", plan_changed=True)
    snapshot = memory.export_state()
    restored = ConversationMemory.from_snapshot("重建system", snapshot)
    assert restored.strategy == strategy
    assert restored.max_chars == memory.max_chars
    assert restored.window_size == memory.window_size
    assert restored.summary == memory.summary
    assert restored.prompt_history == memory.prompt_history
    assert restored.messages[0].content == "重建system"
    assert [message.content for message in restored.get_messages_for_llm()[1:]] == [
        message.content for message in memory.get_messages_for_llm()[1:]
    ]
    if strategy in {"full", "window"}:
        assert len(snapshot.messages) == 16  # 不再截断成固定六轮。
    if strategy == "window":
        assert len(restored.get_messages_for_llm()) == 5  # system + 4条消息。
    if strategy == "summary":
        assert snapshot.summary
        assert len(snapshot.messages) == 2


def test_restore_does_not_call_summary(monkeypatch):
    memory = ConversationMemory(system_prompt="旧", strategy="summary", max_chars=1)
    memory.add_user("问题")
    memory.add_ai("回答")  # 不足两轮，不会调用摘要。
    snapshot = memory.export_state()
    monkeypatch.setattr(memory_module, "create_llm", lambda: pytest.fail("恢复不能调用模型"))
    restored = ConversationMemory.from_snapshot("新", snapshot)
    assert restored.strategy == "summary"


def test_export_refuses_half_turn_and_system_role():
    memory = ConversationMemory(system_prompt="规则")
    memory.add_user("尚未回答")
    with pytest.raises(ValidationError):
        memory.export_state()
    with pytest.raises(ValidationError):
        MemorySnapshot(
            strategy="full", max_chars=100, window_size=4,
            messages=[{"role": "system", "content": "越权规则"}],
        )


@pytest.mark.parametrize("version", ["learning-state-v1", "learning-state-v2", "learning-state-v3"])
def test_old_snapshot_has_no_invented_conversation(tmp_path, version):
    state = make_state().model_dump(mode="json")
    state.pop("conversation_memory", None)
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": version, "state": state}), encoding="utf-8")
    assert load_learning_state(path).conversation_memory is None


@pytest.fixture
def local_chat(monkeypatch, tmp_path):
    requests, saved = [], []
    monkeypatch.setattr(main_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(main_module, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main_module, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(memory_module.ConversationMemory, "_maybe_compress", lambda self: None)

    def save(state):
        path = tmp_path / f"turn-{len(saved)}.json"
        save_learning_state(state, path)
        saved.append(path)
        return path

    def turn(**kwargs):
        requests.append(kwargs)
        return "本次练习约定代号为松针-618。"

    monkeypatch.setattr(main_module, "save_current_learning_state", lambda state, _path: (save)(state))
    monkeypatch.setattr(main_module, "_invoke_interactive_turn", turn)
    return requests, saved


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
def test_saved_memory_reaches_next_session_without_strategy_argument(
    monkeypatch, tmp_path, local_chat, strategy,
):
    requests, saved = local_chat
    answers = iter(["本次练习代号是松针-618。", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))
    main_module.run_interactive_session(
        analysis="", plan="原计划", material_context="", learning_goal="原目标",
        strategy=strategy, material_dir=str(tmp_path), initial_state=make_state(),
    )
    restored = load_learning_state(saved[-1])
    assert restored.conversation_memory.strategy == strategy
    assert json.loads(saved[-1].read_text(encoding="utf-8"))["schema_version"] == "learning-state-v5"
    answers = iter(["刚才的代号是什么？", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))
    main_module.run_interactive_session(
        analysis="", plan=restored.plan_markdown, material_context="",
        learning_goal=restored.learning_goal, strategy=None,
        material_dir=str(tmp_path), initial_state=restored,
    )
    history = requests[-1]["request_messages"]
    assert any(message.type == "human" and "松针-618" in str(message.content) for message in history)
    assert load_learning_state(saved[-1]).conversation_memory.strategy == strategy


@pytest.mark.parametrize("answer", ["y", "n"])
def test_decision_is_persisted_not_unconfirmed_candidate(
    monkeypatch, tmp_path, local_chat, answer,
):
    _, saved = local_chat
    answers = iter(["请修改计划", answer, "exit"])

    def get_input(prompt):
        value = next(answers)
        if value == answer:
            assert saved == []
        return value

    monkeypatch.setattr(builtins, "input", get_input)
    monkeypatch.setattr(main_module, "_invoke_interactive_turn", lambda **kw: "===NEW_PLAN_START===新计划===NEW_PLAN_END===")
    main_module.run_interactive_session(
        analysis="", plan="原计划", material_context="", learning_goal="原目标",
        strategy="summary", material_dir=str(tmp_path), initial_state=make_state(),
    )
    assert len(saved) == 1
    restored = load_learning_state(saved[0])
    status = "confirmed" if answer == "y" else "cancelled"
    assert restored.plan_markdown == ("新计划" if answer == "y" else "原计划")
    assert f"程序处理结果：{status}" in restored.conversation_memory.messages[-1].content


def test_summary_failure_keeps_uncompressed_complete_turn(monkeypatch, tmp_path, local_chat):
    _, saved = local_chat
    answers = iter(["记住代号", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))

    def fail(self):
        raise RuntimeError("模拟摘要故障")

    monkeypatch.setattr(memory_module.ConversationMemory, "_maybe_compress", fail)
    with pytest.raises(RuntimeError, match="摘要故障"):
        main_module.run_interactive_session(
            analysis="", plan="原计划", material_context="", learning_goal="原目标",
            strategy="summary", material_dir=str(tmp_path), initial_state=make_state(),
        )
    memory = load_learning_state(saved[0]).conversation_memory
    assert [message.role for message in memory.messages] == ["human", "ai"]
    assert memory.messages[-1].content.endswith("松针-618。")


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
@pytest.mark.parametrize("branch", [
    "normal", "parse_invalid", "invalid", "unchanged", "cancelled", "confirmed",
])
def test_every_answer_branch_is_saved_before_next_question(
    monkeypatch, tmp_path, local_chat, strategy, branch,
):
    """只问一轮即退出，防止后续普通问答补保存而掩盖分支漏接。"""
    _, saved = local_chat
    if branch == "normal":
        response = "普通解释，不修改。"
    elif branch == "parse_invalid":
        response = "===NEW_PLAN_START===缺少结束标记"
    else:
        goal = "原目标" if branch == "unchanged" else (
            "" if branch == "invalid" else "新目标"
        )
        plan = "原计划" if branch == "unchanged" else "新计划"
        response = (
            f"===NEW_GOAL_START==={goal}===NEW_GOAL_END==="
            f"===NEW_PLAN_START==={plan}===NEW_PLAN_END==="
        )

    inputs = ["本轮验证"]
    if branch in {"cancelled", "confirmed"}:
        inputs.append("y" if branch == "confirmed" else "n")
    inputs.append("exit")
    answers = iter(inputs)

    def get_input(prompt):
        value = next(answers)
        if value in {"y", "n"}:
            assert saved == [], "确认之前不能采用候选或保存未定问答"
        return value

    monkeypatch.setattr(builtins, "input", get_input)
    monkeypatch.setattr(main_module, "_invoke_interactive_turn", lambda **kw: response)
    main_module.run_interactive_session(
        analysis="", plan="原计划", material_context="", learning_goal="原目标",
        strategy=strategy, material_dir=str(tmp_path), initial_state=make_state(),
    )

    assert len(saved) == 1
    restored = load_learning_state(saved[0])
    confirmed = branch == "confirmed"
    assert restored.learning_goal == ("新目标" if confirmed else "原目标")
    assert restored.plan_markdown == ("新计划" if confirmed else "原计划")
    assert restored.context_revision == (3 if confirmed else 2)
    assert [message.role for message in restored.conversation_memory.messages] == ["human", "ai"]
    if branch != "normal":
        expected = "invalid" if branch == "parse_invalid" else branch
        assert f"程序处理结果：{expected}" in restored.conversation_memory.messages[-1].content


@pytest.mark.parametrize("override", [None, "window"])
def test_cli_uses_saved_strategy_unless_explicit_override(
    monkeypatch, tmp_path, local_chat, override,
):
    requests, saved = local_chat
    root = tmp_path / "materials"
    root.mkdir()
    (root / "notes.md").write_text("资料", encoding="utf-8")
    memory = ConversationMemory(system_prompt="旧", strategy="full", window_size=2)
    memory.add_user("历史问题")
    memory.add_ai("历史回答")
    state = LearningState.model_validate({
        **make_state().model_dump(),
        "conversation_memory": memory.export_state().model_dump(),
    })
    state, _ = resolve_resume_materials(state, str(root))
    path = save_learning_state(state, tmp_path / "initial.json")
    args = ["agent", "--resume-state", str(path)]
    if override:
        args += ["--memory-strategy", override]
    monkeypatch.setattr(sys, "argv", args)
    answers = iter(["继续", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))
    main_module.main()
    assert load_learning_state(saved[-1]).conversation_memory.strategy == (override or "full")
