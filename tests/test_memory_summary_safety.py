"""摘要采用预算和可恢复故障：固定输出，不调用在线模型。"""
import builtins
from types import SimpleNamespace

import pytest

import ai_agent.config as config
import ai_agent.persistence.memory as module
from ai_agent.persistence.memory import ConversationMemory


@pytest.fixture
def policy(monkeypatch):
    settings = SimpleNamespace(memory_summary_max_chars=10, log_payloads=False,
                               memory_summary_trigger_chars=1, memory_window_size=6)
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    return settings


def prepared():
    memory = ConversationMemory(system_prompt="当前规则", strategy="full", max_chars=1)
    memory.summary = "原有摘要"
    memory.add_user("问题1")
    memory.add_ai("回答1")
    memory.add_user("问题2")
    memory.strategy = "summary"
    return memory


@pytest.mark.parametrize("content", [None, [], "", " \n ", "字" * 11])
def test_invalid_summary_keeps_complete_turn_and_old_summary(monkeypatch, policy, caplog, content):
    monkeypatch.setattr(module, "create_llm", lambda: SimpleNamespace(
        invoke=lambda _: SimpleNamespace(content=content)))
    memory = prepared()
    memory.add_ai("回答2")
    assert memory.summary == "原有摘要"
    assert [m.content for m in memory.messages] == ["当前规则", "问题1", "回答1", "问题2", "回答2"]
    assert "摘要压缩未采用" in caplog.text


def test_exact_budget_is_accepted_and_sent_in_request(monkeypatch, policy):
    calls = []

    def invoke(messages):
        calls.append(messages)
        return SimpleNamespace(content="字" * 10)

    monkeypatch.setattr(module, "create_llm", lambda: SimpleNamespace(invoke=invoke))
    memory = prepared()
    memory.add_ai("回答2")
    assert memory.summary == "字" * 10
    assert [m.content for m in memory.messages] == ["当前规则", "问题2", "回答2"]
    assert "10" in calls[0][0].content


@pytest.mark.parametrize("stage", ["create", "invoke"])
def test_provider_failures_are_recoverable_without_exposing_message(monkeypatch, policy, caplog, stage):
    def fail(*args):
        raise TimeoutError("DO_NOT_LOG_PRIVATE_ENDPOINT")

    monkeypatch.setattr(module, "create_llm", fail if stage == "create" else
                        lambda: SimpleNamespace(invoke=fail))
    memory = prepared()
    memory.add_ai("回答2")
    assert memory.summary == "原有摘要"
    assert len(memory.export_state().messages) == 4
    assert "TimeoutError" in caplog.text
    assert "DO_NOT_LOG_PRIVATE_ENDPOINT" not in caplog.text


def test_next_turn_can_compress_after_failure(monkeypatch, policy):
    outputs = iter(["字" * 11, "更新后的摘要"])
    monkeypatch.setattr(module, "create_llm", lambda: SimpleNamespace(
        invoke=lambda _: SimpleNamespace(content=next(outputs))))
    memory = prepared()
    memory.add_ai("回答2")
    memory.add_user("问题3")
    memory.add_ai("回答3")
    assert memory.summary == "更新后的摘要"
    assert [m.content for m in memory.messages] == ["当前规则", "问题3", "回答3"]


def test_unclassified_programming_error_is_not_swallowed(monkeypatch, policy):
    def bug(self):
        raise AssertionError("程序错误不能冒充摘要降级")

    monkeypatch.setattr(ConversationMemory, "_maybe_compress", bug)
    with pytest.raises(AssertionError):
        prepared().add_ai("回答2")


def test_global_budget_applies_after_restore_without_new_snapshot_field(monkeypatch, policy):
    memory = prepared()
    memory.strategy = "full"
    memory.add_ai("回答2")
    memory.strategy = "summary"
    snapshot = memory.export_state()
    restored = ConversationMemory.from_snapshot("恢复规则", snapshot)
    assert "summary_max_chars" not in snapshot.model_dump()
    policy.memory_summary_max_chars = 3
    monkeypatch.setattr(module, "create_llm", lambda: SimpleNamespace(
        invoke=lambda _: SimpleNamespace(content="四个字符")))
    restored.add_user("问题3")
    restored.add_ai("回答3")
    assert restored.summary == snapshot.summary
    assert len(restored.export_state().messages) == 6


@pytest.mark.parametrize("value,expected", [(None, 1000), ("700", 700)])
def test_summary_budget_config(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MEMORY_SUMMARY_MAX_CHARS", raising=False)
    else:
        monkeypatch.setenv("MEMORY_SUMMARY_MAX_CHARS", value)
    config.get_settings.cache_clear()
    try:
        assert config.get_settings().memory_summary_max_chars == expected
    finally:
        config.get_settings.cache_clear()


@pytest.mark.parametrize("value,error", [
    ("0", RuntimeError), ("-1", RuntimeError), ("invalid", ValueError),
])
def test_invalid_summary_budget_is_rejected(monkeypatch, value, error):
    monkeypatch.setenv("MEMORY_SUMMARY_MAX_CHARS", value)
    config.get_settings.cache_clear()
    try:
        with pytest.raises(error):
            config.get_settings()
    finally:
        config.get_settings.cache_clear()


def test_real_session_continues_and_saves_after_summary_timeout(monkeypatch, tmp_path, policy):
    import ai_agent.main as app
    from ai_agent.learning.learning_state import LearningState
    from ai_agent.persistence.state_store import load_learning_state

    answers = iter(["问题1", "问题2", "问题3", "exit"])
    calls = []

    def invoke(**kwargs):
        calls.append(kwargs["user_input"])
        return "完整回答"

    def timeout(messages):
        raise TimeoutError("模拟摘要超时")

    monkeypatch.setattr(builtins, "input", lambda _: next(answers))
    monkeypatch.setattr(app, "create_llm", lambda **kw: object())
    monkeypatch.setattr(app, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(app, "needs_rag", lambda _: False)
    monkeypatch.setattr(app, "_invoke_interactive_turn", invoke)
    monkeypatch.setattr(module, "create_llm", lambda: SimpleNamespace(invoke=timeout))
    path = tmp_path / "current.json"
    initial = LearningState(learning_goal="原目标", plan_markdown="原计划")
    app.run_interactive_session("", "原计划", "", "原目标", "summary", str(tmp_path),
                                initial_state=initial, current_state_path=path)
    saved = load_learning_state(path)
    assert calls == ["问题1", "问题2", "问题3"]
    assert len(saved.conversation_memory.messages) == 6
    assert saved.model_dump(exclude={"conversation_memory"}) == initial.model_dump(
        exclude={"conversation_memory"})
