from __future__ import annotations

from langchain_core.messages import AIMessage

import ai_agent.memory as memory_module
from ai_agent.memory import ConversationMemory


def contents(messages):
    return [str(message.content) for message in messages]


def test_full_memory_keeps_all_messages():
    memory = ConversationMemory(
        system_prompt="system",
        strategy="full",
    )
    memory.add_user("问题1")
    memory.add_ai("回答1")
    memory.add_user("问题2")
    memory.add_ai("回答2")

    assert contents(memory.get_messages_for_llm()) == [
        "system",
        "问题1",
        "回答1",
        "问题2",
        "回答2",
    ]


def test_window_memory_keeps_only_recent_messages():
    memory = ConversationMemory(
        system_prompt="system",
        strategy="window",
        window_size=2,
    )
    memory.add_user("问题1")
    memory.add_ai("回答1")
    memory.add_user("问题2")
    memory.add_ai("回答2")

    assert contents(memory.get_messages_for_llm()) == [
        "system",
        "问题2",
        "回答2",
    ]


def test_summary_compresses_old_turns_but_keeps_latest_turn(
    monkeypatch,
):
    class FakeSummaryLLM:
        def invoke(self, messages):
            return AIMessage(content="用户比较过两种学习方案。")

    monkeypatch.setattr(
        memory_module,
        "create_llm",
        lambda: FakeSummaryLLM(),
    )

    memory = ConversationMemory(
        system_prompt="当前目标与计划",
        strategy="summary",
        max_chars=1,
    )
    memory.add_user("问题1")
    memory.add_ai("回答1")
    memory.add_user("问题2")
    memory.add_ai("回答2")

    result = contents(memory.get_messages_for_llm())

    assert result[0] == "当前目标与计划"
    assert "用户比较过两种学习方案" in result[1]
    assert result[-2:] == ["问题2", "回答2"]


def test_system_update_preserves_old_version_and_marks_summary_stale():
    memory = ConversationMemory(
        system_prompt="旧目标与旧计划",
        strategy="summary",
    )
    memory.summary = "用户讨论过旧计划的安排。"

    memory.update_system_message(
        "新目标与新计划",
        plan_changed=True,
    )

    assert memory.prompt_history == ["旧目标与旧计划"]
    assert memory.system_prompt == "新目标与新计划"
    assert "可能已过时" in memory.summary