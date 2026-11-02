"""滚动摘要接线：固定替身检查输入、替换与失败保护，不评价真实模型语义。"""
from types import SimpleNamespace

import pytest
import ai_agent.persistence.memory as memory_module
from ai_agent.persistence.memory import ConversationMemory


OLD = "用户只阅读，不编程；可用时间20分钟。"
NEW = "用户只阅读，不编程；可用时间更正为40分钟。"
TURNS = [
    ("今天最多20分钟，只阅读，不编程。", "已确认安排反馈。"),
    ("把时间更正为40分钟，其余限制不变。", "已确认更正，保留其余条件。"),
    ("只讨论下一步。", "先比较再口述。"),
]


def install_summary(monkeypatch, outputs):
    requests = []
    remaining = iter(outputs)

    def invoke(messages):
        requests.append(messages)
        return SimpleNamespace(content=next(remaining))

    monkeypatch.setattr(memory_module, "create_llm", lambda: SimpleNamespace(invoke=invoke))
    return requests


def add_turn(memory, index):
    question, answer = TURNS[index]
    memory.add_user(question)
    memory.add_ai(answer)


def test_second_compression_receives_old_summary_and_replaces_it(monkeypatch):
    requests = install_summary(monkeypatch, [OLD, NEW])
    memory = ConversationMemory(system_prompt="当前规则", strategy="summary", max_chars=1)
    add_turn(memory, 0)
    add_turn(memory, 1)
    assert memory.summary == OLD
    add_turn(memory, 2)
    assert len(requests) == 2
    second_input = "\n".join(str(m.content) for m in requests[1][1:])
    assert OLD in second_input
    assert TURNS[1][0] in second_input
    assert TURNS[2][0] not in second_input  # 最近一问一答不参与这次压缩。
    assert memory.summary == NEW
    assert [m.content for m in memory.messages] == ["当前规则", *TURNS[2]]


def test_restored_summary_participates_in_next_compression(monkeypatch):
    requests = install_summary(monkeypatch, [OLD, NEW])
    memory = ConversationMemory(system_prompt="旧规则", strategy="summary", max_chars=1)
    add_turn(memory, 0)
    add_turn(memory, 1)
    snapshot = memory.export_state()
    restored = ConversationMemory.from_snapshot("当前规则", snapshot)
    assert len(requests) == 1, "恢复本身不能调用摘要模型"
    add_turn(restored, 2)
    assert OLD in "\n".join(str(m.content) for m in requests[1][1:])
    assert restored.summary == NEW
    assert restored.export_state().summary == NEW
    assert memory.export_state() == snapshot


@pytest.mark.parametrize("invalid", [None, [], "", " \n "])
def test_invalid_summary_does_not_replace_summary_or_drop_messages(monkeypatch, invalid):
    install_summary(monkeypatch, [invalid])
    memory = ConversationMemory(system_prompt="当前规则", strategy="summary", max_chars=1)
    memory.summary = OLD
    # 用公开add_user/add_ai准备历史，但准备阶段不触发压缩。
    memory.strategy = "full"
    add_turn(memory, 0)
    add_turn(memory, 1)
    memory.strategy = "summary"
    before = memory.export_state()
    with pytest.raises(ValueError, match="摘要"):
        memory._maybe_compress()
    assert memory.export_state() == before


def test_model_exception_preserves_original_summary_and_messages(monkeypatch):
    def fail(messages):
        raise RuntimeError("测试摘要请求失败")

    monkeypatch.setattr(memory_module, "create_llm", lambda: SimpleNamespace(invoke=fail))
    memory = ConversationMemory(system_prompt="当前规则", strategy="full", max_chars=1)
    add_turn(memory, 0)
    add_turn(memory, 1)
    memory.strategy = "summary"
    memory.summary = OLD
    before = memory.export_state()
    # 私有压缩入口将外部调用故障归类；add_ai再处理此类可恢复故障。
    with pytest.raises(memory_module.SummaryCompressionError, match="摘要请求失败"):
        memory._maybe_compress()
    assert memory.export_state() == before
