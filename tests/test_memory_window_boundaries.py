"""窗口按消息条数计量，但不能以失去提问的孤立回答开头。

固定输入：问题1/回答1、问题2/回答2；半轮场景再追加问题3。
只检查请求视图和恢复，不调用模型、不评价回答内容。
"""
import pytest

from ai_agent.persistence.memory import ConversationMemory


@pytest.mark.parametrize("size,pending,expected", [
    (1, False, []),
    (2, False, ["问题2", "回答2"]),
    (3, False, ["问题2", "回答2"]),
    (4, False, ["问题1", "回答1", "问题2", "回答2"]),
    (1, True, ["问题3"]),
    (2, True, ["问题3"]),
    (3, True, ["问题2", "回答2", "问题3"]),
    (4, True, ["问题2", "回答2", "问题3"]),
])
def test_window_does_not_start_with_orphan_answer(size, pending, expected):
    memory = ConversationMemory(system_prompt="当前规则", strategy="window", window_size=size)
    for index in (1, 2):
        memory.add_user(f"问题{index}")
        memory.add_ai(f"回答{index}")
    if pending:
        memory.add_user("问题3")
    before = [(m.type, m.content) for m in memory.messages]
    visible = memory.get_messages_for_llm()
    assert [m.content for m in visible] == ["当前规则", *expected]
    assert visible[0].type == "system"
    assert len(visible) - 1 <= size
    assert len(visible) == 1 or visible[1].type == "human"
    # 窗口只改变发送视图，不删除内部历史，也不修改消息对象。
    assert [(m.type, m.content) for m in memory.messages] == before
    assert all(any(m is original for original in memory.messages) for m in visible)


def test_odd_window_roundtrip_keeps_full_history_and_same_view():
    memory = ConversationMemory(system_prompt="旧规则", strategy="window", window_size=3)
    for index in (1, 2):
        memory.add_user(f"问题{index}")
        memory.add_ai(f"回答{index}")
    snapshot = memory.export_state()
    restored = ConversationMemory.from_snapshot("恢复后的当前规则", snapshot)
    assert restored.window_size == 3
    assert restored.export_state() == snapshot
    assert len(snapshot.messages) == 4
    assert [m.content for m in restored.get_messages_for_llm()] == [
        "恢复后的当前规则", "问题2", "回答2",
    ]
