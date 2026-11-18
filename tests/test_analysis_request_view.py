"""请求副本的值等价、隔离与分析循环接线；不请求模型。"""
from dataclasses import replace
from copy import deepcopy
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import pytest

@pytest.fixture
def view(tmp_path):
    # 已落地后直接验证正式模块，避免旧候选替身遮蔽生产代码问题。
    import ai_agent.runtime.agents as agents
    from ai_agent.runtime.request_context import prepare_model_messages
    return agents, prepare_model_messages


def test_copy_preserves_all_message_values_and_pairing(view):
    _, prepare = view
    history = [SystemMessage(content="职责"), HumanMessage(content="用户问题"),
        AIMessage(content="", tool_calls=[{"id": "read-1", "name": "read_material_lines", "args": {"relative_path": "notes.md"}, "type": "tool_call"}]),
        ToolMessage(content='{"ok": false}', tool_call_id="read-1")]
    actual = prepare(history)
    assert [m.model_dump() for m in actual] == [m.model_dump() for m in history]
    assert actual is not history
    assert all(a is not b for a, b in zip(actual, history))


def test_nested_args_and_metadata_do_not_modify_history(view):
    _, prepare = view
    history = [AIMessage(content="", additional_kwargs={"local": {"value": 1}},
        tool_calls=[{"id": "t", "name": "retrieve_learning_context", "args": {"queries": ["定义"]}, "type": "tool_call"}])]
    original = deepcopy(history[0].model_dump())
    actual = prepare(history)
    actual[0].tool_calls[0]["args"]["queries"].append("另一问题")
    actual[0].additional_kwargs["local"]["value"] = 2
    assert history[0].model_dump() == original


def test_snapshot_does_not_accumulate_later_messages(view):
    _, prepare = view
    history = [HumanMessage(content="问题")]
    first = prepare(history)
    history.append(AIMessage(content="答案"))
    second = prepare(history)
    assert len(first) == 1 and len(second) == 2
    assert prepare([]) == []


def test_shadow_loader_accepts_adopted_multiline_call(tmp_path):
    from tests.helpers.analysis_request_view import candidate
    with candidate(tmp_path / "replacement") as (agents, prepare):
        assert callable(agents.analyse_materials_agentic)
        assert prepare([]) == []
        assert prepare([], tool_result_formatter=lambda text: text) == []


def test_analysis_requests_are_value_equivalent_to_working_history(view, monkeypatch, tmp_path):
    agents, prepare = view
    (tmp_path / "notes.md").write_text("正常结束不证明回答正确。", encoding="utf-8")
    sequence = [
        ("list_learning_files", {}),
        ("search_learning_files", {"keyword": "正常结束"}),
        ("read_material_lines", {"relative_path": "notes.md", "start_line": 1, "end_line": 1}),
    ]
    responses = iter([AIMessage(content="", tool_calls=[{"id": f"call-{i}", "name": name, "args": args, "type": "tool_call"}])
        for i, (name, args) in enumerate(sequence, 1)] + [AIMessage(content="正常结束不证明回答正确。[source=notes.md, lines=1-1]")])
    expected, requests = [], []

    def observe_history(history, **kwargs):
        result = prepare(history, **kwargs)
        expected.append(deepcopy([m.model_dump() for m in result]))
        return result

    def invoke(llm, messages, **kwargs):
        requests.append([m.model_dump() for m in messages])
        return next(responses)

    monkeypatch.setattr(agents, "prepare_model_messages", observe_history)
    monkeypatch.setattr(agents, "create_llm_with_tools", lambda _: object())
    monkeypatch.setattr(agents, "_invoke_llm_raw", invoke)
    monkeypatch.setattr(agents, "settings", replace(agents.settings, agent_max_tool_calls=12))
    monkeypatch.setattr(agents, "MAX_TOOL_ROUNDS", 8)
    answer, context = agents.analyse_materials_agentic(str(tmp_path), "请解释notes.md中的定义，依据原文，不修改记录。")
    assert requests == expected and len(requests) == 4
    assert [len([m for m in request if m["type"] == "tool"]) for request in requests] == [0, 1, 2, 3]
    assert [m["tool_call_id"] for m in requests[-1] if m["type"] == "tool"] == ["call-1", "call-2", "call-3"]
    assert "正常结束不证明回答正确" in answer
    assert json.loads(context)["data"]["lines"][0]["text"] == "正常结束不证明回答正确。"
