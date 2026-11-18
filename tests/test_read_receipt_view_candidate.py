"""候选实验：验证表示整理，不评价真实模型回答质量。"""
from copy import deepcopy
from dataclasses import replace
import json
import types

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest

from tests.helpers.read_receipt_view import SOURCE
import ai_agent.runtime.agents as agents


@pytest.fixture(params=["production", "candidate"])
def prepare(request):
    if request.param == "production":
        from ai_agent.runtime.request_context import prepare_model_messages
        return prepare_model_messages
    module = types.ModuleType("read_receipt_candidate")
    exec(compile(SOURCE, "read_receipt_candidate", "exec"), module.__dict__)
    return module.prepare_model_messages


def receipt():
    return json.dumps({"ok": True, "data": {
        "source": "notes.md", "citation": "[source=notes.md, lines=1-1]",
        "requested_start_line": 1, "requested_end_line": 200,
        "returned_start_line": 1, "returned_end_line": 1, "truncated": False,
        "lines": [{"line": 1, "text": "正常结束不证明回答正确。", "partial": False}],
    }}, ensure_ascii=False)


def history(content, name="read_material_lines"):
    return [HumanMessage(content="依据原文回答"),
        AIMessage(content="", tool_calls=[{"id": "r1", "name": name,
            "args": {"relative_path": "notes.md", "start_line": 1, "end_line": 200},
            "type": "tool_call"}]),
        ToolMessage(content=content, tool_call_id="r1")]


def test_compacts_only_view_and_preserves_pairing(prepare):
    raw = history(receipt())
    before = deepcopy([m.model_dump() for m in raw])
    result = prepare(raw, tool_result_formatter=agents.stage_evidence)
    assert [m.model_dump() for m in raw] == before
    assert result[1].model_dump() == before[1]
    assert result[2].tool_call_id == "r1"
    assert result[2].content == (
        "[source=notes.md, lines=1-1]\n实际行范围：1-1；截断=False\n"
        "1: 正常结束不证明回答正确。")
    assert len(result[2].content) < len(raw[2].content)
    assert [m.model_dump() for m in prepare(raw)] == before


@pytest.mark.parametrize("content", [
    '{"ok": false, "error": "read failed"}',
    "not json", '{"ok": true, "data": {"new_schema": 1}}',
])
def test_failure_and_unknown_are_not_rewritten(prepare, content):
    raw = history(content)
    assert prepare(raw, tool_result_formatter=agents.stage_evidence)[-1].content == content


def test_other_tools_and_unpaired_results_are_preserved(prepare):
    raw = history(receipt(), name="search_learning_files")
    assert prepare(raw, tool_result_formatter=agents.stage_evidence)[-1].content == receipt()
    raw = [ToolMessage(content=receipt(), tool_call_id="unknown")]
    assert prepare(raw, tool_result_formatter=agents.stage_evidence)[0].content == receipt()


def test_partial_and_truncated_are_visible(prepare):
    payload = json.loads(receipt())
    payload["data"]["truncated"] = True
    payload["data"]["lines"][0]["partial"] = True
    raw = json.dumps(payload)
    result = prepare(history(raw), tool_result_formatter=agents.stage_evidence)
    # 新策略：完整性不足时保留原始JSON，不再要求转换后的标记。
    assert result[-1].content == raw
    assert json.loads(result[-1].content)["data"]["truncated"] is True
    assert json.loads(result[-1].content)["data"]["lines"][0]["partial"] is True


def test_real_tool_loop_candidate(prepare, monkeypatch, tmp_path):
    (tmp_path / "notes.md").write_text("正常结束不证明回答正确。", encoding="utf-8")
    sequence = [("list_learning_files", {}),
        ("search_learning_files", {"keyword": "正常结束"}),
        ("read_material_lines", {"relative_path": "notes.md", "start_line": 1, "end_line": 1})]
    responses = iter([AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": name,
        "args": args, "type": "tool_call"}]) for i, (name, args) in enumerate(sequence)]
        + [AIMessage(content="正常结束不证明回答正确。[source=notes.md, lines=1-1]")])
    raw, requests = [], []

    def build(messages, **kwargs):
        raw.append(deepcopy(messages))
        return prepare(messages, tool_result_formatter=agents.stage_evidence)

    def invoke(llm, messages, **kwargs):
        requests.append(deepcopy(messages))
        return next(responses)

    monkeypatch.setattr(agents, "prepare_model_messages", build)
    monkeypatch.setattr(agents, "_invoke_llm_raw", invoke)
    monkeypatch.setattr(agents, "create_llm_with_tools", lambda _: object())
    monkeypatch.setattr(agents, "settings", replace(agents.settings, agent_max_tool_calls=12))
    monkeypatch.setattr(agents, "MAX_TOOL_ROUNDS", 8)
    _, context = agents.analyse_materials_agentic(str(tmp_path),
        "请解释notes.md中的定义，依据原文，不修改记录。")
    assert len(requests) == 4
    assert [m.model_dump() for m in raw[-1][:-1]] == [m.model_dump() for m in requests[-1][:-1]]
    assert json.loads(raw[-1][-1].content)["ok"] is True
    assert requests[-1][-1].content.startswith("[source=notes.md, lines=1-1]")
    assert [m.tool_call_id for m in requests[-1] if isinstance(m, ToolMessage)] == ["c0", "c1", "c2"]
    assert json.loads(context)["data"]["lines"][0]["text"] == "正常结束不证明回答正确。"
