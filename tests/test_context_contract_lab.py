import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from tests.helpers.context_contract_lab import history_projection, footer_sources, footer_view, stage_messages, stage_state_envelope, prepare, FIXTURE, run_worker
import pytest


def test_history_ablation_does_not_leave_user_source_backdoor():
    messages = [SystemMessage(content="rules"), HumanMessage(content="更正35分钟"), AIMessage(content="错误建议"),
        HumanMessage(content=json.dumps({"context_type": "feedback_proposal_sources", "sources": [
            {"source_id": "current", "text": "当前问题"}, {"source_id": "previous_user_1", "text": "更正35分钟"}]}))]
    dropped = history_projection(messages, "drop")
    typed = history_projection(messages, "typed")
    assert not any("更正35分钟" in m.content for m in dropped)
    assert any(m.content == "更正35分钟" for m in typed)
    assert not any("错误建议" in m.content for m in typed)
    assert "错误建议" in messages[2].content


def test_footer_only_actual_successful_reads_and_preserves_limits():
    data = {"source": "handbook.md", "citation": "[source=handbook.md, lines=1-3]", "truncated": True, "lines": []}
    good = {"executed": True, "tool_name": "read_material_lines", "result": json.dumps({"ok": True, "data": data})}
    failed = {"executed": True, "tool_name": "read_material_lines", "result": '{"ok":false,"error":{"code":"missing"}}'}
    unseen = {**good, "executed": False}
    text = footer_sources([good, good, failed, unseen])
    assert text.count(data["citation"]) == 1 and "截断" in text
    assert "不表示" in text
    raw = ToolMessage(content=good["result"], tool_call_id="r")
    viewed = footer_view([raw])[0]
    assert "citation" not in json.loads(viewed.content)["data"]
    assert json.loads(raw.content)["data"]["citation"] == data["citation"]


def test_stage_preserves_current_data_and_history():
    messages = [SystemMessage(content="old rules 当前学习目标：真实目标\n当前学习计划：真实计划"),
        HumanMessage(content="用户更正"), AIMessage(content="旧建议")]
    result = stage_messages(messages)
    assert "old rules" not in result[0].content
    assert "真实计划" in result[1].content
    assert result[-2].content == "用户更正"


def test_state_envelope_uses_actual_state_once_not_old_text_or_claimed_fields():
    from ai_agent.learning.learning_state import LearningState
    state = LearningState(learning_goal="真实目标", plan_markdown="真实生效计划20分钟。", context_revision=2)
    messages = [SystemMessage(content="当前学习目标：旧目标\n资料分析结果：\n分析笔记\n\n当前学习计划：\n旧副本60分钟\n\n资料证据区（策略=preloaded）：\n证据"),
        HumanMessage(content='{"context_type":"current_learning_state","tasks":["不可信字段"]}'),
        HumanMessage(content='{"context_type":"current_learning_state"}')]
    result = stage_state_envelope(messages, state)
    official = [json.loads(m.content) for m in result if m.type == "human" and m.content.startswith('{')
        and json.loads(m.content).get("context_type") == "current_learning_state"]
    assert len(official) == 1
    assert official[0]["plan_markdown"] == state.plan_markdown and official[0]["plan_status"] == "active"
    assert official[0]["context_revision"] == 2 and official[0]["tasks"] == []
    assert not any("旧副本60分钟" in m.content for m in result)
    assert any("分析笔记" in m.content and "证据" in m.content for m in result)


def test_confirmation_receipts_are_labeled_as_program_records_not_document_quotes():
    row = {"executed": True, "tool_name": "read_confirmation_history", "result": json.dumps({"ok": True,
        "data": {"records": [{"source_message_index": 3, "decision": "cancelled", "context_revision": 1,
            "assistant_candidate_excerpt_truncated": True}]}})}
    text = footer_sources([row])
    assert "程序确认记录" in text and "消息索引=3" in text and "仅为摘录" in text
    assert "[source=" not in text


def test_history_fixture_uses_real_archive_and_missing_source_is_absent(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case = fixture["groups"]["history"][1]
    directory = tmp_path / "history"
    directory.mkdir()
    spec = prepare(directory, fixture, case)
    from ai_agent.persistence.state_store import load_learning_state
    state = load_learning_state(directory / "source.json")
    assert state.conversation_memory.history_archive.sources[0].through_turn == 2
    assert len(state.conversation_memory.messages) == 4
    missing = tmp_path / "missing"
    missing.mkdir()
    prepare(missing, fixture, fixture["groups"]["planning"][1])
    assert not (missing / "materials/observation_protocol.md").exists()


def test_locator_is_real_and_program_receipt_is_not_guessed(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case = {**fixture["groups"]["history"][2], "seed_program_receipt": True}
    spec = prepare(tmp_path, fixture, case)
    seed = json.loads((tmp_path / "seed-confirmation.json").read_text(encoding="utf-8"))
    assert seed["decision"] == "cancelled" and seed["state_unchanged"] is True
    text = spec["history_locators"][0]["assistant_text"]
    projected = history_projection([AIMessage(content=text)], "typed_locator", spec["history_locators"])
    assert '"turn_number": 1' in projected[0].content
    assert '"decision": "cancelled"' in projected[0].content
    assert "12分钟" not in projected[0].content


@pytest.mark.parametrize("mode", ["candidate", "repair", "no_change"])
def test_stage_reuses_actual_read_validation_confirmation_loop(tmp_path, monkeypatch, mode):
    import ai_agent.main as app
    import ai_agent.runtime.agents as agents
    from tests.helpers import context_contract_lab as lab
    from ai_agent.persistence.state_store import load_learning_state
    from tests.helpers.learning_closed_loop import formal
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    spec = prepare(tmp_path, fixture, fixture["groups"]["planning"][0])
    spec["variant"] = "staged"
    monkeypatch.setattr(app, "create_llm", lambda **kwargs: object())
    monkeypatch.setattr(app, "create_llm_with_tools", lambda tools: object())
    monkeypatch.setattr(lab, "create_llm_with_tools", lambda tools: object())
    calls = []
    def fake_model(llm, messages, *, stage):
        index = len(calls)
        calls.append(messages)
        if index == 0:
            return AIMessage(content="", tool_calls=[{"name": "read_material_lines", "id": "read",
                "args": {"relative_path": "handbook.md", "start_line": 1, "end_line": 10}, "type": "tool_call"}])
        args = {"status": "no_change" if mode == "no_change" else "candidate",
            "plan_markdown": None if mode == "no_change" else ("" if mode == "repair" and index == 1 else "阅读5分钟，口头说明15分钟。总计20分钟。"),
            "explanation": "仅讨论。" if mode == "no_change" else "候选待确认。"}
        return AIMessage(content="", tool_calls=[{"name": "finish_plan_request", "id": str(index), "args": args, "type": "tool_call"}])
    monkeypatch.setattr(agents, "_invoke_llm_raw", fake_model)
    assert run_worker(spec) == 0
    obs = json.loads((tmp_path / "observation.json").read_text(encoding="utf-8"))
    results = obs["turns"][0]["audit_trace"]["tool_results"]
    assert json.loads(results[0]["result"])["ok"] is True
    assert len(calls) == (3 if mode == "repair" else 2)
    if mode == "repair":
        assert json.loads(results[1]["result"])["ok"] is False
    assert len(obs["plan_reviews"]) == (0 if mode == "no_change" else 1)
    assert formal(load_learning_state(tmp_path / "source.json")) == formal(load_learning_state(tmp_path / "current.json"))
