"""在线验收辅助脚本的离线测试；所有模型均为替身，禁止真实请求。"""
import importlib.util
import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
import pytest

import ai_agent.main as app
from ai_agent.runtime.agents import LLMInvocationError


@pytest.fixture
def helper():
    path = Path(__file__).parent / "helpers/history_readback_online.py"
    spec = importlib.util.spec_from_file_location("history_readback_online_test_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_case(helper, suffix="001"):
    return helper.load_case(helper.DEFAULT_CASES, f"history-readback-online-{suffix}")


def tool_call(name, args, ident):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": ident, "type": "tool_call"}])


def prepare_models(monkeypatch, suffix):
    monkeypatch.setattr(app, "create_llm", lambda **kw: object())
    monkeypatch.setattr(app, "create_llm_with_tools", lambda *a, **kw: object())
    def response(llm, messages, *, stage):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if suffix == "002" and not results:
            return tool_call("list_conversation_sources", {}, "list")
        if not results or (suffix == "002" and len(results) == 1):
            return tool_call("read_conversation_turn", {"source_index": 1, "turn_number": 2}, "read")
        result = json.loads(results[-1].content)
        if result["ok"]:
            return AIMessage(content=f"> {result['data']['user_text']}\n{result['data']['citation']}")
        return AIMessage(content="读取被快照上限拒绝，无法引用该轮原话。")
    monkeypatch.setattr(app, "invoke_llm_response", response)


def test_preflight_is_readable_and_never_calls_model_or_creates_output(helper, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app, "create_llm", lambda **kw: pytest.fail("预检不能创建模型"))
    output = tmp_path / "results.jsonl"
    assert helper.main(["--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "preflight_only" and not output.exists()
    assert payload["case"]["question"] == payload["automatic_input"][0]
    assert payload["fixture_history"][1]["user"].startswith("更正：今天最多35分钟")
    assert payload["initial_context"]["messages"] == []
    assert "api_key" not in json.dumps(payload)


@pytest.mark.parametrize("suffix", ["001", "002", "003"])
def test_case_runs_real_tools_with_stub_model_and_saves_full_requests(helper, tmp_path, monkeypatch, suffix):
    prepare_models(monkeypatch, suffix)
    fixture, case = fixture_case(helper, suffix)
    output = tmp_path / "results.jsonl"
    row = helper.run_case(fixture, case, output, helper.manifest(helper.DEFAULT_CASES))
    assert row["status"] == "completed" and row["error_type"] is None
    assert row["business_state_unchanged"] and row["source_unchanged"] and row["inherited_scope_unchanged"]
    assert row["assessment"] == "not_evaluated"
    facts = row["observations"]
    assert facts["first_request_contains_target_quote"] is False
    assert facts["unexpected_confirmation_count"] == 0
    obs = json.loads(Path(row["observation_path"]).read_text(encoding="utf-8"))
    assert obs["user_input"] == case["question"]
    assert obs["input_events"][-1]["value"] == "exit"
    tools = obs["audit_trace"]["tool_results"]
    assert tools[-1]["tool_name"] == "read_conversation_turn" and tools[-1]["executed"]
    assert "tool" in [m["role"] for m in obs["model_requests"][-1]["request_messages"]]
    if suffix == "003":
        assert facts["scope_denial_count"] == 1 and facts["successful_read_count"] == 0
        assert facts["requested_read_succeeded"] is False
        assert facts["exact_target_quote_in_answer"] is False
        assert "未取得可确认的成功结果" in row["final_answer"]
    else:
        assert facts["successful_read_count"] == 1 and facts["scope_denial_count"] == 0
        assert facts["requested_read_succeeded"] is True
        assert facts["exact_target_quote_in_answer"] and facts["returned_citation_in_answer"]
        assert fixture["history"][1]["user"] in row["final_answer"]
    if suffix == "002":
        assert facts["successful_source_list_count"] == 1
        assert [t["tool_name"] for t in tools] == ["list_conversation_sources", "read_conversation_turn"]


def test_raw_model_citation_observation_does_not_use_program_appended_sources(helper, tmp_path, monkeypatch):
    prepare_models(monkeypatch, "001")
    original = app.invoke_llm_response

    def corrupt_citation(llm, messages, *, stage):
        response = original(llm, messages, stage=stage)
        if response.tool_calls:
            return response
        return AIMessage(content="找到原话。[history=wrong/session, turn=2]")

    monkeypatch.setattr(app, "invoke_llm_response", corrupt_citation)
    fixture, case = fixture_case(helper, "001")
    row = helper.run_case(fixture, case, tmp_path / "result.jsonl", {})
    assert row["observations"]["model_returned_citation_in_answer"] is False
    # 此处不假定候选已应用；该字段只观察最终正文是否出现程序核验提示。
    assert isinstance(row["observations"]["contains_history_citation_notice"], bool)


def test_unexpected_plan_confirmation_is_declined_and_not_hidden(helper, tmp_path, monkeypatch):
    prepare_models(monkeypatch, "001")
    monkeypatch.setattr(app, "invoke_llm_response", lambda *a, **kw: AIMessage(content=(
        "===NEW_PLAN_START===\n擅自提出的新计划\n===NEW_PLAN_END===")))
    fixture, case = fixture_case(helper, "002")
    row = helper.run_case(fixture, case, tmp_path / "result.jsonl", {})
    assert row["status"] == "completed" and row["business_state_unchanged"]
    assert row["observations"]["unexpected_confirmation_count"] > 0
    assert row["assessment"] == "not_evaluated"
    obs = json.loads(Path(row["observation_path"]).read_text(encoding="utf-8"))
    assert any(e["kind"] == "decline_unexpected_confirmation" and e["value"] == "n" for e in obs["input_events"])
    assert "cancelled" in row["final_answer"]


def test_short_reference_observation_is_separate_from_long_citation(helper, tmp_path, monkeypatch):
    prepare_models(monkeypatch, "001")
    original = app.invoke_llm_response

    def use_reference(llm, messages, *, stage):
        response = original(llm, messages, stage=stage)
        if response.tool_calls:
            return response
        return AIMessage(content="已回读。[history-ref=1:2]")

    monkeypatch.setattr(app, "invoke_llm_response", use_reference)
    fixture, case = fixture_case(helper, "001")
    row = helper.run_case(fixture, case, tmp_path / "short-ref.jsonl", {})
    facts = row["observations"]
    assert facts["model_returned_reference_in_answer"] is True
    assert facts["model_returned_citation_in_answer"] is False
    assert facts["returned_citation_in_answer"] is True
    assert facts["contains_history_citation_notice"] is False


def test_model_failure_records_unknown_state_and_preserves_trace(helper, tmp_path, monkeypatch):
    prepare_models(monkeypatch, "001")
    def fail(*a, **kw):
        raise LLMInvocationError("PRIVATE_MODEL_DETAIL")
    monkeypatch.setattr(app, "invoke_llm_response", fail)
    fixture, case = fixture_case(helper)
    row = helper.run_case(fixture, case, tmp_path / "result.jsonl", {})
    assert row["status"] == "failed" and row["error_type"] == "LLMInvocationError"
    assert row["business_state_unchanged"] is None and row["final_answer"] is None
    assert row["source_unchanged"] and "PRIVATE_MODEL_DETAIL" not in json.dumps(row)
    obs = json.loads(Path(row["observation_path"]).read_text(encoding="utf-8"))
    assert len(obs["model_requests"]) == 1


def test_each_run_has_independent_artifacts_and_appends_result(helper, tmp_path, monkeypatch):
    prepare_models(monkeypatch, "001")
    fixture, case = fixture_case(helper)
    output = tmp_path / "result.jsonl"
    a = helper.run_case(fixture, case, output, {})
    b = helper.run_case(fixture, case, output, {})
    assert a["run_id"] != b["run_id"] and a["artifacts"] != b["artifacts"]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    assert a["observations"] == b["observations"]


def test_unrecognized_case_rejected(helper):
    with pytest.raises(ValueError, match="未找到"):
        helper.load_case(helper.DEFAULT_CASES, "unknown")


def test_usage_summary_distinguishes_execution_failure_and_budget_denial(helper):
    def row(executed, payload):
        return {"tool_name": "read_conversation_turn", "executed": executed,
                "result": json.dumps(payload)}
    facts = helper.summarize_tool_usage([
        row(True, {"ok": True, "data": {"source_index": 2, "turn_number": 1}}),
        row(True, {"ok": False, "error": {"code": "turn_out_of_scope"}}),
        row(False, {"ok": False, "error": {"code": "budget_exceeded"}}),
    ])
    assert facts["successful_read_positions"] == [{"source_index": 2, "turn_number": 1}]
    assert facts["budget_denial_count"] == 1
    assert [item["ok"] for item in facts["tool_sequence"]] == [True, False, False]
    assert facts["tool_sequence"][-1]["executed"] is False


def test_extended_case_target_is_not_hardcoded_to_second_turn(helper, tmp_path, monkeypatch):
    cases = helper.ROOT / 'tests/fixtures/evaluation/history_bm25_extended_cases.json'
    fixture, case = helper.load_case(cases, 'history-search-online-005')
    assert len(fixture['history']) == 8 and case['target_turn'] == 4
    prepare_models(monkeypatch, '001')
    def response(llm, messages, *, stage):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            return tool_call('read_conversation_turn', {'source_index':1,'turn_number':4}, 'read4')
        d = json.loads(results[-1].content)['data']
        return AIMessage(content=d['user_text']+d['citation'])
    monkeypatch.setattr(app, 'invoke_llm_response', response)
    row = helper.run_case(fixture, case, tmp_path/'extended.jsonl', {})
    assert row['observations']['requested_read_succeeded'] is True
    assert row['observations']['exact_target_quote_in_answer'] is True


@pytest.mark.parametrize("suffix", ["008", "009"])
def test_multisource_search_scope_and_read_observation(helper, tmp_path, monkeypatch, suffix):
    cases = helper.ROOT / "tests/fixtures/evaluation/history_multisource_online_cases.json"
    fixture, case = helper.load_case(cases, "history-multisource-online-" + suffix)
    prepare_models(monkeypatch, "001")
    def response(llm, messages, *, stage):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            return tool_call("search_conversation_turns", {"query": "更正" if suffix == "008" else "暗号松柏"}, "search")
        data = json.loads(results[-1].content)["data"]
        if "matches" in data:
            assert len(data["searched_sources"]) == 2
            if suffix == "009":
                assert data["matches"] == []
                return AIMessage(content="本次查询范围未找到，不能断言你从未说过。")
            hit = data["matches"][0]
            assert hit["source_index"] == 2 and hit["turn_number"] == 1
            return tool_call("read_conversation_turn", {"source_index": 2, "turn_number": 1}, "read")
        return AIMessage(content=data["user_text"] + data["citation"])
    monkeypatch.setattr(app, "invoke_llm_response", response)
    row = helper.run_case(fixture, case, tmp_path / "results.jsonl", {})
    assert row["status"] == "completed"
    assert row["business_state_unchanged"] and row["source_unchanged"] and row["inherited_scope_unchanged"]
    assert row["observations"]["first_request_contains_target_quote"] is False
    assert row["observations"]["requested_read_succeeded"] is (suffix == "008")
    assert row["observations"]["exact_target_quote_in_answer"] is (suffix == "008")
