"""冻结生成对比的离线契约，不调用真实模型。"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def helper():
    path = Path(__file__).parent / "helpers/history_generation_comparison.py"
    spec = importlib.util.spec_from_file_location("generation_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_result(tmp_path, *, stage="交互回答-工具判断3", evidence=True):
    citation = "[history=archive/session, turn=1]"
    messages = [{"role": "system", "content": "规则保持原样。"},
                {"role": "human", "content": "核对原话。"},
                {"role": "ai", "content": "", "tool_calls": [{"name": "read_conversation_turn",
                    "args": {"source_index": 1, "turn_number": 1}, "id": "read", "type": "tool_call"}]},
                {"role": "tool", "tool_call_id": "read", "content": json.dumps({
                    "ok": evidence, "data": {"evidence_kind": "archived_conversation", "citation": citation}})}]
    observation = tmp_path / "observation.json"
    observation.write_text(json.dumps({"user_input": "核对原话。", "fixture_history": [{"user": "原话", "assistant": "了解。"}],
        "model_requests": [{"stage": stage, "request_messages": messages}]}), encoding="utf-8")
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"case_id": "test-001", "run_id": "source-run",
        "manifest": {"source_py_sha256": "source-hash"}, "observation_path": str(observation)}) + "\n", encoding="utf-8")
    return path


def test_preflight_has_readable_inputs_and_never_creates_models_or_output(helper, tmp_path, monkeypatch, capsys):
    path = source_result(tmp_path)
    output = tmp_path / "output.json"
    monkeypatch.setattr(helper, "get_settings", lambda: SimpleNamespace(chat_model="a", enhanced_model="b",
        llm_temperature=0.2, request_timeout=180, llm_retry_attempts=3))
    monkeypatch.setattr(helper, "create_llm", lambda **kw: pytest.fail("预检不能创建模型"))
    assert helper.main(["--results", str(path), "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["calls_planned"] == 2
    assert payload["cases"][0]["question"] == "核对原话。"
    assert not output.exists()


@pytest.mark.parametrize("stage,evidence", [("交互回答-强制收束", True), ("交互回答-工具判断3", False)])
def test_input_requires_normal_generation_and_successful_read(helper, tmp_path, stage, evidence):
    with pytest.raises(ValueError):
        helper.load_frozen_case(source_result(tmp_path, stage=stage, evidence=evidence))


def test_models_receive_same_messages_and_report_raw_output_without_repair(helper, tmp_path, monkeypatch):
    case = helper.load_frozen_case(source_result(tmp_path))
    monkeypatch.setattr(helper, "create_llm", lambda *, role: role)
    seen = []
    def invoke(client, messages, *, stage):
        seen.append((client, [m.model_dump() for m in messages]))
        return AIMessage(content="没有老师姓名。[history=wrong/session, turn=1]",
                         usage_metadata={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18})
    monkeypatch.setattr(helper, "invoke_llm_response", invoke)
    rows = helper.run_comparison([case], {"standard": "a", "enhanced": "b"})
    assert seen[0][1] == seen[1][1]
    assert {r["request_sha256"] for r in rows} == {case["request_sha256"]}
    assert all(r["unknown_history_citations"] and not r["has_verified_citation"] for r in rows)
    assert all(r["usage"]["total_tokens"] == 18 for r in rows)
    assert "wrong/session" in rows[0]["answer"]


def test_failure_is_recorded_without_extra_retries(helper, tmp_path, monkeypatch):
    case = helper.load_frozen_case(source_result(tmp_path))
    monkeypatch.setattr(helper, "create_llm", lambda *, role: role)
    calls = []
    def invoke(client, messages, *, stage):
        calls.append(client)
        raise RuntimeError("PRIVATE_DETAIL")
    monkeypatch.setattr(helper, "invoke_llm_response", invoke)
    rows = helper.run_comparison([case], {"standard": "a", "enhanced": "b"})
    assert len(calls) == 2
    assert all(r["status"] == "failed" and r["usage"] is None for r in rows)
    assert "PRIVATE_DETAIL" not in json.dumps(rows)
