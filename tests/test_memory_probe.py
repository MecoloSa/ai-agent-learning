"""固定历史记忆诊断候选的离线验证；不请求网络。"""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


CASE = Path(__file__).parent / "fixtures/evaluation/memory_probe_case.json"
QUOTE_CASE = CASE.with_name("memory_probe_quote_case.json")


@pytest.fixture
def api():
    return importlib.import_module("ai_agent.evaluation.memory_probe")


@pytest.fixture
def fake_models(api, monkeypatch):
    calls = {"final": [], "summary": []}

    def summarize(messages):
        calls["summary"].append(messages)
        return SimpleNamespace(content="松针练习；当前35分钟；只阅读与口头说明，不绘图、不编程。",
                               usage_metadata=None)

    def answer(messages):
        calls["final"].append(messages)
        return SimpleNamespace(content="用于检查接线的固定最终回答，不代表真实召回。", usage_metadata=None)

    monkeypatch.setattr(api.memory_module, "_request_summary", summarize)
    monkeypatch.setattr(api, "create_llm", lambda: SimpleNamespace(invoke=answer))
    return calls


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
def test_same_history_only_real_memory_strategy_changes_view(api, fake_models, strategy):
    case = api.load_probe(CASE)
    before = case.model_dump()
    record = api.run_probe(case, strategy, 6, 120)
    assert record["status"] == "completed"
    assert case.model_dump() == before
    assert record["source_history"] == before["history"]
    assert len(fake_models["final"]) == 1
    assert record["final_request_messages"][-1]["content"] == case.question
    assert record["assessment"] == "not_evaluated"
    assert record["final_usage"] is None
    assert len(record["memory_steps"]) == 6
    text = json.dumps(record["final_request_messages"], ensure_ascii=False)
    if strategy == "window":
        assert len(record["final_request_messages"]) == 8
        assert "松针练习" not in text and "35分钟" not in text
    else:
        assert "松针练习" in text and "35分钟" in text
    if strategy == "summary":
        assert record["summary_adoptions"] > 0
        assert len(record["summary_calls"]) == len(fake_models["summary"])
        assert record["summary_calls"][0]["request_messages"]
    else:
        assert record["summary_calls"] == [] and record["summary_adoptions"] == 0


def test_criteria_are_not_sent_to_either_model(api, fake_models):
    case = api.load_probe(CASE).model_copy(update={"criteria": ["PRIVATE_LABEL_NOT_A_PROMPT"]})
    record = api.run_probe(case, "summary", 6, 120)
    assert record["criteria"] == case.criteria
    sent = [[m.content for m in call] for group in fake_models.values() for call in group]
    assert "PRIVATE_LABEL_NOT_A_PROMPT" not in json.dumps(sent)


def test_returned_oversized_summary_is_not_reported_as_adopted(api, fake_models, monkeypatch):
    monkeypatch.setattr(api.memory_module, "_request_summary", lambda _: SimpleNamespace(
        content="字" * (api.get_settings().memory_summary_max_chars + 1), usage_metadata=None))
    record = api.run_probe(api.load_probe(CASE), "summary", 6, 120)
    assert record["status"] == "completed"
    assert record["summary_calls"] and all(c["status"] == "returned" for c in record["summary_calls"])
    assert record["summary_adoptions"] == 0
    assert record["memory_steps"][-1]["retained_history_messages"] == 12


def test_failed_final_call_keeps_partial_observations(api, fake_models, monkeypatch):
    def fail(messages):
        raise TimeoutError("DO_NOT_EXPORT_ENDPOINT")

    monkeypatch.setattr(api, "create_llm", lambda: SimpleNamespace(invoke=fail))
    record = api.run_probe(api.load_probe(CASE), "summary", 6, 120)
    assert record["status"] == "failed" and record["error_type"] == "TimeoutError"
    assert record["summary_calls"] and record["final_request_messages"]
    assert record["final_latency_ms"] is not None
    assert "DO_NOT_EXPORT_ENDPOINT" not in json.dumps(record)


def test_preflight_creates_no_output_or_model_requests(api, tmp_path, monkeypatch, capsys):
    output = tmp_path / "new" / "result.jsonl"
    monkeypatch.setattr(sys, "argv", ["probe", "--case", str(CASE), "--strategy", "summary",
                                     "--output", str(output)])
    monkeypatch.setattr(api, "run_probe", lambda *a: pytest.fail("预检不应运行"))
    api.main()
    assert json.loads(capsys.readouterr().out)["mode"] == "preflight_only"
    assert not output.parent.exists()


def test_live_cli_saves_manifest_without_credentials(api, fake_models, tmp_path, monkeypatch):
    output = tmp_path / "result.jsonl"
    monkeypatch.setattr(sys, "argv", ["probe", "--case", str(CASE), "--strategy", "summary",
                                     "--output", str(output), "--llm"])
    api.main()
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment"]["conditions_unchanged"] is True
    manifest = record["experiment"]["before"]
    assert manifest["strategy"] == "summary" and manifest["trigger_chars"] == 120
    assert "chat_api_key" not in manifest and "chat_base_url" not in manifest
    assert len(manifest["chat_endpoint_sha256"]) == 64


def test_existing_result_is_never_overwritten(api, tmp_path, monkeypatch):
    output = tmp_path / "result.jsonl"
    output.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["probe", "--case", str(CASE), "--strategy", "full",
                                     "--output", str(output), "--llm"])
    monkeypatch.setattr(api, "run_probe", lambda *a: pytest.fail("已有结果不能执行"))
    with pytest.raises(SystemExit) as exc:
        api.main()
    assert exc.value.code == 2 and output.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("strategy,size,trigger", [("bad", 6, 120), ("full", 0, 120), ("summary", 6, 0)])
def test_invalid_runtime_profile_rejected_before_calls(api, strategy, size, trigger, monkeypatch):
    monkeypatch.setattr(api, "create_llm", lambda: pytest.fail("参数非法不能请求模型"))
    with pytest.raises(ValueError):
        api.run_probe(api.load_probe(CASE), strategy, size, trigger)


@pytest.mark.parametrize("trigger,rounds,final_count", [
    (120, [2, 3, 4, 5, 6], 5),
    (300, [5], 7),
    (2000, [], 14),
])
def test_trigger_comparison_reuses_history_and_preserves_latest_rounds(
    api, fake_models, trigger, rounds, final_count,
):
    """只核对固定历史下的触发接线，不用替身回答评价真实压缩质量。"""
    case = api.load_probe(CASE)
    record = api.run_probe(case, "summary", 6, trigger)
    assert record["status"] == "completed"
    assert [s["round"] for s in record["memory_steps"] if s["summary_adopted"]] == rounds
    assert len(record["summary_calls"]) == len(rounds)
    assert len(fake_models["final"]) == 1
    assert len(record["final_request_messages"]) == final_count
    assert record["source_history"] == [t.model_dump() for t in case.history]
    # 所有条件都保留最新一轮原文和完全相同的最终问题。
    visible = record["final_request_messages"]
    assert [m["content"] for m in visible[-3:]] == [
        case.history[-1].user, case.history[-1].assistant, case.question,
    ]


def test_quote_probe_changes_question_not_history_or_system(api):
    original, quote = api.load_probe(CASE), api.load_probe(QUOTE_CASE)
    assert quote.case_id != original.case_id
    assert quote.history == original.history and quote.system_prompt == original.system_prompt
    assert "逐字引用" in quote.question and "没有完整原文" in quote.question
    # 原文答案不得通过最后问题或系统提示泄漏给模型。
    assert quote.history[1].user not in quote.question
    assert quote.history[1].user not in quote.system_prompt


@pytest.mark.parametrize("strategy,has_quote", [("full", True), ("window", False), ("summary", False)])
def test_quote_availability_is_observed_from_actual_context(api, fake_models, strategy, has_quote):
    """摘要替身只返回概括：测试观测能区分原文与摘要，不预判真实模型输出。"""
    case = api.load_probe(QUOTE_CASE)
    record = api.run_probe(case, strategy, 6, 300)
    assert record["status"] == "completed"
    assert any(case.history[1].user in m["content"]
               for m in record["final_request_messages"]) is has_quote
    assert record["source_history"][1]["user"] == case.history[1].user
    assert record["final_request_messages"][-1]["content"] == case.question
