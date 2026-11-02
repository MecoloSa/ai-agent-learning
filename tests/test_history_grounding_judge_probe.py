"""裁判探针的结构与标签隔离测试，不评价真实模型的判断能力。"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def helper():
    import sys
    path = Path(__file__).parent / "helpers/history_grounding_judge_probe.py"
    spec = importlib.util.spec_from_file_location("history_judge_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def case(helper):
    return json.loads(helper.DEFAULT_CASES.read_text(encoding="utf-8"))["cases"][0]


def test_labels_and_basis_are_not_sent_to_judge(helper):
    value = case(helper)
    value.update(expected="SECRET_EXPECTED", basis="SECRET_BASIS")
    messages = helper.review_messages(value)
    payload = json.loads(messages[1].content)
    assert set(payload) == {"evidence", "question", "answer"}
    assert "SECRET" not in str([m.content for m in messages])


def test_preflight_never_creates_client_or_result(helper, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(helper, "get_settings", lambda: SimpleNamespace(chat_model="a", feedback_model="b", enhanced_model="c",
        llm_temperature=0.2, request_timeout=180, llm_retry_attempts=3))
    monkeypatch.setattr(helper, "create_llm", lambda **kw: pytest.fail("预检不能创建客户端"))
    output = tmp_path / "result.json"
    assert helper.main(["--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["calls_planned"] == 9 and not output.exists()


def test_accepts_valid_review_without_confusing_structure_and_semantics(helper):
    value = case(helper)
    raw = {"verdict": "pass", "problem_spans": [], "evidence_quotes": ["不方便看图"], "reason": "原话支持。"}
    assert helper.parse_review(json.dumps(raw), value) == raw


@pytest.mark.parametrize("changes", [
    {"verdict": "fail", "problem_spans": []},
    {"verdict": "pass", "problem_spans": ["能听讲解"]},
    {"verdict": "fail", "problem_spans": ["不存在的片段"]},
    {"evidence_quotes": ["用户说晕车"]},
    {"reason": ""},
    {"verdict": True},
    {"extra_field": "不应接收"},
])
def test_rejects_invalid_or_unlocatable_review(helper, changes):
    raw = {"verdict": "pass", "problem_spans": [], "evidence_quotes": [], "reason": "简短理由。", **changes}
    with pytest.raises(ValueError):
        helper.parse_review(json.dumps(raw), case(helper))


def test_disagreement_does_not_change_reference_label(helper, monkeypatch):
    value = case(helper)
    monkeypatch.setattr(helper, "create_llm", lambda **kw: object())
    monkeypatch.setattr(helper, "invoke_llm_response", lambda *a, **kw: AIMessage(content=json.dumps({
        "verdict": "fail", "problem_spans": ["能听讲解"], "evidence_quotes": [], "reason": "模拟误报。"})))
    rows = helper.run_probe([value], "feedback")
    assert rows[0]["status"] == "completed"
    assert rows[0]["expected"] == "pass" and rows[0]["agrees_with_reference"] is False
    assert value["expected"] == "pass"


def test_invocation_failure_is_not_a_semantic_failure(helper, monkeypatch):
    monkeypatch.setattr(helper, "create_llm", lambda **kw: object())
    def fail(*a, **kw):
        raise RuntimeError("PRIVATE_SECRET")
    monkeypatch.setattr(helper, "invoke_llm_response", fail)
    row = helper.run_probe([case(helper)], "feedback")[0]
    assert row["status"] == "invalid_or_failed" and row["review"] is None
    assert row["agrees_with_reference"] is None
    assert "PRIVATE_SECRET" not in json.dumps(row)
