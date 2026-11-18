"""比较实验本身的输入隔离、真实读取与usage记录。"""
from copy import deepcopy
import json

from langchain_core.messages import AIMessage, ToolMessage
import pytest

from tests.helpers import receipt_view_compare as experiment


@pytest.fixture
def cases():
    return json.loads(experiment.CASES.read_text(encoding="utf-8"))


@pytest.mark.parametrize("index", [0, 1, 2])
def test_views_only_change_successful_read_content(cases, index, tmp_path):
    history = experiment.build_history(cases[index], tmp_path)
    before = deepcopy([m.model_dump() for m in history])
    raw = experiment.request(history, "raw")
    compact = experiment.request(history, "compact")
    assert [m.model_dump() for m in history] == before
    assert len(raw) == len(compact)
    for first, second in zip(raw, compact):
        payload = json.loads(first.content) if isinstance(first, ToolMessage) else None
        if payload and payload["ok"] and not payload["data"]["truncated"]:
            a, b = first.model_dump(), second.model_dump()
            a.pop("content")
            b.pop("content")
            assert a == b
            assert first.content != second.content
        else:
            assert first.model_dump() == second.model_dump()


def test_actual_range_and_truncation(cases, tmp_path):
    history = experiment.build_history(cases[0], tmp_path / "range")
    data = json.loads(history[-1].content)["data"]
    assert data["requested_end_line"] == 200
    assert data["returned_end_line"] == 4
    history = experiment.build_history(cases[1], tmp_path / "truncated")
    data = json.loads(history[-1].content)["data"]
    assert data["truncated"] is True
    assert data["text"] == cases[1]["material"].splitlines()[0]
    assert "橙签" not in data["text"]


def test_records_actual_usage_and_keeps_quality_pending(cases, tmp_path):
    class Fake:
        def invoke(self, messages):
            return AIMessage(content="回答", usage_metadata={
                "input_tokens": 15, "output_tokens": 5, "total_tokens": 20})
    history = experiment.build_history(cases[0], tmp_path)
    row = experiment.run_one(cases[0], history, "compact", 1, Fake(), experiment.fingerprint())
    assert row["usage"]["input_tokens"] == 15
    assert row["usage"]["usage_complete"] is True
    assert len(row["attempts"]) == 1
    assert row["quality_status"] == "pending_manual_review"
    assert row["source_unchanged"] is True
    assert row["response_kind"] == "answer"


@pytest.mark.parametrize("response, kind", [
    (AIMessage(content=""), "empty"),
    (AIMessage(content="", tool_calls=[{"id": "r", "name": "read_learning_file",
        "args": {"relative_path": "markers.md"}, "type": "tool_call"}]), "tool_call"),
])
def test_successful_request_does_not_imply_answer(cases, tmp_path, response, kind):
    class Fake:
        def invoke(self, messages):
            return response
    history = experiment.build_history(cases[0], tmp_path)
    row = experiment.run_one(cases[0], history, "raw", 1, Fake(), experiment.fingerprint())
    if kind == "empty":
        assert row["status"] == "failed"
        assert row["error_type"] == "EmptyLLMResponseError"
        assert len(row["attempts"]) == 2
    else:
        assert row["status"] == "completed"
        assert row["response_kind"] == kind
    assert row["usage"]["input_tokens"] is None
