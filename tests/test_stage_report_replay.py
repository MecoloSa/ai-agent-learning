import asyncio
import json

from langchain_core.messages import AIMessage
import pytest

from tests.helpers import stage_report_replay
from tests.helpers.stage_report_replay import select_receipts, validate_report


def test_replay_uses_only_confirmed_receipts_and_deduplicates_per_page():
    def receipt(key, page, **metadata):
        return {"evidence_id": key, "page": page, "source": "book.pdf",
                "text": "tool_call_id 关联工具调用", **metadata}

    chunk = receipt("fragment", 41, chunk_index=11)
    full = receipt("full", 41)
    late = receipt("late", 42)
    other = receipt("another-task", 43)
    data = {
        "question": "tool_call_id 如何关联",
        "results": {"A": {"evidence_ids": ["fragment", "full", "another-task"]}},
        "trace": [
            {"task": "A", "result": {"evidence": [chunk, late, full]}},
            {"task": "B", "result": {"evidence": [other]}},
        ],
    }
    assert select_receipts(data, pages_per_branch=8) == {"A": [full]}


def test_stage_report_preserves_partial_findings_and_explicit_gaps():
    report = validate_report(json.dumps({
        "findings": [{"statement": "工具结果用调用 ID 关联", "evidence_ids": ["p41"]}],
        "gaps": ["尚未取得该书说明 Observation 的正文"],
    }), [{"evidence_id": "p41"}])
    assert report.findings[0].evidence_ids == ["p41"]
    assert report.gaps == ["尚未取得该书说明 Observation 的正文"]


def test_stage_report_rejects_an_id_from_another_branch():
    with pytest.raises(ValueError, match="本路未取得"):
        validate_report(json.dumps({
            "findings": [{"statement": "已找到正文", "evidence_ids": ["other"]}], "gaps": [],
        }), [{"evidence_id": "p41"}])


def test_invalid_report_leaves_confirmed_receipt_available_for_delivery(monkeypatch):
    class Model:
        model_name = "fixed"

        def model_copy(self, **kwargs):
            return self

        def bind_tools(self, tools):
            assert "read_delivery_evidence" in {tool.name for tool in tools}
            return self

        async def ainvoke(self, messages):
            payload = json.loads(messages[-1].content)
            if "raw_receipts" in payload:
                return AIMessage(content="直接交付")
            if "evidence" in payload:
                return AIMessage(content="无效阶段报告")
            assert payload["reports"]["A"]["status"] == "unavailable"
            assert payload["evidence_catalog"]["p41"]["page"] == 41
            return AIMessage(content="保留可回查证据")

    monkeypatch.setattr(stage_report_replay, "create_llm", lambda **kwargs: Model())
    record = {
        "question": "tool_call_id 如何关联",
        "results": {"A": {"evidence_ids": ["p41"]}},
        "trace": [{"task": "A", "result": {"evidence": [
            {"source": "book.pdf", "page": 41, "evidence_id": "p41", "text": "tool_call_id 关联调用"},
        ]}}],
    }
    result = asyncio.run(stage_report_replay.replay(record, pages_per_branch=8))
    assert result["direct"]["answer"] == "直接交付"
    assert result["staged"]["answer"] == "保留可回查证据"
