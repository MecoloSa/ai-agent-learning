import pytest

from tests.helpers.context_exploration import RawReceiptStore, offload_probe, retry_empty_generation_once
from langchain_core.messages import AIMessage, ToolMessage
from ai_agent.runtime.agents import _require_analysis_text, AnalysisExecutionError


def test_reopen_restores_exact_receipt_and_deduplicates(tmp_path):
    raw = '中文\r\n    缩进和"引号"\n'
    first = RawReceiptStore(tmp_path).put(raw)
    assert RawReceiptStore(tmp_path).get(first["receipt_id"]) == raw
    assert RawReceiptStore(tmp_path).put(raw)["receipt_id"] == first["receipt_id"]
    assert len(list(tmp_path.glob("*.txt"))) == 1


def test_bad_id_and_tampered_data_are_rejected(tmp_path):
    store = RawReceiptStore(tmp_path)
    with pytest.raises(ValueError):
        store.get("../outside")
    ref = store.put("原文")
    (tmp_path / f"{ref['receipt_id']}.txt").write_text("被改动", encoding="utf-8")
    with pytest.raises(ValueError):
        store.get(ref["receipt_id"])


def test_offload_probe_does_not_claim_model_quality(tmp_path):
    result = offload_probe(tmp_path)
    assert result["exact_readback"] is True
    assert result["production_enabled"] is False
    assert result["preview_reference_chars"] < result["raw_chars"]


@pytest.mark.parametrize("kind", ["transient_empty", "persistent_empty", "tool_request"])
def test_empty_retry_prototype_preserves_evidence_and_stops(kind):
    history = [ToolMessage(content="已读取的原始证据", tool_call_id="read-1")]
    first = AIMessage(content="", tool_calls=[{"id": "next", "name": "read_material_lines",
        "args": {"relative_path": "notes.md"}, "type": "tool_call"}]) if kind == "tool_request" else AIMessage(content="")
    replies = iter([first, AIMessage(content="") if kind == "persistent_empty" else AIMessage(content="依据原文的回答")])
    seen = []
    def invoke(messages):
        seen.append([m.model_dump() for m in messages])
        return next(replies)
    response = retry_empty_generation_once(invoke, history)
    assert len(seen) == (1 if kind == "tool_request" else 2)
    assert all(request == [m.model_dump() for m in history] for request in seen)
    if kind == "persistent_empty":
        with pytest.raises(AnalysisExecutionError):
            _require_analysis_text(response.content)
    elif kind == "transient_empty":
        assert _require_analysis_text(response.content) == "依据原文的回答"
    else:
        assert response.tool_calls == first.tool_calls
