import json
from langchain_core.messages import AIMessage
from tests.helpers import context_pressure_online as experiment


def test_pressure_fixture_returns_complete_real_receipts(tmp_path):
    history = experiment.prepare_fixture(tmp_path / "materials", 4)
    receipts = [json.loads(m.content)["data"] for m in history if m.type == "tool"]
    assert len(receipts) == 4
    assert all(not receipt["truncated"] for receipt in receipts)
    assert sum("LARCH_CODE" in receipt["text"] for receipt in receipts) == 1


def test_offload_candidate_really_reads_original(tmp_path, monkeypatch):
    class Fake:
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            if messages[-1].type == "tool" and messages[-1].tool_call_id == "readback":
                return AIMessage(content="要求7次。[source=target.md]")
            ref = next(json.loads(m.content) for m in messages if m.type == "tool"
                and json.loads(m.content).get("source") == "target.md")
            assert "LARCH_CODE" not in ref["preview"]
            return AIMessage(content="", tool_calls=[{"id": "readback", "name": "read_stored_receipt",
                "args": {"receipt_id": ref["receipt_id"]}, "type": "tool_call"}])
    monkeypatch.setattr(experiment, "create_llm", lambda: Fake())
    history = experiment.prepare_fixture(tmp_path / "materials", 1)
    result = experiment.run(history, "offload", tmp_path / "store")
    assert result["status"] == "completed"
    assert result["readbacks"][0]["source"] == "target.md"
    assert len(result["requests"]) == 2
    assert "LARCH_CODE" in history[-1].content
