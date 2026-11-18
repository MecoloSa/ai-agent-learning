"""留出实验脚本的输入与隔离机制；不把替身结果当模型质量成绩。"""
import json
import sys

from tests.helpers import learning_holdout as probe


def test_event_order_keeps_two_independent_confirmation_inputs():
    events = probe.flow_events()
    assert [(e["stage"], e["text"]) for e in events if e["stage"] == "confirm"] == [
        ("confirm", "n"), ("confirm", "y")]
    assert events[3]["text"] == events[5]["text"] == probe.CANDIDATE
    feedback = json.loads(events[1]["text"].removeprefix("/feedback "))
    assert feedback["text"] == probe.FEEDBACK
    assert feedback["activity_status"] == "completed"
    assert feedback["understanding_status"] == "self_reported_unclear"


def test_preflight_cannot_invoke_case_and_records_actual_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["probe", "--output-dir", str(tmp_path / "preflight")])
    def forbidden(*args, **kwargs):
        raise AssertionError("预检不能执行模型任务")
    monkeypatch.setattr(probe, "run_case", forbidden)
    assert probe.main() == 0
    manifest = json.loads((tmp_path / "preflight/manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == 2
    assert manifest["mode"] == "preflight_only"
    assert manifest["events"] == probe.flow_events()
    assert all(c["planning_request"].startswith(c["goal"]) for c in manifest["cases"])


def test_failed_case_has_nonzero_exit_and_does_not_hide_second_case(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["probe", "--output-dir", str(tmp_path / "run"), "--llm"])
    called = []
    def failed(case, *args, **kwargs):
        called.append(case["case_id"])
        return {"case_id": case["case_id"], "status": "failed"}
    monkeypatch.setattr(probe, "run_case", failed)
    assert probe.main() == 1
    assert called == ["validation_test", "correlation_causation"]
