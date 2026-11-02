"""离线运行器只验证回放/隔离/失败分类，固定坏候选也不能变成质量通过。"""
import importlib
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def api():
    return importlib.import_module("ai_agent.evaluation.feedback_plan_runner")


@pytest.fixture
def seeds():
    from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases
    root = Path(__file__).parent / "fixtures/evaluation"
    return (root / "feedback_plan_cases.jsonl", load_feedback_plan_cases(root / "feedback_plan_cases.jsonl"),
            root / "feedback_plan/offline_responses.json")


@pytest.mark.parametrize("index", [0, 1, 2])
def test_seed_replay_preserves_declined_state_without_quality_grade(api, seeds, tmp_path, index):
    _, cases, responses_file = seeds
    case = cases[index]
    before = case.initial_state_file.read_bytes()
    result = api.run_case(case, json.loads(responses_file.read_text(encoding="utf-8"))[case.case_id], tmp_path)
    assert result["status"] == "completed"
    assert result["execution_mode"] == "offline_stub"
    assert result["business_state_unchanged"] is True
    assert result["source_unchanged"] is True
    assert result["assessment"] == "not_evaluated"
    assert case.initial_state_file.read_bytes() == before
    assert (Path(result["artifacts"]) / "current.json").is_file()
    assert (Path(result["artifacts"]) / "console.txt").is_file()


def test_no_candidate_is_script_mismatch_and_preserves_partial_artifacts(api, seeds, tmp_path):
    result = api.run_case(seeds[1][1], ["没有候选"], tmp_path)
    assert result["status"] == "script_mismatch"
    assert result["stub_response_count"] == 1
    assert len(result["input_events"]) == 1
    assert (Path(result["artifacts"]) / "current.json").exists()
    assert result["assessment"] == "not_evaluated"


def test_missing_reply_is_not_implicit_empty_answer(api, seeds, tmp_path):
    result = api.run_case(seeds[1][0], [], tmp_path)
    assert result["status"] == "script_mismatch"
    assert result["stub_response_count"] == 0


def test_extra_reply_is_not_silently_ignored(api, seeds, tmp_path):
    result = api.run_case(seeds[1][0], ["只讨论", "未使用的回复"], tmp_path)
    assert result["status"] == "script_mismatch"


def test_runtime_error_is_failed_and_patches_restore(api, seeds, tmp_path, monkeypatch):
    original = api.app.create_llm
    def fail(*args, **kwargs):
        raise RuntimeError("不应公开的配置内容")
    monkeypatch.setattr(api.app, "run_interactive_session", fail)
    result = api.run_case(seeds[1][0], ["回复"], tmp_path)
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert "不应公开" not in json.dumps(result, ensure_ascii=False)
    assert api.app.create_llm is original


def test_state_commands_are_not_supported_in_this_step(api, seeds, tmp_path):
    case = seeds[1][0].model_copy(update={"inputs": ["/save-state", "exit"]})
    result = api.run_case(case, [], tmp_path)
    assert result["status"] == "script_mismatch"


def test_independent_runs_never_share_current_file(api, seeds, tmp_path):
    case = seeds[1][0]
    first = api.run_case(case, ["只讨论"], tmp_path)
    second = api.run_case(case, ["只讨论"], tmp_path)
    assert first["artifacts"] != second["artifacts"]
    assert Path(first["artifacts"]).parent == tmp_path.resolve()


def test_completed_is_not_unchanged_for_adopted_candidate(api, seeds, tmp_path):
    case = seeds[1][1].model_copy(update={"inputs": ["给候选", "v", "y", "exit"]})
    reply = "===NEW_PLAN_START===\n一个新的固定测试计划。\n===NEW_PLAN_END==="
    result = api.run_case(case, [reply], tmp_path)
    assert result["status"] == "completed"
    assert result["business_state_unchanged"] is False
    assert result["assessment"] == "not_evaluated"


def test_cli_persists_each_case_and_rejects_existing_output(api, seeds, tmp_path, monkeypatch):
    case_file, _, responses_file = seeds
    output = tmp_path / "results.jsonl"
    argv = ["runner", "--cases", str(case_file), "--responses", str(responses_file), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    api.main()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert all(r["status"] == "completed" and r["assessment"] == "not_evaluated" for r in rows)
    original = output.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("不应重新执行案例")
    monkeypatch.setattr(api, "run_case", forbidden)
    with pytest.raises(SystemExit) as exc:
        api.main()
    assert exc.value.code == 2
    assert output.read_bytes() == original


def test_cli_keeps_later_cases_and_returns_nonzero_after_mismatch(api, seeds, tmp_path, monkeypatch):
    case_file, cases, responses_file = seeds
    replies = json.loads(responses_file.read_text(encoding="utf-8"))
    replies[cases[0].case_id] = []
    changed = tmp_path / "responses.json"
    changed.write_text(json.dumps(replies, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(sys, "argv", ["runner", "--cases", str(case_file), "--responses", str(changed), "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        api.main()
    assert exc.value.code == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in rows] == ["script_mismatch", "completed", "completed"]
