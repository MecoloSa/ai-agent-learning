"""在线入口的离线接线与预检测试；没有真实网络请求或效果评分。"""
import importlib
import json
from pathlib import Path
import sys

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def api():
    return importlib.import_module("ai_agent.evaluation.feedback_plan_online")


@pytest.fixture
def seeds():
    from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases
    path = Path(__file__).parent / "fixtures/evaluation/feedback_plan_cases.jsonl"
    return path, load_feedback_plan_cases(path)


def set_args(monkeypatch, path, output, *, case_id="feedback-plan-dev-002", llm=False):
    argv = ["online", "--cases", str(path), "--case-id", case_id, "--output", str(output)]
    if llm:
        argv.append("--llm")
    monkeypatch.setattr(sys, "argv", argv)


def test_default_preflight_never_executes_or_creates_output(api, seeds, tmp_path, monkeypatch, capsys):
    output = tmp_path / "new/results.jsonl"
    set_args(monkeypatch, seeds[0], output)
    def forbidden(*args, **kwargs):
        raise AssertionError("预检不应执行案例")
    monkeypatch.setattr(api, "run_case", forbidden)
    api.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "preflight_only"
    assert payload["case_id"] == "feedback-plan-dev-002"
    assert payload["manifest"]["profile"]["preloaded_material_context"] == "empty"
    assert not output.parent.exists()


def test_manifest_has_allowlisted_settings_and_no_credentials(api, seeds, monkeypatch):
    from dataclasses import replace
    settings = replace(api.get_settings(), chat_api_key="DO_NOT_EXPORT_CHAT_KEY",
                       embedding_api_key="DO_NOT_EXPORT_EMBEDDING_KEY",
                       chat_base_url="https://private-endpoint.example/secret")
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    manifest = api.build_manifest(seeds[1][1], seeds[0])
    text = json.dumps(manifest)
    assert "DO_NOT_EXPORT" not in text
    assert "private-endpoint" not in text
    assert manifest["config"]["settings"]["chat_model"] == settings.chat_model
    assert manifest["config"]["settings"]["feedback_model"] == settings.feedback_model
    assert manifest["config"]["settings"]["memory_summary_max_chars"] == settings.memory_summary_max_chars
    assert len(manifest["source_py_sha256"]) == 64
    assert manifest["materials_fingerprint"].startswith("materials-v1:")


def test_manifest_detects_same_length_material_edit(api, seeds, tmp_path):
    material_dir = tmp_path / "materials"
    material_dir.mkdir()
    material = material_dir / "note.md"
    material.write_text("aaaa", encoding="utf-8")
    case = seeds[1][0].model_copy(update={"material_dir": material_dir})
    before = api.build_manifest(case, seeds[0])
    material.write_text("bbbb", encoding="utf-8")
    after = api.build_manifest(case, seeds[0])
    assert before["materials_fingerprint"] != after["materials_fingerprint"]


@pytest.mark.parametrize("situation", ["unknown_case", "existing_output", "inside_materials"])
def test_invalid_request_fails_before_execution(api, seeds, tmp_path, monkeypatch, situation):
    output = tmp_path / "results.jsonl"
    case_id = "missing" if situation == "unknown_case" else "feedback-plan-dev-002"
    if situation == "existing_output":
        output.write_text("保留原结果", encoding="utf-8")
    elif situation == "inside_materials":
        output = seeds[1][1].material_dir / "must-not-create.jsonl"
    set_args(monkeypatch, seeds[0], output, case_id=case_id, llm=True)
    monkeypatch.setattr(api, "run_case", lambda *a, **k: pytest.fail("不应执行"))
    with pytest.raises(SystemExit) as exc:
        api.main()
    assert exc.value.code == 2
    if situation == "existing_output":
        assert output.read_text(encoding="utf-8") == "保留原结果"
    else:
        assert not output.exists()


def test_live_cli_runs_only_selected_case_with_real_control_loop(api, seeds, tmp_path, monkeypatch):
    import ai_agent.main as app
    monkeypatch.setattr(app, "create_llm", lambda **kwargs: object())
    monkeypatch.setattr(app, "create_llm_with_tools", lambda *a, **k: object())
    monkeypatch.setattr(app, "needs_rag", lambda _: False)
    monkeypatch.setattr(app, "invoke_llm_response", lambda *a, **k: AIMessage(
        content="===NEW_PLAN_START===\n阅读15分钟，口述15分钟。\n===NEW_PLAN_END==="))
    output = tmp_path / "result.jsonl"
    set_args(monkeypatch, seeds[0], output, llm=True)
    before = seeds[1][1].initial_state_file.read_bytes()
    api.main()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    record = rows[0]
    assert record["case_id"] == "feedback-plan-dev-002"
    assert record["execution_mode"] == "online_model"
    assert record["status"] == "completed"
    assert record["experiment"]["conditions_unchanged"] is True
    assert record["plan_changes"][0]["decision"] == "cancelled"
    assert record["checks"][0]["value"] is True
    assert record["checks"][1]["value"] is None
    assert record["turns"][0]["request_messages"][0]["type"] == "system"
    assert record["turns"][0]["audit_trace"]["finished"] is True
    assert seeds[1][1].initial_state_file.read_bytes() == before


@pytest.mark.parametrize("situation", ["changed", "missing_after", "failed", "control_false"])
def test_nonzero_exit_preserves_result_and_condition_diagnostics(api, seeds, tmp_path, monkeypatch, situation):
    output = tmp_path / "result.jsonl"
    set_args(monkeypatch, seeds[0], output, llm=True)
    calls = 0
    def manifest(*args):
        nonlocal calls
        calls += 1
        if calls == 2 and situation == "missing_after":
            raise FileNotFoundError("不公开异常正文")
        return {"version": 2 if calls == 2 and situation == "changed" else 1}
    monkeypatch.setattr(api, "build_manifest", manifest)
    def execute(case, responses, artifacts, *, online):
        assert case.case_id == "feedback-plan-dev-002" and responses is None and online is True
        # 模型开始前必须已创建结果文件。
        assert output.exists()
        return {"status": "failed" if situation == "failed" else "completed",
                "turns": [{"response": "已收集的回复"}],
                "checks": [{"category": "control", "value": situation != "control_false"}]}
    monkeypatch.setattr(api, "run_case", execute)
    with pytest.raises(SystemExit) as exc:
        api.main()
    assert exc.value.code == 1
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["turns"][0]["response"] == "已收集的回复"
    if situation == "changed":
        assert record["experiment"]["conditions_unchanged"] is False
    elif situation == "missing_after":
        assert record["experiment"]["conditions_unchanged"] is None
        assert record["experiment"]["condition_error_type"] == "FileNotFoundError"
        assert "不公开异常正文" not in output.read_text(encoding="utf-8")


def test_output_creation_failure_prevents_execution(api, seeds, tmp_path, monkeypatch):
    parent = tmp_path / "parent-is-file"
    parent.write_text("x", encoding="utf-8")
    set_args(monkeypatch, seeds[0], parent / "result.jsonl", llm=True)
    monkeypatch.setattr(api, "run_case", lambda *a, **k: pytest.fail("结果不能创建时不应调用模型"))
    with pytest.raises(SystemExit) as exc:
        api.main()
    assert exc.value.code == 2


def test_case_id_is_required(api, seeds, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["online", "--cases", str(seeds[0]), "--output", str(tmp_path / "result.jsonl")])
    with pytest.raises(SystemExit) as exc:
        api.main()
    assert exc.value.code == 2
