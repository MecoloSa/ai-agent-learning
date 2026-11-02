"""反馈计划案例加载契约；候选模块未应用时应报错，不跳过。"""
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def api():
    return importlib.import_module("ai_agent.evaluation.feedback_plan_cases")


@pytest.fixture
def sample(tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "note.md").write_text("合成资料", encoding="utf-8")
    snapshot = tmp_path / "state.json"
    snapshot.write_text(json.dumps({
        "schema_version": "learning-state-v7",
        "state": {"learning_goal": "阅读", "plan_markdown": "阅读10分钟"},
    }, ensure_ascii=False), encoding="utf-8")
    row = {
        "case_id": "dev-1", "split": "dev", "material_dir": "materials",
        "initial_state_file": "state.json", "inputs": ["只讨论", "exit"],
        "checks": [{"check_id": "no_write", "category": "control",
                    "criterion": "业务状态不变", "basis": "用户未授权修改"}],
    }
    return tmp_path / "cases.jsonl", row, snapshot


def write_rows(path, *rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_load_resolves_paths_without_mutating_inputs(api, sample, monkeypatch, tmp_path):
    path, row, snapshot = sample
    write_rows(path, row)
    original = snapshot.read_bytes()
    manifest = path.read_bytes()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cases = api.load_feedback_plan_cases(path)
    assert cases[0].material_dir == path.parent / "materials"
    assert cases[0].initial_state_file == snapshot
    assert snapshot.read_bytes() == original
    assert path.read_bytes() == manifest


@pytest.mark.parametrize("field,value", [
    ("case_id", "  "), ("case_id", 1), ("split", "test"),
    ("inputs", []), ("inputs", [" "]), ("checks", []),
    ("unexpected", True), ("material_dir", "missing"),
    ("material_dir", "../outside"), ("initial_state_file", "missing.json"),
    ("initial_state_file", "../outside.json"),
])
def test_invalid_contract_or_resource_is_rejected(api, sample, field, value):
    path, row, _ = sample
    row[field] = value
    write_rows(path, row)
    with pytest.raises(ValueError, match=r"cases.jsonl:1:"):
        api.load_feedback_plan_cases(path)


@pytest.mark.parametrize("field,value", [
    ("check_id", ""), ("category", "success"), ("criterion", "  "),
    ("basis", ""), ("passed", True),
])
def test_expectations_are_rules_not_unvalidated_scores(api, sample, field, value):
    path, row, _ = sample
    row["checks"][0][field] = value
    write_rows(path, row)
    with pytest.raises(ValueError):
        api.load_feedback_plan_cases(path)


def test_duplicate_case_id_reports_real_line(api, sample):
    path, row, _ = sample
    write_rows(path, row, row)
    with pytest.raises(ValueError, match=r"cases.jsonl:2:"):
        api.load_feedback_plan_cases(path)


def test_duplicate_check_id_is_rejected(api, sample):
    path, row, _ = sample
    row["checks"].append(dict(row["checks"][0]))
    write_rows(path, row)
    with pytest.raises(ValueError):
        api.load_feedback_plan_cases(path)


@pytest.mark.parametrize("payload", ["", " \n\n", "{bad", "null", "[]"])
def test_empty_or_malformed_file(api, sample, payload):
    path, _, _ = sample
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        api.load_feedback_plan_cases(path)


@pytest.mark.parametrize("payload", ["broken", '{"state":{}}', '{"schema_version":"future","state":{}}'])
def test_snapshot_must_pass_real_state_schema(api, sample, payload):
    path, row, snapshot = sample
    snapshot.write_text(payload, encoding="utf-8")
    write_rows(path, row)
    with pytest.raises(ValueError):
        api.load_feedback_plan_cases(path)


def test_absolute_resource_is_rejected(api, sample):
    path, row, snapshot = sample
    row["initial_state_file"] = str(snapshot)
    write_rows(path, row)
    with pytest.raises(ValueError):
        api.load_feedback_plan_cases(path)


def test_bom_and_blank_lines_are_accepted(api, sample):
    path, row, _ = sample
    path.write_text("\n" + json.dumps(row) + "\n\n", encoding="utf-8-sig")
    assert len(api.load_feedback_plan_cases(path)) == 1


def test_repository_seed_cases_have_rules_but_no_quality_results(api):
    path = Path(__file__).parent / "fixtures/evaluation/feedback_plan_cases.jsonl"
    cases = api.load_feedback_plan_cases(path)
    assert len(cases) == 3
    assert {c.split for c in cases} == {"dev"}
    assert sum(len(c.checks) for c in cases) == 7
    assert {c.category for case in cases for c in case.checks} == {"control", "quality", "evidence"}
    assert all("passed" not in c.model_dump() for case in cases for c in case.checks)
