"""在线辅助仅验证输入和预检边界，测试本身不请求模型。"""
import json
import sys
import pytest

from tests.helpers import scaffold_online as module


def test_preflight_does_not_call_model_or_create_results(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["probe", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(module, "run", lambda *a, **k: pytest.fail("预检不得生成"))
    assert module.main() == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["max_tasks"] == 12
    assert manifest["stage"] == "planning_only"
    assert len(manifest["cases"]) == 3
    assert not (tmp_path / "results.jsonl").exists()


def test_preflight_can_limit_to_revised_case(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["probe", "--output-dir", str(tmp_path), "--case-id", "three_tasks", "--repeats", "1"])
    assert module.main() == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["max_tasks"] == 2
    assert manifest["cases"][0]["case_id"] == "three_tasks"


def test_preflight_never_overwrites_existing_manifest(monkeypatch, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("原始实验", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["probe", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        module.main()
    assert path.read_text(encoding="utf-8") == "原始实验"
