"""pipeline源码证据必须准确、位于绑定目录内；不执行源码或调用模型。"""
from pathlib import Path

import pytest
import ai_agent.runtime.agents as agents


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_valid_python_source_is_read_without_silent_rewriting(tmp_path, encoding):
    body = "# 中文规则\ndef rule():\n    return 2, 3\n"
    (tmp_path / "rule.py").write_text(body, encoding=encoding)
    context = agents._collect_python_sources(str(tmp_path))
    assert context == "[source=rule.py, mode=full_read]\n" + body


def test_invalid_encoding_stops_before_any_model_is_created(tmp_path, monkeypatch):
    (tmp_path / "broken.py").write_bytes(b"def rule():\n    return '\xff'\n")

    def unexpected(*args, **kwargs):
        raise AssertionError("坏源码不应发送到模型")

    monkeypatch.setattr(agents, "create_llm", unexpected)
    with pytest.raises(agents.AnalysisExecutionError) as caught:
        agents.analyse_materials(str(tmp_path), "解释rule的返回值")
    assert caught.value.reason == "source_read_failed"
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


def test_collector_uses_shared_path_boundary_before_reading(tmp_path, monkeypatch):
    (tmp_path / "rule.py").write_text("return_value = 3", encoding="utf-8")
    calls = []

    def blocked(root_dir, relative_path):
        calls.append((root_dir, relative_path))
        raise ValueError("模拟公共目录边界校验拒绝")

    # 此项验证接线契约，不冒充在当前Windows环境真实创建了越界符号链接。
    monkeypatch.setattr(agents, "safe_material_path", blocked, raising=False)
    with pytest.raises(agents.AnalysisExecutionError) as caught:
        agents._collect_python_sources(str(tmp_path))
    assert caught.value.reason == "source_read_failed"
    assert calls == [(str(tmp_path.resolve()), "rule.py")]


def test_io_failure_is_a_typed_analysis_failure(tmp_path, monkeypatch):
    source = tmp_path / "rule.py"
    source.write_text("return_value = 3", encoding="utf-8")
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == source:
            raise PermissionError("模拟读取权限失败")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(agents.AnalysisExecutionError) as caught:
        agents._collect_python_sources(str(tmp_path))
    assert caught.value.reason == "source_read_failed"
    assert isinstance(caught.value.__cause__, PermissionError)


@pytest.mark.parametrize("orchestrator", ["legacy", "langgraph"])
def test_source_failure_stops_analysis_and_planning_in_both_workflows(tmp_path, monkeypatch, orchestrator):
    (tmp_path / "broken.py").write_bytes(b"value = '\xff'\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("不应创建模型或进入规划")

    monkeypatch.setattr(agents, "create_llm", forbidden)
    monkeypatch.setattr(agents, "create_study_plan", forbidden)
    with pytest.raises(agents.AnalysisExecutionError) as caught:
        agents.run_learning_assistant(str(tmp_path), "解释源码", 1, "pipeline", orchestrator=orchestrator)
    assert caught.value.reason == "source_read_failed"


def test_valid_replacement_character_is_not_rejected_as_bad_encoding(tmp_path):
    body = "# 合法UTF-8文本中的字符：�\nvalue = 3\n"
    (tmp_path / "rule.py").write_text(body, encoding="utf-8")
    assert body in agents._collect_python_sources(str(tmp_path))
