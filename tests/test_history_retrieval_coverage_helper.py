"""检索诊断脚本的计分与预检契约，不要求当前检索器覆盖所有换说法。"""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def helper():
    path = Path(__file__).parent / "helpers/history_retrieval_coverage.py"
    spec = importlib.util.spec_from_file_location("coverage_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_does_not_create_archives_or_output(helper, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(helper, "evaluate", lambda *a: pytest.fail("预检不能执行工具"))
    output = tmp_path / "result.json"
    assert helper.main(["--output", str(output)]) == 0
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["mode"] == "preflight_only"


def test_target_rank_negative_and_tool_failure_are_scored_separately(helper, tmp_path, monkeypatch):
    fixture = {"sources": [{"through_turn": 1, "user_turns": ["原话"]}], "assistant_text": "回复",
               "cases": [
                   {"case_id": "positive", "group": "positive", "query": "a", "targets": [[1, 1]]},
                   {"case_id": "negative", "group": "negative", "query": "b", "targets": []},
                   {"case_id": "error", "group": "negative", "query": "c", "targets": []},
               ]}
    responses = iter([
        {"ok": True, "data": {"matches": [{"source_index": 1, "turn_number": 1, "rank": 2}]}},
        {"ok": True, "data": {"matches": []}},
        {"ok": False, "error": {"code": "archive_unavailable"}},
    ])
    class Search:
        name = "search_conversation_turns"
        def invoke(self, args):
            return json.dumps(next(responses))
    class Access:
        def __init__(self, root):
            self.tools = (Search(),)
        def start_turn(self, state):
            pass
    monkeypatch.setattr(helper, "HistoryReadAccess", Access)
    result = helper.evaluate(fixture, tmp_path)
    positive, negative, error = result["rows"]
    assert positive["hit_at_1"] is False and positive["hit_at_5"] is True
    assert positive["reciprocal_rank_at_5"] == 0.5
    assert negative["negative_empty"] is True
    assert error["negative_empty"] is False and error["tool_ok"] is False
    assert result["groups"]["negative"]["negative_empty_count"] == 1
    assert result["groups"]["negative"]["tool_error_count"] == 1
