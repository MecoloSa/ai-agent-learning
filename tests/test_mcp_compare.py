"""对照结果统计测试；真实协议由stdio与适配测试覆盖。"""
import json

import pytest

comparison = pytest.importorskip(
    "ai_agent.evaluation.mcp_compare",
    reason="MCP对照入口方案尚未应用",
)


class Reader:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def invoke(self, arguments):
        self.calls += 1
        return json.dumps({"ok": True, "data": {"text": self.text}})


def test_compare_counts_both_readers_and_omits_body():
    local, remote = Reader("PRIVATE-TEXT"), Reader("PRIVATE-TEXT")
    result = comparison.compare_readers(local, remote, {}, 3)
    assert result["all_results_equal"] is True
    assert local.calls == remote.calls == 3
    assert len(result["pairs"]) == 3
    assert "PRIVATE-TEXT" not in json.dumps(result)


def test_difference_is_not_reported_as_success():
    result = comparison.compare_readers(Reader("a"), Reader("b"), {}, 2)
    assert result["all_results_equal"] is False


def test_invalid_count_does_not_execute_tools():
    reader = Reader("a")
    with pytest.raises(ValueError):
        comparison.compare_readers(reader, reader, {}, 21)
    assert reader.calls == 0
