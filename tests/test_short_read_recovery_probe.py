import json

import pytest

from ai_agent.resources.tools import build_read_material_lines_tool
from tests.helpers.short_read_recovery import recover_short_range


@pytest.mark.parametrize(("lines", "expected"), [(2, True), (0, False), (41, False)])
def test_recovery_only_returns_complete_short_file(tmp_path, lines, expected):
    (tmp_path / "notes.md").write_text("正文\n" * lines, encoding="utf8")
    read = build_read_material_lines_tool(str(tmp_path)).func
    result = json.loads(recover_short_range(read, "notes.md", 70, 80))
    assert result["ok"] is expected
    if expected:
        assert result["data"]["citation"] == "[source=notes.md, lines=1-2]"
        assert result["data"]["range_recovery"]["requested_start_line"] == 70


def test_truncated_fallback_preserves_explicit_error(tmp_path):
    (tmp_path / "notes.md").write_text("长" * 9000, encoding="utf8")
    read = build_read_material_lines_tool(str(tmp_path)).func
    result = json.loads(recover_short_range(read, "notes.md", 70, 80))
    assert result["error"]["code"] == "line_out_of_range"
