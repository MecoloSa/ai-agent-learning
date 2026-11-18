"""替换候选的机制与边界；不调用在线模型。"""
from dataclasses import replace
import json

from langchain_core.messages import AIMessage, ToolMessage
import pytest

import ai_agent.runtime.agents as agents
from tests.helpers.scaffold_lab import candidate_sources, fingerprint, replacement, stage_evidence


def test_all_profiles_compile():
    for profile in ("baseline", "freedom", "recovery", "evidence", "combined"):
        for name, source in candidate_sources(profile).items():
            compile(source, name, "exec")


def test_shadow_load_restores_original_module_and_disk(tmp_path):
    before, original = fingerprint(), agents._limit_evidence_context
    with replacement("combined", tmp_path):
        assert agents._limit_evidence_context is not original
    assert agents._limit_evidence_context is original
    assert fingerprint() == before


def receipt():
    return {"ok": True, "data": {
        "source": "notes.md", "citation": "[source=notes.md, lines=301-302]",
        "requested_start_line": 301, "requested_end_line": 400,
        "returned_start_line": 301, "returned_end_line": 302, "truncated": True,
        "lines": [{"line": 301, "text": "  required_count = 2", "partial": False},
                  {"line": 302, "text": "  total_count =", "partial": True}],
    }}


def test_stage_retains_actual_range_original_body_and_partial():
    result = stage_evidence(json.dumps(receipt(), ensure_ascii=False))
    assert "lines=301-302" in result and "400" not in result
    assert "301:   required_count = 2" in result
    assert "302（部分）:   total_count =" in result and "截断=True" in result


@pytest.mark.parametrize("kind", ["unknown", "failure", "noncontiguous", "extra_row", "extra_payload"])
def test_stage_unknown_or_noncanonical_receipts_are_preserved(kind):
    data = receipt()
    if kind == "unknown":
        data["data"]["new_metadata"] = "不能丢失"
    elif kind == "failure":
        data["ok"] = False
    elif kind == "noncontiguous":
        data["data"]["lines"][1]["line"] = 304
    elif kind == "extra_row":
        data["data"]["lines"][0]["extra"] = "必须保留"
    else:
        data["extra"] = "必须保留"
    text = json.dumps(data, ensure_ascii=False)
    assert stage_evidence(text) == text


def test_other_sources_and_non_json_stay_unchanged():
    text = "[source=notes.pdf, page=2]\n原文证据"
    assert stage_evidence(text) == text


def call(number, path="a.md"):
    return AIMessage(content="", tool_calls=[{
        "id": f"call-{number}", "name": "read_learning_file", "args": {"relative_path": path}, "type": "tool_call",
    }])


def run_script(monkeypatch, responses, tool, budget=12):
    iterator = iter(responses)
    observed = []
    monkeypatch.setattr(agents, "settings", replace(agents.settings, agent_max_tool_calls=budget))
    monkeypatch.setattr(agents, "MAX_TOOL_ROUNDS", 12)
    monkeypatch.setattr(agents, "build_agent_tools", lambda *a, **k: [tool])
    monkeypatch.setattr(agents, "create_llm_with_tools", lambda *a, **k: object())

    def invoke(llm, messages, **kwargs):
        observed[:] = [m for m in messages if isinstance(m, ToolMessage)]
        return next(iterator)

    monkeypatch.setattr(agents, "_invoke_llm_raw", invoke)
    return lambda: agents.analyse_materials_agentic("unused", "读取资料并比较差异"), observed


class ReadTool:
    name = "read_learning_file"

    def __init__(self, failures=0, error_type=TimeoutError):
        self.count, self.failures, self.error_type = 0, failures, error_type

    def invoke(self, args):
        self.count += 1
        if self.count <= self.failures:
            raise self.error_type("本地故障注入")
        return "[source=a.md]\n读取成功"


@pytest.mark.parametrize("profile", ["applied", "recovery"])
def test_transient_same_args_can_retry_with_existing_budget(monkeypatch, tmp_path, profile):
    with replacement(profile, tmp_path):
        tool = ReadTool(failures=1)
        run, observed = run_script(monkeypatch, [call(1), call(2), AIMessage(content="有据分析")], tool)
        assert run()[0] == "有据分析"
        assert tool.count == 2
        assert json.loads(observed[0].content)["error"]["retryable"] is True
        assert [m.tool_call_id for m in observed] == ["call-1", "call-2"]


@pytest.mark.parametrize("error_type,expected_calls", [(TimeoutError, 3), (ValueError, 1)])
@pytest.mark.parametrize("profile", ["applied", "recovery"])
def test_retries_remain_bounded_and_deterministic_errors_do_not_repeat(monkeypatch, tmp_path, error_type, expected_calls, profile):
    with replacement(profile, tmp_path):
        tool = ReadTool(failures=99, error_type=error_type)
        run, _ = run_script(monkeypatch, [*[call(n) for n in range(1, 5)], AIMessage(content="证据不足")], tool)
        try:
            run()
        except agents.AnalysisExecutionError:
            pass
        assert tool.count == expected_calls


@pytest.mark.parametrize("profile", ["applied", "combined"])
def test_retry_does_not_bypass_total_budget(monkeypatch, tmp_path, profile):
    with replacement(profile, tmp_path):
        tool = ReadTool(failures=99)
        run, _ = run_script(monkeypatch, [call(1), call(2), call(3)], tool, budget=2)
        with pytest.raises(agents.AnalysisExecutionError, match="总执行预算"):
            run()
        assert tool.count == 2


@pytest.mark.parametrize("profile", ["applied", "combined"])
def test_seven_distinct_reads_share_total_instead_of_six_read_quota(monkeypatch, tmp_path, profile):
    with replacement(profile, tmp_path):
        tool = ReadTool()
        run, observed = run_script(monkeypatch, [*[call(n, f"{n}.md") for n in range(1, 8)], AIMessage(content="比较完成")], tool)
        assert run()[0] == "比较完成"
        assert tool.count == 7
        assert len(observed) == 7


@pytest.mark.parametrize("profile", ["applied", "combined"])
def test_same_success_is_still_not_reexecuted(monkeypatch, tmp_path, profile):
    with replacement(profile, tmp_path):
        tool = ReadTool()
        run, observed = run_script(monkeypatch, [call(1), call(2), AIMessage(content="完成")], tool)
        run()
        assert tool.count == 1
        assert json.loads(observed[1].content)["error"] == "duplicate_tool_call"


def test_real_reading_is_raw_in_audit_but_rendered_for_planner(monkeypatch, tmp_path):
    import ai_agent.resources.tools as tools
    (tmp_path / "notes.md").write_text("定义A\n定义B\n", encoding="utf-8")
    blocks = []
    reader = tools.build_read_material_lines_tool(str(tmp_path), evidence_blocks=blocks)
    result = reader.invoke({"relative_path": "notes.md", "start_line": 1, "end_line": 100})
    assert blocks == [result]
    assert json.loads(result)["data"]["requested_end_line"] == 100
    with replacement("evidence", tmp_path / "replacement"):
        captured = []
        monkeypatch.setattr(agents, "needs_rag", lambda _: False)
        monkeypatch.setattr(agents, "create_llm", lambda: object())

        def invoke(llm, messages, **kwargs):
            captured.extend(messages)
            return "计划正文"

        monkeypatch.setattr(agents, "_invoke_llm", invoke)
        trace = {}
        agents.create_study_plan("分析", result, "理解定义", 1, material_dir=str(tmp_path), mode="pipeline", plan_evidence_trace=trace)
        evidence = trace["initial_evidence"]
        assert "[source=notes.md, lines=1-2]" in evidence
        assert "1: 定义A" in evidence and "2: 定义B" in evidence
        assert "100" not in evidence and "requested_end_line" not in evidence
        assert evidence in captured[1].content
        assert json.loads(result)["data"]["requested_end_line"] == 100


def test_failure_fact_stays_separate_from_planning_evidence(monkeypatch, tmp_path):
    from ai_agent.runtime.context_handoff import unpack_material_context
    (tmp_path / "notes.md").write_text("定义A", encoding="utf-8")
    with replacement("combined", tmp_path / "replacement"):
        responses = iter([call(1, "missing.md"), call(2, "notes.md"), AIMessage(content="有据分析")])
        observations = []
        monkeypatch.setattr(agents, "create_llm_with_tools", lambda _: object())

        def invoke(llm, messages, **kwargs):
            observations[:] = [m for m in messages if isinstance(m, ToolMessage)]
            return next(responses)

        monkeypatch.setattr(agents, "_invoke_llm_raw", invoke)
        _, context = agents.analyse_materials_agentic(str(tmp_path), "理解定义")
        evidence, facts = unpack_material_context(context)
        assert json.loads(observations[0].content)["error"]["code"] == "FileNotFoundError"
        assert json.loads(observations[0].content)["error"]["retryable"] is False
        assert "missing.md" not in evidence
        assert "定义A" in evidence
        assert facts == [{"tool": "read_learning_file", "status": "failed", "reason": "FileNotFoundError", "source": "missing.md"}]
