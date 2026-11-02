"""独立资料依据案例的加载和真实读取接线；模型使用替身，未测在线效果。"""
import json
from pathlib import Path

from langchain_core.messages import AIMessage

from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases
from ai_agent.evaluation.feedback_plan_runner import run_case


CASE_FILE = Path(__file__).parent / "fixtures/evaluation/feedback_plan_evidence_cases.jsonl"


def test_evidence_case_loads_without_changing_original_three_cases():
    original = CASE_FILE.parent / "feedback_plan_cases.jsonl"
    before = original.read_bytes()
    cases = load_feedback_plan_cases(CASE_FILE)
    assert len(cases) == 1 and cases[0].case_id == "feedback-plan-dev-004"
    assert cases[0].split == "dev"
    assert [c.category for c in cases[0].checks] == ["control", "quality", "evidence"]
    assert cases[0].checks[0].rule == "plan_declined"
    assert all(c.rule is None for c in cases[0].checks[1:])
    assert len(load_feedback_plan_cases(original)) == 3
    assert original.read_bytes() == before


def test_evidence_case_uses_real_reader_and_original_confirmation(tmp_path, monkeypatch):
    import ai_agent.main as app
    case = load_feedback_plan_cases(CASE_FILE)[0]
    before = case.initial_state_file.read_bytes()
    monkeypatch.setattr(app, "create_llm", lambda **kwargs: object())
    monkeypatch.setattr(app, "create_llm_with_tools", lambda *a, **k: object())
    monkeypatch.setattr(app, "needs_rag", lambda _: False)
    calls = 0

    def model(llm, messages, *, stage):
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "read_material_lines", "id": "evidence-read-001",
                "args": {"relative_path": "literature_notes.md", "start_line": 1, "end_line": 20},
                "type": "tool_call",
            }])
        # 工具真实读取，引用由工具实际返回，不能把测试预写引用当成读取事实。
        payload = json.loads(messages[-1].content)
        assert payload["ok"] is True
        body = "\n".join(row["text"] for row in payload["data"]["lines"])
        assert "藏" in body and "真实动机" in body
        return AIMessage(content=(
            '“我看到她把信藏了起来，却不知道她为何这样做。”'
            + payload["data"]["citation"]
            + "这体现叙述者的信息限制，不能据此认定人物真实动机。\n"
            + "===NEW_PLAN_START===\n阅读例句15分钟，口头说明15分钟。\n===NEW_PLAN_END==="
        ))

    monkeypatch.setattr(app, "invoke_llm_response", model)
    result = run_case(case, None, tmp_path, online=True)
    assert result["status"] == "completed"
    assert result["business_state_unchanged"] is True
    assert calls == 2
    tools = result["turns"][0]["audit_trace"]["tool_results"]
    assert len(tools) == 1
    assert tools[0]["tool_name"] == "read_material_lines" and tools[0]["executed"] is True
    payload = json.loads(tools[0]["result"])
    assert payload["ok"] is True and payload["data"]["source"] == "literature_notes.md"
    assert payload["data"]["citation"] in result["turns"][0]["response"]
    assert result["plan_changes"][0]["decision"] == "cancelled"
    assert result["checks"][0]["value"] is True
    assert all(c["value"] is None for c in result["checks"][1:])
    assert case.initial_state_file.read_bytes() == before
