"""历史引用出口契约；业务候选应用前，部分断言应失败而不能跳过。"""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import ai_agent.main as app


CITATION = "[history=archive-a/session-a, turn=1]"
FOREIGN = "[history=archive-other/session-other, turn=1]"
QUOTE = "今天在地铁上，能听讲解但不方便看图。"


def run_turn(monkeypatch, text, *, read=True, success=True, reuse=False,
             request_messages=()):
    settings = replace(app.get_settings(), interactive_max_retrieval_calls=8,
                       interactive_max_no_progress_calls=4)
    monkeypatch.setattr(app, "get_settings", lambda: settings)
    invocations = []

    def invoke(args):
        invocations.append(args)
        if not success:
            return json.dumps({"ok": False, "error": {"code": "archive_unavailable"}})
        return json.dumps({"ok": True, "data": {
            "user_text": QUOTE, "citation": CITATION,
            "citation_ref": "[history-ref=1:1]",
            "source_index": 1, "turn_number": 1,
        }})

    tool = SimpleNamespace(name="read_conversation_turn", invoke=invoke)

    def model(llm, messages, *, stage):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if read and len(results) < (2 if reuse else 1):
            return AIMessage(content="", tool_calls=[{
                "name": tool.name, "args": {"source_index": 1, "turn_number": 1},
                "id": str(len(results)), "type": "tool_call",
            }])
        return AIMessage(content=text)

    monkeypatch.setattr(app, "invoke_llm_response", model)
    trace = {}
    answer = app._invoke_interactive_turn(
        request_messages=list(request_messages), user_input="核对地铁原话，不修改记录。",
        plain_llm=object(), tool_llm=object() if read else None,
        retrieval_tool=None, extra_tools=(tool,) if read else (), audit_trace=trace,
    )
    return answer, trace, invocations


@pytest.mark.parametrize("text", [QUOTE + CITATION, "记录里没有老师姓名。", "[source=notes.md, lines=1-2]"])
def test_valid_or_absent_history_citation_does_not_change_answer(monkeypatch, text):
    answer, trace, _ = run_turn(monkeypatch, text)
    assert answer == text
    assert trace["final_answer"] == answer


@pytest.mark.parametrize("bad", [
    "[history=archive-a/session, turn=1]", FOREIGN,
    "[history=archive-a/session-a, turn=2]",
    "[history=archive-a/session-a, turn=1",
])
def test_unverified_citation_is_not_delivered_as_a_history_reference(monkeypatch, bad):
    raw = QUOTE + "\n" + bad
    answer, trace, _ = run_turn(monkeypatch, raw)
    # 未闭合引用可能恰好是正确引用的前缀；只检查模型正文，
    # 不把程序随后附加的完整真实来源误判成错误引用残留。
    body = answer.split("\n\n【历史引用核验】", 1)[0]
    assert bad not in body
    assert "历史引用未核实" in answer
    assert CITATION in answer  # 程序附上真实读取来源，而不是猜测修复错误编号。
    assert "不代表上述论断已核实" in answer
    assert trace["model_responses"][-1]["content"] == raw
    assert trace["final_answer"] == answer


def test_mixed_references_preserve_valid_one_and_remove_foreign_one(monkeypatch):
    answer, _, _ = run_turn(monkeypatch, QUOTE + CITATION + FOREIGN)
    assert CITATION in answer and FOREIGN not in answer
    assert "历史引用未核实" in answer


@pytest.mark.parametrize("read,success", [(False, True), (True, False)])
def test_no_read_or_failed_read_cannot_authorize_reference(monkeypatch, read, success):
    answer, _, _ = run_turn(monkeypatch, "原话" + CITATION, read=read, success=success)
    assert CITATION not in answer
    assert "历史引用未核实" in answer
    assert "本轮实际回读来源" not in answer


def test_verified_references_are_reset_for_next_user_turn(monkeypatch):
    answer, _, _ = run_turn(monkeypatch, QUOTE + CITATION)
    assert answer == QUOTE + CITATION
    # 上轮助手正文不是本轮实际回读凭据。
    second, _, _ = run_turn(monkeypatch, QUOTE + CITATION, read=False,
                           request_messages=[AIMessage(content=answer)])
    assert CITATION not in second and "历史引用未核实" in second


def test_cached_success_retains_original_verified_citation(monkeypatch):
    answer, trace, invocations = run_turn(monkeypatch, QUOTE + CITATION, reuse=True)
    assert answer == QUOTE + CITATION
    assert len(invocations) == 1
    assert trace["tool_results"][-1]["result_reused"] is True


def test_citation_guard_does_not_remove_plan_confirmation_protocol(monkeypatch):
    raw = "===NEW_PLAN_START===\n仅听讲解，待确认。" + FOREIGN + "\n===NEW_PLAN_END==="
    answer, _, _ = run_turn(monkeypatch, raw)
    assert app._extract_updated_plan(answer) is not None
    assert FOREIGN not in answer
    assert answer.count(app.PLAN_START_TAG) == 1
    assert answer.count(app.PLAN_END_TAG) == 1


@pytest.mark.parametrize("reuse", [False, True])
def test_short_reference_is_rendered_by_program_after_successful_read(monkeypatch, reuse):
    raw = QUOTE + "[history-ref=1:1]"
    answer, trace, invocations = run_turn(monkeypatch, raw, reuse=reuse)
    assert answer == QUOTE + CITATION
    assert len(invocations) == 1
    assert trace["model_responses"][-1]["content"] == raw
    assert trace["final_answer"] == answer


@pytest.mark.parametrize("reference", ["[history-ref=2:1]", "[history-ref=1:2]", "[history-ref=1:1"])
def test_unknown_or_incomplete_short_reference_is_not_guessed(monkeypatch, reference):
    answer, _, _ = run_turn(monkeypatch, QUOTE + reference)
    assert reference not in answer.split("\n\n【历史引用核验】", 1)[0]
    assert "历史引用未核实" in answer


def test_previous_turn_short_reference_is_not_authorized_without_new_read(monkeypatch):
    first, _, _ = run_turn(monkeypatch, QUOTE + "[history-ref=1:1]")
    assert first == QUOTE + CITATION
    answer, _, _ = run_turn(monkeypatch, "[history-ref=1:1]", read=False,
                           request_messages=[AIMessage(content=first)])
    assert "[history-ref=" not in answer and CITATION not in answer
    assert "历史引用未核实" in answer
