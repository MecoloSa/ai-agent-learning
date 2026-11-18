from threading import Event

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from tests.helpers.wink_observer import Observer


def test_observer_pass_keeps_original_and_does_not_regenerate():
    hook = Observer("agent", lambda _: {"decision": "pass"}, {}, "解释材料")
    answer = AIMessage(content="原答复")
    calls = []
    try:
        result = hook.invoke(lambda *a, **k: calls.append(a) or answer, None, [], stage="交互回答-工具判断1")
        assert result is answer and len(calls) == 1
        assert hook.events == [{"status": "passed"}]
    finally:
        hook.close()


def test_nudge_passes_original_draft_and_message_back_only_once():
    checked = []
    hook = Observer("agent", lambda snapshot: checked.append(snapshot) or {
        "decision": "nudge", "feedback": "目录没有覆盖小节。"
    }, {}, "请核实小节")
    calls = []

    def main(_, messages, **options):
        calls.append(messages)
        return AIMessage(content="未核对正文" if len(calls) == 1 else "仅凭目录无法确认")

    try:
        first = [HumanMessage(content="请核实小节")]
        result = hook.invoke(main, None, first, stage="交互回答-工具判断1")
        assert checked[0]["draft"] == "未核对正文"
        assert "目录没有覆盖小节" in calls[1][-1].content
        assert result.content == "仅凭目录无法确认"
        hook.invoke(main, None, first, stage="交互回答-工具判断2")
        assert len(calls) == 3 and len(checked) == 1
    finally:
        hook.close()


def test_observer_failure_does_not_block_answer():
    hook = Observer("agent", lambda _: {"decision": "unknown"}, {}, "请核实小节")
    try:
        result = hook.invoke(lambda *a, **k: AIMessage(content="保留原答复"), None, [],
                             stage="交互回答-工具判断1")
        assert result.content == "保留原答复" and hook.events[0]["status"] == "error"
    finally:
        hook.close()


def test_async_observer_does_not_wait_before_final_reply():
    release = Event()
    audit = {"tool_results": [{"tool_name": "read_material_lines", "args": {}, "result": "{}"}]}

    def review(_):
        assert release.wait(3)
        return {"decision": "nudge", "feedback": "此建议已错过交付边界。"}

    hook = Observer("async", review, audit, "请核实小节")
    try:
        result = hook.invoke(lambda *a, **k: AIMessage(content="原答复"), None, [],
                             stage="交互回答-工具判断2")
        assert result.content == "原答复" and hook.events == [{"status": "scheduled"}]
        hook.close()
        assert hook.events[-1] == {"status": "no_later_boundary", "result_ready": False}
    finally:
        release.set()
        hook.close()


def test_grounded_observer_distinguishes_failed_read_from_source_text():
    audit = {"tool_results": [{"tool_name": "read_material_lines", "args": {"relative_path": "notes.md"},
                               "result": '{"ok":false,"error":{"code":"line_out_of_range"}}'}]}
    hook = Observer("grounded", lambda _: {"decision": "pass"}, audit, "求原文")
    try:
        assert hook._snapshot()["program_read_outcomes"] == [{
            "tool": "read_material_lines", "source": "notes.md",
            "returned_body_text": False, "error_code": "line_out_of_range",
        }]
    finally:
        hook.close()


@pytest.mark.parametrize("calls", [
    [],
    [{"executed": True, "tool_name": "retrieve_learning_context", "args": {},
      "result": '{"ok":true,"data":{"text":"原文片段"}}'}],
    [{"executed": True, "tool_name": "read_pdf_pages", "args": {},
      "result": '{"ok":true,"data":{"pages":[{"text":"PDF原文"}]}}'}],
])
def test_gated_observer_skips_no_lookup_and_successful_body(calls):
    hook = Observer("gated", lambda _: pytest.fail("正常轨迹不应审查"), {"tool_results": calls}, "学习问题")
    try:
        hook.invoke(lambda *a, **k: AIMessage(content="直接回答"), None, [], stage="交互回答-工具判断1")
        assert hook.events == [{"status": "skipped_by_gate"}]
    finally:
        hook.close()


def test_precise_feedback_reports_failed_read_without_calling_observer():
    calls = []
    audit = {"tool_results": [{"tool_name": "read_material_lines", "args": {},
                               "result": '{"ok":false,"error":{"code":"line_out_of_range"}}'}]}
    hook = Observer("precise", lambda _: pytest.fail("程序组不应调用审查模型"), audit, "求原文")

    def main(model, messages, **kwargs):
        calls.append(messages)
        return AIMessage(content="原答复")

    hook.invoke(main, None, [], stage="交互回答-工具判断1")
    assert len(calls) == 2 and "尚未取得正文" in calls[1][-1].content


def test_named_sources_can_trigger_on_partial_coverage():
    call = {"tool_name": "read_material_lines", "args": {"relative_path": "a.md"},
            "result": '{"ok":true,"data":{"lines":[{"text":"第一份正文"}]}}'}
    hook = Observer("precise", None, {"tool_results": [call]}, "比较两份资料", expected_sources=("a.md", "b.md"))
    assert hook._evidence_gap()
    assert hook._snapshot()["requested_sources_without_body"] == ["b.md"]


@pytest.mark.parametrize("has_more", [True, False])
def test_catalog_only_reports_absence_for_complete_first_page(has_more):
    import json

    result = {"ok": True, "data": {"offset": 0, "has_more": has_more,
                                    "files": [{"relative_path": "a.md"}]}}
    entry = {"tool_name": "list_material_files", "executed": True, "result": json.dumps(result)}
    hook = Observer("catalog", None, {"tool_results": [entry]}, "核对两份资料")
    status = hook._catalog_status(["a.md", "b.md"])
    assert status["not_listed"] == ([] if has_more else ["b.md"])
    assert status["can_check"] == ([] if has_more else ["a.md"])


@pytest.mark.parametrize(("reply", "expected_calls"), [
    ("我只取得第一份正文，第二份尚未核实。", 1),
    ("两份材料我都读完了。", 2),
])
def test_catalog_guard_distinguishes_explicit_read_claim(reply, expected_calls):
    import json

    audit = {"tool_results": [
        {"tool_name": "list_material_files", "executed": True, "args": {},
         "result": json.dumps({"ok": True, "data": {"offset": 0, "has_more": False,
                      "files": [{"relative_path": path} for path in ("a.md", "b.md")]}})},
        {"tool_name": "read_material_lines", "executed": True, "args": {"relative_path": "a.md"},
         "result": '{"ok":true,"data":{"lines":[{"text":"正文"}]}}'},
    ]}
    hook = Observer("catalog_guard", None, audit, "对照两份资料", expected_sources=("a.md", "b.md"))
    calls = []
    hook.invoke(lambda *a, **k: calls.append(a) or AIMessage(content=reply), None, [],
                stage="交互回答-工具判断3")
    assert len(calls) == expected_calls


def test_async3_counts_executed_actions_and_only_injects_ready_result(monkeypatch):
    from concurrent.futures import Future

    future = Future()
    actions = []
    hook = Observer("async3", None, {"tool_results": actions}, "比较两份资料")
    snapshots = []
    monkeypatch.setattr(hook.pool, "submit", lambda review, snapshot: snapshots.append(snapshot) or future)
    requests = []

    def invoke():
        return hook.invoke(lambda model, messages, **kw:
                           requests.append(messages) or AIMessage(content="继续"),
                           None, [], stage="交互回答-工具判断")

    def add(executed=True, name="list_material_files", result='{"ok":true}'):
        actions.append({"executed": executed, "tool_name": name, "result": result, "args": {}})

    try:
        add()
        add(result='{"ok":false}')
        add(executed=False)
        invoke()
        assert not snapshots
        add()
        invoke()
        assert len(snapshots) == 1 and hook.events == [{"status": "scheduled", "at_action": 3}]
        invoke()
        assert requests == [[], [], []]
        future.set_result({"decision": "nudge", "feedback": "查阅第二份资料的正文"})
        invoke()
        assert "查阅第二份资料的正文" in requests[-1][-1].content
        invoke()
        assert requests[-1] == []
    finally:
        hook.close()


def test_async3_current_stale_rule_discards_nudge_after_successful_read(monkeypatch):
    from concurrent.futures import Future

    future = Future()
    actions = [{"executed": True, "tool_name": "list_material_files", "result": '{"ok":true}', "args": {}}] * 3
    hook = Observer("async3", None, {"tool_results": actions}, "比较两份资料")
    monkeypatch.setattr(hook.pool, "submit", lambda *args: future)

    def main(model, messages, **kwargs):
        assert messages == []
        return AIMessage(content="继续")

    try:
        hook.invoke(main, None, [], stage="交互回答-工具判断")
        future.set_result({"decision": "nudge", "feedback": "补查正文"})
        actions.append({"executed": True, "tool_name": "read_material_lines", "args": {},
                        "result": '{"ok":true,"data":{"lines":[{"text":"原文"}]}}'})
        hook.invoke(main, None, [], stage="交互回答-工具判断")
        assert hook.events[-1] == {"status": "stale_after_new_evidence", "at_action": 4}
    finally:
        hook.close()
