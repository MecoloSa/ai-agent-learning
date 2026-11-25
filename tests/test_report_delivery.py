import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from tests.helpers.report_delivery import deliver_reports


def test_report_delivery_reads_only_catalogued_evidence_and_does_not_duplicate_text():
    reports = {
        "A": {"status": "done", "findings": [{"statement": "A 的答案", "evidence_ids": ["a"]}],
              "evidence_ids": ["a", "unused"]},
        "B": {"status": "unavailable", "reason": "尚未提交", "evidence_ids": ["b"]},
    }
    store = {key: {"source": "test.pdf", "page": page, "text": f"{key} 原文"}
             for page, key in enumerate(("a", "b", "unused"), 1)}
    trace = []
    delivered = set()
    requests = 0

    async def ask(role, stage, messages, tools):
        nonlocal requests
        requests += 1
        assert role == "standard" and stage == "deliver"
        if requests == 1:
            payload = json.loads(messages[1].content)
            assert payload["reports"]["A"]["findings"][0]["statement"] == "A 的答案"
            assert set(payload["evidence_catalog"]) == {"a", "b"}
            assert "原文" not in messages[1].content
            return AIMessage(content="", tool_calls=[
                {"name": "read_delivery_evidence", "id": str(i), "args": {"evidence_id": key}}
                for i, key in enumerate(("a", "a", "unused", "b"))
            ])
        returned = [m for m in messages if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in returned] == ["0", "1", "2", "3"]
        assert json.loads(returned[0].content)["text"] == "a 原文"
        assert json.loads(returned[1].content)["status"] == "already_returned"
        assert json.loads(returned[2].content)["error"] == "unknown_evidence_id"
        assert json.loads(returned[3].content)["text"] == "b 原文"
        return AIMessage(content="交付")

    result = asyncio.run(deliver_reports("问题", reports, store, ask, trace, "汇总",
                                        delivered_evidence_ids=delivered))
    assert result.content == "交付"
    assert len(trace) == 4
    assert delivered == {"a", "b"}


def test_report_delivery_can_recover_omitted_receipts_without_worker_gap():
    reports = {"read": {
        "status": "done",
        "findings": [{"statement": "已解释 36 页", "evidence_ids": ["p36"]}],
        "gaps": [],
        "evidence_ids": ["p36", "p41", "p41"],
    }}
    store = {f"p{page}": {"source": "test.pdf", "page": page, "text": f"原文 {page}"}
             for page in (36, 41, 90)}
    trace = []

    async def ask(role, stage, messages, tools):
        if not trace:
            payload = json.loads(messages[1].content)
            assert set(payload["evidence_catalog"]) == {"p36"}
            assert "原文 41" not in messages[1].content
            assert {tool.name for tool in tools} == {"read_delivery_evidence", "read_additional_receipt"}
            return AIMessage(content="", tool_calls=[
                {"name": "read_additional_receipt", "id": str(index),
                 "args": {"source": "test.pdf", "page": page}}
                for index, page in enumerate((41, 41, 90))
            ])
        assert [item["result"] for item in trace] == [
            {"evidence": [store["p41"]]},
            {"status": "already_returned"},
            {"error": "not_in_read_receipts"},
        ]
        assert [m.tool_call_id for m in messages if isinstance(m, ToolMessage)] == ["0", "1", "2"]
        return AIMessage(content="已核对第41页")

    delivered = set()
    result = asyncio.run(deliver_reports("核对第41页", reports, store, ask, trace, "汇总",
                                        delivered_evidence_ids=delivered))
    assert result.content == "已核对第41页"
    assert delivered == {"p41"}


def evidence_request(*evidence_ids, prefix):
    return AIMessage(content="", tool_calls=[
        {"name": "read_delivery_evidence", "id": f"{prefix}-{index}",
         "args": {"evidence_id": evidence_id}}
        for index, evidence_id in enumerate(evidence_ids)
    ])


def run_delivery_script(replies):
    reports = {"read": {"status": "done", "findings": [
        {"statement": "已核实两页", "evidence_ids": ["p36", "p41"]},
    ]}}
    store = {f"p{page}": {"source": "test.pdf", "page": page, "text": f"正文 {page}"}
             for page in (36, 41)}
    requests, trace = [], []
    scripted = iter(replies)

    async def ask(role, stage, messages, tools):
        requests.append((list(messages), [tool.name for tool in tools]))
        return next(scripted)

    async def run():
        return await deliver_reports("核实 PDF 第36、41页并给出处", reports, store, ask, trace, "汇总")

    return run, requests, trace


def test_delivery_keeps_tools_when_the_model_corrects_a_failed_request():
    run, requests, trace = run_delivery_script([
        evidence_request("missing", prefix="wrong"),
        evidence_request("p36", prefix="first"),
        evidence_request("missing", prefix="try-again"),
        evidence_request("p41", prefix="second"),
        AIMessage(content="已核实第36、41页"),
    ])
    assert asyncio.run(run()).text == "已核实第36、41页"
    assert all(tools for _, tools in requests)
    assert [item["result"].get("error") for item in trace] == [
        "unknown_evidence_id", None, "unknown_evidence_id", None,
    ]


@pytest.mark.parametrize("closing_reply, error", [
    (AIMessage(content="已按现有依据收尾"), None),
    (AIMessage(content=" \n"), "汇总未生成可交付正文"),
    (evidence_request("p36", prefix="late"), "汇总收尾未生成正文"),
])
def test_delivery_closes_after_two_identical_batches_without_new_text(closing_reply, error):
    run, requests, trace = run_delivery_script([
        evidence_request("missing", "other", prefix="first"),
        evidence_request("other", "missing", prefix="second"),
        closing_reply,
    ])
    if error:
        with pytest.raises(ValueError, match=error):
            asyncio.run(run())
    else:
        assert asyncio.run(run()).text == "已按现有依据收尾"
    assert [bool(tools) for _, tools in requests] == [True, True, False]
    assert len(trace) == 4
    assert [m.tool_call_id for m in requests[-1][0] if isinstance(m, ToolMessage)] == [
        "first-0", "first-1", "second-0", "second-1",
    ]


@pytest.mark.parametrize("second_text", ["已交付", " \n"])
def test_delivery_retries_an_empty_reply_once(second_text):
    run, requests, trace = run_delivery_script([
        AIMessage(content=""),
        AIMessage(content=second_text),
    ])
    if second_text.strip():
        assert asyncio.run(run()).text == second_text
    else:
        with pytest.raises(ValueError, match="汇总未生成可交付正文"):
            asyncio.run(run())
    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert not trace


def test_delivery_closes_when_failed_requests_alternate():
    run, requests, trace = run_delivery_script([
        evidence_request("missing-a", prefix="first"),
        evidence_request("missing-b", prefix="second"),
        evidence_request("missing-a", prefix="third"),
        AIMessage(content="两条证据均未找到"),
    ])
    assert asyncio.run(run()).text == "两条证据均未找到"
    assert [bool(tools) for _, tools in requests] == [True, True, True, False]
    assert len(trace) == 3
