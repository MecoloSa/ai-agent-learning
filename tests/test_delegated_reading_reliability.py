import asyncio
import json
from types import SimpleNamespace

import pytest
from chromadb.errors import InternalError
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from pypdf.errors import PdfReadError

from ai_agent.runtime import delegated_reading as runtime
from tests.helpers import delegated_reading as graph

BOOK = graph.BOOKS[0]


def read_args(page=41):
    return {"relative_path": BOOK, "start_page": page, "end_page": page}


def call(name, identifier, args):
    return {"id": identifier, "name": name, "args": args}


def report(key):
    return {"findings": [{"statement": "核对得到的内容", "evidence_ids": [key]}], "gaps": []}


def fake_reader(*args):
    return {"pages": [{"page": args[-2], "text": "已取得的正文"}], "truncated": False}


def test_chroma_failure_does_not_cancel_other_branch(tmp_path, monkeypatch):
    @tool
    def search_material(queries: list[str]) -> str:
        """Search indexed evidence."""
        raise InternalError("unable to open database file")

    monkeypatch.setattr(runtime, "build_scoped_search", lambda *args: search_material)
    monkeypatch.setattr(runtime, "read_pdf_pages", fake_reader)
    trace = []

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                {"id": "search", "instruction": "检索资料", "sources": [BOOK]},
                {"id": "read", "instruction": "核对第41页", "sources": [BOOK]},
            ]}))
        if not isinstance(messages[-1], ToolMessage):
            request = (call("search_material", "search", {"queries": ["主题"]}) if stage == "search"
                       else call("read_pdf", "read", read_args()))
            return AIMessage(content="", tool_calls=[request])
        receipt = json.loads(messages[-1].content)
        if stage == "search":
            assert receipt["error"] == "InternalError"
            return AIMessage(content='{"findings":[],"gaps":["检索服务失败"]}')
        return AIMessage(content=json.dumps(report(receipt["evidence"][0]["evidence_id"])))

    workflow = graph.build_graph(tmp_path, {BOOK: None}, ask, trace, roomy=True,
                                 reuse_retrieval=True, collect_only=True)
    result = asyncio.run(workflow.ainvoke({"question": "核对资料", "delegation_goal": "分路取证"}))
    assert result["results"]["search"]["status"] == "unavailable"
    assert result["results"]["read"]["status"] == "done"
    assert result["delegated_report"]["evidence_catalog"]


@pytest.mark.parametrize("has_receipt", [True, False])
def test_reserved_report_phase_only_uses_confirmed_evidence(tmp_path, monkeypatch, has_receipt):
    monkeypatch.setattr(runtime, "read_pdf_pages", fake_reader)
    trace = []
    report_requests = []

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                {"id": "read", "instruction": "核对正文", "sources": [BOOK]},
            ]}))
        if not tools:
            evidence = json.loads(messages[-1].content)["evidence"]
            report_requests.append(evidence)
            assert [item["page"] for item in evidence] == [41]
            return AIMessage(content=json.dumps(report(evidence[0]["evidence_id"])))
        if has_receipt and len(messages) == 2:
            return AIMessage(content="", tool_calls=[call("read_pdf", "read", read_args())])
        await asyncio.sleep(10)

    workflow = graph.build_graph(tmp_path, {BOOK: {41}}, ask, trace, roomy=True,
                                 collect_only=True, turn_timeout=1.2, delivery_reserve=0.1,
                                 branch_report_reserve=0.7)
    result = asyncio.run(workflow.ainvoke({"question": "理解教材", "delegation_goal": "核对41页"}))
    assert result["results"]["read"]["status"] == ("done" if has_receipt else "unavailable")
    assert len(report_requests) == int(has_receipt)
    assert any(event["tool"] == "branch_deadline_closeout" for event in trace) == has_receipt


@pytest.mark.parametrize("scenario", ["submit", "continue", "reread", "missing", "dependency"])
@pytest.mark.parametrize("threshold", [2, 3, 4])
def test_acquisition_notice_preserves_worker_choices(tmp_path, monkeypatch, scenario, threshold):
    def read(*args):
        if scenario == "missing":
            raise OSError("资料暂不可读取")
        return fake_reader(*args)

    monkeypatch.setattr(runtime, "read_pdf_pages", read)
    trace = []
    returned_ids = {}
    noticed = []
    checkpoint = threshold + 1  # 首次读到新正文，之后才开始累计重复。

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            tasks = [{"id": "A", "instruction": "核对指定资料", "sources": [BOOK], "depends_on": []}]
            if scenario == "dependency":
                tasks.append({"id": "B", "instruction": "仅按上游证据出一道练习",
                              "sources": [], "depends_on": ["A"]})
            return AIMessage(content=json.dumps({"tasks": tasks}))
        if stage == "deliver":
            return AIMessage(content="已完成有证据支持的部分。")
        if stage == "B":
            assert all(message.content != graph.ACQUISITION_NOTICE for message in messages)
            evidence = json.loads(messages[1].content)["upstream_evidence"][0]
            return AIMessage(content=json.dumps(report(evidence["evidence_id"])))

        receipts = [message for message in messages if isinstance(message, ToolMessage)]
        count = len(receipts)
        if count:
            items = json.loads(receipts[-1].content).get("evidence", [])
            for item in items:
                returned_ids[item["page"]] = item["evidence_id"]
        if messages[-1].content == graph.ACQUISITION_NOTICE:
            noticed.append(count)
            assert {"read_pdf", "recall_evidence", "submit_report"} <= {tool.name for tool in tools}
        if scenario == "dependency" and count == 1:
            return AIMessage(content=json.dumps(report(returned_ids[41])))
        if scenario == "continue" and count == checkpoint:
            return AIMessage(content="", tool_calls=[call("read_pdf", "new", read_args(42))])
        if scenario == "continue" and count == checkpoint + 1:
            assert messages[-1].content != graph.ACQUISITION_NOTICE
            return AIMessage(content=json.dumps(report(returned_ids[42])))
        if count == checkpoint:
            body = ({"findings": [], "gaps": ["资料不可读"]} if scenario == "missing"
                    else report(returned_ids[41]))
            return AIMessage(content=json.dumps(body))
        if scenario == "reread" and count:
            request = call("recall_evidence", f"recall-{count}", {"evidence_id": returned_ids[41]})
        else:
            request = call("read_pdf", f"read-{count}", read_args())
        return AIMessage(content="", tool_calls=[request])

    result = asyncio.run(graph.build_graph(
        tmp_path, {BOOK: {41, 42}}, ask, trace, roomy=True, project=True, acquisition_notice=True,
        notice_after=threshold,
    ).ainvoke({"question": "核对资料；需要时继续查找"}))
    assert noticed == ([checkpoint] if scenario in {"submit", "continue"} else [])
    assert result["results"]["A"]["status"] == ("unavailable" if scenario == "missing" else "done")
    assert result["delivery_status"] == "completed"


def test_equal_length_receipts_keep_separate_content(tmp_path, monkeypatch):
    texts = iter(("版本甲", "版本乙"))
    monkeypatch.setattr(runtime, "read_pdf_pages", lambda *args: {
        "pages": [{"page": 41, "text": next(texts)}], "truncated": False,
    })
    store = {}
    read, = graph.build_tools(tmp_path, {BOOK: {41}}, store, set(), roomy=True)
    a, b = [json.loads(read.invoke(read_args()))["evidence"][0]["evidence_id"] for _ in range(2)]
    assert a != b
    assert (store[a]["text"], store[b]["text"]) == ("版本甲", "版本乙")


def test_index_refreshes_after_pdf_replacement(tmp_path, monkeypatch):
    path = tmp_path / "book.pdf"
    path.write_bytes(b"alpha")
    monkeypatch.setattr(runtime, "PdfReader", lambda stream: SimpleNamespace(
        pages=[SimpleNamespace(extract_text=lambda: stream.read().decode())]))
    assert graph.index_pdf(path) == ("alpha",)
    path.write_bytes(b"new beta")
    assert graph.index_pdf(path) == ("new beta",)


def test_worker_recovery_with_only_upstream_text(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "read_pdf_pages", fake_reader)

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                {"id": "A", "instruction": "读41页", "sources": [BOOK], "depends_on": []},
                {"id": "B", "instruction": "解释A已经得到的依据", "sources": [], "depends_on": ["A"]},
            ]}))
        if stage == "A":
            if len(messages) == 2:
                return AIMessage(content="", tool_calls=[call("read_pdf", "read", read_args())])
            item = json.loads(messages[-1].content)["evidence"][0]
            return AIMessage(content=json.dumps(report(item["evidence_id"])))
        if stage == "B":
            if tools:
                return AIMessage(content='{"findings":[],"gaps":["待提交"]}')
            (item,) = json.loads(messages[-1].content)["evidence"]
            assert item["text"] == "已取得的正文"
            return AIMessage(content=json.dumps(report(item["evidence_id"])))
        return AIMessage(content="完成")

    result = asyncio.run(graph.build_graph(tmp_path, {BOOK: {41}}, ask, [], roomy=True)
                         .ainvoke({"question": "读41页并解释"}))
    assert result["results"]["B"]["status"] == "done"
    assert "evidence_store" not in result


def test_corrupt_locator_is_reported_without_losing_independent_read(tmp_path, monkeypatch):
    bad = "broken.pdf"
    monkeypatch.setattr(runtime, "read_pdf_pages", fake_reader)

    def broken(_path):
        raise PdfReadError("corrupt PDF")

    monkeypatch.setattr(runtime, "index_pdf", broken)

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                {"id": "A", "instruction": "搜索损坏资料", "sources": [bad], "depends_on": []},
                {"id": "B", "instruction": "读41页", "sources": [BOOK], "depends_on": []},
            ]}))
        if stage == "A":
            if len(messages) == 2:
                return AIMessage(content="", tool_calls=[call("locate_pdf", "locate", {
                    "relative_path": bad, "terms": ["Observation"]})])
            assert json.loads(messages[-1].content)["error"] == "PdfReadError"
            return AIMessage(content='{"findings":[],"gaps":["损坏PDF不可检索"]}')
        if stage == "B":
            if len(messages) == 2:
                return AIMessage(content="", tool_calls=[call("read_pdf", "read", read_args())])
            item = json.loads(messages[-1].content)["evidence"][0]
            return AIMessage(content=json.dumps(report(item["evidence_id"])))
        return AIMessage(content="保留成功分支，报告另一份资料损坏")

    result = asyncio.run(graph.build_graph(tmp_path, {BOOK: {41}, bad: None}, ask, [], roomy=True)
                         .ainvoke({"question": "读一本并搜索另一本"}))
    assert result["results"]["B"]["status"] == "done"
    assert result["results"]["A"]["status"] == "unavailable"


def test_mixed_submit_has_recovery_instruction(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "read_pdf_pages", fake_reader)

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                {"id": "A", "instruction": "读41页", "sources": [BOOK], "depends_on": []},
            ]}))
        if stage == "A" and len(messages) == 2:
            return AIMessage(content="", tool_calls=[
                call("submit_report", "submit", {"findings": [], "gaps": []}),
                call("read_pdf", "read", read_args()),
            ])
        if stage == "A":
            assert messages[-2].tool_call_id == "submit"
            assert json.loads(messages[-2].content)["error"] == "submit_after_tool_results"
            item = json.loads(messages[-1].content)["evidence"][0]
            return AIMessage(content="", tool_calls=[call("submit_report", "final", report(item["evidence_id"]))])
        return AIMessage(content="完成")

    result = asyncio.run(graph.build_graph(tmp_path, {BOOK: {41}}, ask, [], roomy=True)
                         .ainvoke({"question": "读41页"}))
    assert result["results"]["A"]["status"] == "done"


@pytest.mark.parametrize("weather", ["missing", "unverified", "verified", "no_findings"])
def test_named_dependency_filters_start_without_losing_partial_report(tmp_path, monkeypatch, weather):
    monkeypatch.setattr(runtime, "read_pdf_pages", fake_reader)
    started = set()

    def task(name, deps=(), required=None):
        item = {"id": name, "instruction": name, "sources": [BOOK], "depends_on": list(deps)}
        if required is not None:
            item["required_artifacts"] = required
        return item

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                task("pair", ("read",), {"read": ["time_id", "weather_id"]}),
                task("read"),
                task("explain_available", ("read",)),
                task("independent"),
            ]}))
        if stage == "read":
            payload = json.loads(messages[1].content)
            assert payload["requested_artifacts"] == ["time_id", "weather_id"]
            if len(messages) == 2 and tools:
                return AIMessage(content="", tool_calls=[call("read_pdf", "read", read_args())])
            item = (json.loads(messages[-1].content)["evidence"])[0]
            findings = []
            if weather != "no_findings":
                findings.append({"statement": "时间 ID 是 call_abc123", "evidence_ids": [item["evidence_id"]],
                                 "artifact": "time_id"})
            if weather in {"verified", "unverified"}:
                findings.append({"statement": "天气 ID 是 call_def456", "artifact": "weather_id",
                                 "evidence_ids": [item["evidence_id"] if weather == "verified" else "forged"]})
            gaps = ["额外背景尚未核对"] if weather == "verified" else ["天气 ID 未取得"]
            return AIMessage(content=json.dumps({"findings": findings, "gaps": gaps}))
        if stage == "deliver":
            payload = json.loads(messages[1].content)
            if weather != "verified":
                assert "weather_id" in payload["reports"]["pair"]["missing_artifacts"]["read"]
            return AIMessage(content="有据部分已交付")
        payload = json.loads(messages[1].content)
        started.add(stage)
        if stage == "explain_available":
            upstream = payload["upstream_reports"]["read"]
            assert upstream["gaps"] == (["额外背景尚未核对"] if weather == "verified" else ["天气 ID 未取得"])
            assert len(payload["upstream_evidence"]) == (0 if weather == "no_findings" else 1)
        if stage == "pair":
            assert {f.get("artifact") for f in payload["upstream_reports"]["read"]["findings"]} == {
                "time_id", "weather_id",
            }
        if payload["upstream_evidence"]:
            return AIMessage(content=json.dumps(report(payload["upstream_evidence"][0]["evidence_id"])))
        return AIMessage(content='{"findings":[],"gaps":[]}')

    result = asyncio.run(graph.build_graph(tmp_path, {BOOK: {41}}, ask, [], roomy=True)
                         .ainvoke({"question": "取出时间与天气 ID，核对配对，另解释已找到的内容"}))
    assert {"independent", "explain_available"} <= started
    assert ("pair" in started) == (weather == "verified")
    if weather != "verified":
        assert result["results"]["pair"]["status"] == "blocked"


def test_required_artifacts_must_reference_direct_dependencies():
    task = graph.Assignment(id="pair", instruction="核对", sources=[BOOK], depends_on=[],
                            required_artifacts={"read": ["weather_id"]})
    with pytest.raises(ValueError, match="direct dependencies"):
        graph.validate([task], {BOOK: {41}})


@pytest.mark.parametrize("mode", ["corrected", "repeated", "temporary_error"])
def test_worker_reuses_terminal_pdf_failure_and_preserves_possible_recovery(tmp_path, monkeypatch, mode):
    reads = []
    trace = []
    initial_page = 41 if mode == "temporary_error" else 9999

    def reader(root, path, first, last):
        reads.append(first)
        if mode == "temporary_error" and len(reads) == 1:
            raise OSError("temporary reader failure")
        if first == 9999:
            raise graph.PdfPageOutOfBounds(first, 720)
        return fake_reader(root, path, first, last)

    monkeypatch.setattr(runtime, "read_pdf_pages", reader)

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                {"id": "A", "instruction": "核对资料", "sources": [BOOK], "depends_on": []},
            ]}))
        if stage == "deliver":
            return AIMessage(content="交付")
        if len(messages) == 2:
            return AIMessage(content="", tool_calls=[call("read_pdf", "first", read_args(initial_page))])
        if len(messages) == 4:
            reads_to_try = [call("read_pdf", "repeat", read_args(initial_page))]
            if mode == "corrected":
                reads_to_try.append(call("read_pdf", "corrected", read_args(41)))
            return AIMessage(content="", tool_calls=reads_to_try)
        assert mode != "repeated"
        evidence = json.loads(messages[-1].content)["evidence"][0]
        return AIMessage(content=json.dumps(report(evidence["evidence_id"])))

    result = asyncio.run(graph.build_graph(tmp_path, {BOOK: None}, ask, trace, roomy=True)
                         .ainvoke({"question": "核对 PDF 页码和证据"}))
    assert reads == {"corrected": [9999, 41], "repeated": [9999], "temporary_error": [41, 41]}[mode]
    assert trace[1].get("reused_terminal_receipt", False) is (mode != "temporary_error")
    assert result["results"]["A"]["status"] == ("unavailable" if mode == "repeated" else "done")


def test_delivery_failure_does_not_erase_finished_branches(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "read_pdf_pages", fake_reader)

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                {"id": key, "instruction": "核对", "sources": [BOOK], "depends_on": []}
                for key in ("A", "B")
            ]}))
        if stage == "deliver":
            return AIMessage(content="")
        if len(messages) == 2:
            return AIMessage(content="", tool_calls=[call("read_pdf", stage, read_args())])
        evidence = json.loads(messages[-1].content)["evidence"][0]
        return AIMessage(content=json.dumps(report(evidence["evidence_id"])))

    result = asyncio.run(graph.build_graph(tmp_path, {BOOK: {41}}, ask, [], roomy=True)
                         .ainvoke({"question": "分别核实两项，再做汇总"}))
    assert result["delivery_status"] == "failed"
    assert result["delivery_error"]["type"] == "DeliveryFailure"
    assert "answer" not in result
    assert {key: result["results"][key]["status"] for key in ("A", "B")} == {"A": "done", "B": "done"}
    assert all(result["results"][key]["findings"][0]["evidence_ids"] for key in ("A", "B"))


@pytest.mark.parametrize("timeout_option", ["worker_timeout", "model_request_timeout"])
def test_worker_timeout_passes_read_receipts_to_final_synthesis(tmp_path, monkeypatch, timeout_option):
    monkeypatch.setattr(runtime, "read_pdf_pages", fake_reader)
    acquired = {}
    trace = []

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                {"id": key, "instruction": "核对各自的资料页", "sources": [BOOK], "depends_on": []}
                for key in ("slow", "ready")
            ]}))
        if stage == "deliver":
            handoff = json.loads(messages[1].content)
            assert handoff["reports"]["slow"]["status"] == "unavailable"
            assert set(handoff["evidence_catalog"]) == set(acquired.values())
            if len(messages) == 2:
                return AIMessage(content="", tool_calls=[call(
                    "read_delivery_evidence", "recover", {"evidence_id": acquired["slow"]},
                )])
            assert json.loads(messages[-1].content)["page"] == 41
            return AIMessage(content="保留两页资料，慢分支报告未完成。")
        if len(messages) == 2:
            return AIMessage(content="", tool_calls=[call("read_pdf", stage, read_args(41 if stage == "slow" else 36))])
        evidence_id = json.loads(messages[-1].content)["evidence"][0]["evidence_id"]
        acquired[stage] = evidence_id
        if stage == "slow":
            await asyncio.sleep(5)
        return AIMessage(content=json.dumps(report(evidence_id)))

    result = asyncio.run(graph.build_graph(tmp_path, {BOOK: {36, 41}}, ask, trace, roomy=True,
                                          **{timeout_option: 0.25})
                         .ainvoke({"question": "分别读第36、41页；取证未全部完成也保留已有证据"}))
    assert result["results"]["slow"]["evidence_ids"] == [acquired["slow"]]
    assert result["results"]["ready"]["status"] == "done"
    assert result["delivery_status"] == "completed"
    assert [item["tool"] for item in trace].count("read_pdf") == 2
    assert set(result["worker_seconds"]) == {"slow", "ready"}


@pytest.mark.parametrize("slow_stage", ["planner", "worker", "deliver"])
@pytest.mark.parametrize("timeout_options", [
    {"turn_timeout": 0.65, "delivery_reserve": 0.25},
    {"model_request_timeout": 0.4},
])
def test_deadline_preserves_available_results(tmp_path, monkeypatch, slow_stage, timeout_options):
    monkeypatch.setattr(runtime, "read_pdf_pages", fake_reader)
    visited = []

    async def ask(role, stage, messages, tools=()):
        visited.append(stage)
        if stage == "planner":
            if slow_stage == stage:
                await asyncio.sleep(5)
            return AIMessage(content=json.dumps({"tasks": [
                {"id": "worker", "instruction": "核对第41页", "sources": [BOOK], "depends_on": []},
            ]}))
        if stage == "deliver":
            if slow_stage == stage:
                await asyncio.sleep(5)
            return AIMessage(content="已取得第41页，按已有依据回答。")
        if len(messages) == 2:
            return AIMessage(content="", tool_calls=[call("read_pdf", "read", read_args())])
        key = json.loads(messages[-1].content)["evidence"][0]["evidence_id"]
        if slow_stage == stage:
            await asyncio.sleep(5)
        return AIMessage(content=json.dumps(report(key)))

    result = asyncio.run(graph.build_graph(tmp_path, {BOOK: {41}}, ask, [], roomy=True,
                         **timeout_options).ainvoke({"question": "核对第41页"}))
    if slow_stage == "planner":
        assert visited == ["planner"]
        assert result["results"] == {} and result["delivery_status"] == "failed"
    else:
        assert result["results"]["worker"]["evidence_ids"]
        assert result["delivery_status"] == ("completed" if slow_stage == "worker" else "failed")
        if slow_stage == "deliver" and "model_request_timeout" in timeout_options:
            assert result["delivery_error"]["type"] == "ModelRequestTimeout"


def test_roomy_cli_sets_only_the_per_request_deadline(tmp_path, monkeypatch):
    seen = []

    async def run(*args, **kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(graph, "run", run)
    asyncio.run(graph.main(tmp_path, tmp_path, ["--roomy", "single"]))
    assert seen[0]["model_request_timeout"] == 60
    assert "worker_timeout" not in seen[0] and "turn_timeout" not in seen[0]


def test_serial_tasks_can_outlast_one_request_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "read_pdf_pages", fake_reader)
    visited = []

    async def ask(role, stage, messages, tools=()):
        visited.append(stage)
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                {"id": "A", "instruction": "读第41页", "sources": [BOOK], "depends_on": []},
                {"id": "B", "instruction": "根据上游说明", "sources": [], "depends_on": ["A"]},
            ]}))
        if stage == "deliver":
            return AIMessage(content="两路完成")
        await asyncio.sleep(0.12)
        if stage == "B":
            item = json.loads(messages[1].content)["upstream_evidence"][0]
        elif len(messages) == 2:
            return AIMessage(content="", tool_calls=[call("read_pdf", "read", read_args())])
        else:
            item = json.loads(messages[-1].content)["evidence"][0]
        return AIMessage(content=json.dumps(report(item["evidence_id"])))

    result = asyncio.run(graph.build_graph(tmp_path, {BOOK: {41}}, ask, [], roomy=True,
                                          model_request_timeout=0.3).ainvoke({"question": "先读后解释"}))
    assert visited == ["planner", "A", "A", "B", "deliver"]
    assert sum(result["worker_seconds"].values()) > 0.3
    assert result["delivery_status"] == "completed"


@pytest.mark.parametrize("returns_late", [False, True])
def test_cancelling_turn_does_not_start_delivery(tmp_path, returns_late):
    async def run():
        worker_started, worker_cancelled = asyncio.Event(), asyncio.Event()
        visited = []

        async def ask(role, stage, messages, tools=()):
            visited.append(stage)
            if stage == "planner":
                return AIMessage(content=json.dumps({"tasks": [
                    {"id": "worker", "instruction": "核对教材", "sources": [BOOK], "depends_on": []},
                ]}))
            worker_started.set()
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                if returns_late:
                    # 模拟在收到取消后仍返回的延后模型结果。
                    return AIMessage(content='{"findings":[],"gaps":["late"]}')
                raise
            finally:
                worker_cancelled.set()

        workflow = graph.build_graph(tmp_path, {BOOK: {41}}, ask, [], roomy=True)
        request = asyncio.create_task(workflow.ainvoke({"question": "核对教材"}))
        await asyncio.wait_for(worker_started.wait(), 2)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert worker_cancelled.is_set() and visited == ["planner", "worker"]

    asyncio.run(run())


def test_saved_evidence_finding_is_delivered_after_report_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "read_pdf_pages", fake_reader)
    trace = []

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                {"id": "W", "instruction": "核对第41页", "sources": [BOOK], "depends_on": []},
            ]}))
        if stage == "deliver":
            payload = json.loads(messages[1].content)
            (finding,) = payload["reports"]["W"]["findings"]
            assert finding["statement"] == "已核对第41页"
            assert payload["already_read"][finding["evidence_ids"][0]] == "已取得的正文"
            return AIMessage(content="按已保存的发现核对。")
        if len(messages) == 2:
            return AIMessage(content="", tool_calls=[call("read_pdf", "read", read_args())])
        if len(messages) == 4:
            key = json.loads(messages[-1].content)["evidence"][0]["evidence_id"]
            item = {"statement": "已核对第41页", "evidence_ids": [key]}
            return AIMessage(content="", tool_calls=[call(
                "save_findings", "save", {"findings": [item, item, {"statement": "无依据", "evidence_ids": ["fake"]}]},
            )])
        await asyncio.sleep(5)

    result = asyncio.run(graph.build_graph(tmp_path, {BOOK: {41}}, ask, trace, roomy=True,
                         worker_timeout=0.3, progressive_reports=True).ainvoke({"question": "核对第41页"}))
    assert result["results"]["W"]["status"] == "unavailable"
    assert result["delivery_status"] == "completed"
    assert [entry["result"] for entry in trace if entry["tool"] == "save_findings"] == [{"saved_count": 1}]
