import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from time import sleep

from langchain_core.messages import AIMessage, ToolMessage
import pytest

from ai_agent.runtime import delegated_reading as probe
from tests.helpers.langgraph_reading_compare import BOOKS
from tests.helpers.report_delivery import deliver_reports
from tests.helpers.shared_reading_evidence import CitationArchive, SharedPdfReads


def call(name, identifier, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "id": identifier, "args": args}])


@pytest.mark.parametrize("shared_reads", [False, True])
def test_two_graph_workers_read_real_page_and_archive_only_the_served_citation(tmp_path, monkeypatch, shared_reads):
    root = Path("src/ai_agent/eval/ai_agent_book").resolve()
    book = BOOKS[0]
    real_reader = probe.read_pdf_pages
    reads = []

    def count_reads(*args):
        reads.append(args[2:])
        return real_reader(*args)

    monkeypatch.setattr(probe, "read_pdf_pages", count_reads)
    archive = CitationArchive(tmp_path / "citations.sqlite3", "session-a", "turn-1")
    citations = {}

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                {"id": task, "instruction": "读第44页并核对工具结果如何回填", "sources": [book], "depends_on": []}
                for task in ("A", "B")
            ]}))
        if stage == "deliver":
            if len(messages) == 2:
                (key,) = json.loads(messages[1].content)["evidence_catalog"]
                return call("read_delivery_evidence", "cite", evidence_id=key)
            evidence = json.loads(messages[-1].content)
            citations.update(evidence)
            assert "tool_call_id" in evidence["text"]
            return AIMessage(content=f"回填调用 ID [evidence={evidence['evidence_id']}] [evidence=forged]")
        if len(messages) == 2:
            return call("read_pdf", f"read-{stage}", relative_path=book, start_page=44, end_page=44)
        item = json.loads(messages[-1].content)["evidence"][0]
        return AIMessage(content=json.dumps({
            "findings": [{"statement": "回填工具调用 ID", "evidence_ids": [item["evidence_id"]]}], "gaps": [],
        }))

    trace = []
    result = asyncio.run(probe.build_graph(
        root, {book: {44}}, ask, trace, roomy=True, shared_pdf_reads=shared_reads, citation_archive=archive,
    ).ainvoke({"question": "两路都核对第44页的工具结果回填，再引用原文。"}))
    assert result["delivery_status"] == "completed"
    assert all(task["status"] == "done" for task in result["results"].values())
    expected = [(44, 44)] * (1 if shared_reads else 2)
    assert reads == expected
    assert sum(bool(row["result"].get("reused_shared_read")) for row in trace) == int(shared_reads)
    assert "【资料引用未核实】" in result["answer"]

    next_turn = CitationArchive(archive.path, "session-a", "turn-2")

    async def follow_up(role, stage, messages, tools=()):
        if not isinstance(messages[-1], ToolMessage):
            return call("read_previous_citations", "previous", source=book, page=44)
        (item,) = json.loads(messages[-1].content)["historical_citations"]
        assert item["turn"] == "turn-1"
        assert item["text"] == citations["text"]
        return AIMessage(content=f"已按旧引用核对原文 [evidence={item['evidence_id']}]")

    reply = asyncio.run(deliver_reports(
        "你上一轮引用的第44页原文是什么？", {}, {}, follow_up, [], "", citation_archive=next_turn,
        citation_scope={book: {44}},
    ))
    assert reply.text == f"已按旧引用核对原文 [source={book}, page=44]（历史引用）"
    assert reads == expected
    other = CitationArchive(archive.path, "session-b", "turn-2")
    assert json.loads(other.bind_read({book: {44}}).invoke({"source": book, "page": 44})) == {
        "historical_citations": [],
    }
    assert json.loads(next_turn.bind_read({book: {43}}).invoke({"source": book, "page": 44})) == {
        "error": "source_scope",
    }


def test_lru_cache_eviction_and_document_replacement_keep_original_citation(tmp_path):
    book = tmp_path / "book.pdf"
    book.write_text("原版正文", encoding="utf8")
    archive = CitationArchive(tmp_path / "citations.sqlite3", "session", "initial")
    reads = []

    def read(page):
        def fetch():
            reads.append(page)
            item = {"source": book.name, "page": page, "text": book.read_text(encoding="utf8")}
            return json.dumps({"evidence": [item]}), False
        return fetch

    first_raw, _ = read(1)()
    archive.save("p1-old", json.loads(first_raw)["evidence"][0])
    pool = SharedPdfReads(max_chars=len(first_raw))
    for page in (1, 2, 1):
        assert not pool.read(book, book.name, page, page, read(page))[1]
    assert reads == [1, 1, 2, 1]
    assert pool.read(book, book.name, 1, 1, read(1))[1]
    book.write_text("这是新版正文，文件长度也已改变", encoding="utf8")
    new, reused = pool.read(book, book.name, 1, 1, read(1))
    assert not reused and "新版" in json.loads(new)["evidence"][0]["text"]
    (previous,) = json.loads(CitationArchive(archive.path, "session", "next").bind_read(
        {book.name: None},
    ).invoke({"source": book.name, "page": 1}))["historical_citations"]
    assert previous["text"] == "原版正文"


def test_concurrent_same_range_shares_one_read(tmp_path):
    path = tmp_path / "book.pdf"
    path.touch()
    cache = SharedPdfReads()
    start = Barrier(2)
    reads = []

    def invoke():
        def fetch():
            reads.append(1)
            sleep(0.03)
            return json.dumps({"evidence": [{"page": 44, "text": "正文"}]}), False
        start.wait(timeout=2)
        return cache.read(path, path.name, 44, 44, fetch)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: invoke(), range(2)))
    assert reads == [1]
    assert sorted(reused for _, reused in outcomes) == [False, True]


def test_cached_page_cannot_expand_another_workers_scope(tmp_path, monkeypatch):
    (tmp_path / "book.pdf").touch()
    reads = []

    def read(root, path, start, end):
        reads.append(start)
        return {"pages": [{"page": start, "text": f"第{start}页"}], "truncated": start == 45}

    monkeypatch.setattr(probe, "read_pdf_pages", read)
    pool, store = SharedPdfReads(), {}
    (a,) = probe.build_tools(tmp_path, {"book.pdf": {44}}, store, set(), shared_reads=pool)
    (b,) = probe.build_tools(tmp_path, {"book.pdf": {43, 45}}, store, set(), shared_reads=pool)

    def invoke(tool, page):
        return json.loads(tool.invoke({"relative_path": "book.pdf", "start_page": page, "end_page": page}))

    invoke(a, 44)
    assert invoke(b, 44) == {"error": "page_scope"}
    assert invoke(b, 43)["evidence"][0]["page"] == 43
    assert invoke(b, 45)["truncated"] and invoke(b, 45)["truncated"]
    assert reads == [44, 43, 45, 45]
