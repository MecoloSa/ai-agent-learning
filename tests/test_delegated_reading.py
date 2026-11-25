import asyncio
import json
from dataclasses import replace
import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_agent.runtime import delegated_reading as runtime
from tests.helpers import delegated_reading as probe
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.history_session import ArchiveCapture
from ai_agent.persistence.memory import ConversationMemory


@pytest.mark.parametrize("roomy", [False, True])
def test_direct_route_preserves_history_without_binding_tools(tmp_path, roomy):
    memory = ConversationMemory("", strategy="full")
    memory.add_user("我的笔记是：短期记忆保留最近对话。")
    stages = []

    async def ask(role, stage, messages, tools=()):
        stages.append(stage)
        assert not tools
        assert any("短期记忆" in message.content for message in messages)
        if stage == "planner":
            return AIMessage(content='{"mode":"direct","tasks":[]}')
        assert messages[-1].content == "改成一句口语。"
        return AIMessage(content="短期记忆就是先留住最近聊过的内容。")

    result = asyncio.run(probe.build_graph(tmp_path, {}, ask, [], roomy=roomy, conversation_memory=memory)
                         .ainvoke({"question": "改成一句口语。"}))
    assert stages == ["planner", "deliver"]
    assert result["tasks"] == [] and result["results"] == {}
    assert result["delivery_status"] == "completed"


def test_named_generated_deliverable_is_handed_off_without_evidence(tmp_path):
    book = probe.BOOKS[0]
    seen = []

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [
                {"id": "draft", "instruction": "拟口头练习", "sources": [book], "generates_content": True},
                {"id": "review", "instruction": "审核题面", "sources": [], "depends_on": ["draft"],
                 "required_artifacts": {"draft": ["quiz"]}, "generates_content": True},
            ]}, ensure_ascii=False))
        payload = json.loads(messages[-1].content)
        if stage == "draft":
            submit = next(tool for tool in tools if tool.name == "submit_report")
            report = submit.invoke({"findings": [], "gaps": [], "deliverables": [
                {"content": "这条消息应该是什么角色？", "artifact": "quiz"},
            ]})
            return AIMessage(content=report)
        if stage == "review":
            assert payload["upstream_evidence"] == []
            assert payload["upstream_reports"]["draft"]["deliverables"][0]["artifact"] == "quiz"
            seen.append(stage)
            return AIMessage(content='{"findings":[],"gaps":[],"deliverables":[{"content":"题面未标注答案"}]}')
        assert payload["evidence_catalog"] == {}
        assert payload["reports"]["draft"]["deliverables"][0]["content"] == "这条消息应该是什么角色？"
        assert not tools
        return AIMessage(content="这条消息应该是什么角色？")

    result = asyncio.run(probe.build_graph(tmp_path, {book: {36}}, ask, [], roomy=True)
                         .ainvoke({"question": "出一道口头练习并检查题面"}))
    assert seen == ["review"] and result["delivery_status"] == "completed"


def test_planner_may_attach_an_already_authorized_page_to_the_source_name():
    task = probe.Assignment(id="A", instruction="读取第 36 页", sources=["book.pdf#page=36"], depends_on=[])
    probe.validate([task], {"book.pdf": {36}})
    assert task.sources == ["book.pdf"]
    task.sources = ["book.pdf#page=37"]
    with pytest.raises(ValueError, match="invalid source"):
        probe.validate([task], {"book.pdf": {36}})


def test_native_search_scopes_before_lookup_and_keeps_locations_without_registering_evidence(tmp_path, monkeypatch):
    first, second = probe.BOOKS
    invocations = []

    class Search:
        def invoke(self, args):
            assert args == {"queries": ["Observation"], "neighbor_window": 0}
            return json.dumps({"data": {"text": "定位片段", "locations": [
                {"source": first, "page": 85, "chunk_index": 5, "text": "Observation 原文片段"},
            ]}})

    def create(root, **kwargs):
        assert (kwargs["strategy"], kwargs["rerank"]) == ("dense", True)
        invocations.append(kwargs["source_paths"])
        return Search()

    monkeypatch.setattr(runtime, "build_retrieve_learning_context_tool", create)
    search = probe.build_scoped_search(tmp_path, {first: None, second: {36}}, probe.Lock())
    assert probe.build_scoped_search(tmp_path, {second: {36}}, probe.Lock()) is None
    assert json.loads(search.invoke({"queries": ["Observation"], "relative_path": second})) == {
        "error": "source_not_open_for_search",
    }
    result = json.loads(search.invoke({"queries": ["Observation"], "relative_path": first}))
    assert invocations == [frozenset({first})]
    assert result["locations"] == [{"source": first, "page": 85, "chunk_index": 5}]
    assert "evidence" not in result
    store, acquired = {}, set()
    with_evidence = probe.build_scoped_search(tmp_path, {first: None}, probe.Lock(), store, acquired)
    reply = json.loads(with_evidence.invoke({"queries": ["Observation"]}))
    (item,) = reply["evidence"]
    assert "snippets" not in reply
    assert item["text"] == "Observation 原文片段"
    assert store[item["evidence_id"]] == item
    assert acquired == {item["evidence_id"]}


def test_structure_tool_allows_open_document_and_rejects_fixed_page_scope(tmp_path):
    (tmp_path / "learning.md").write_text("# 第一节\n本节讲工具结果。\n", encoding="utf-8")
    tool = probe.build_scoped_structure_reader(tmp_path, {"learning.md": None, "book.pdf": {3}})
    assert json.loads(tool.invoke({"relative_path": "book.pdf"})) == {"error": "source_scope"}
    data = json.loads(tool.invoke({"relative_path": "learning.md"}))["data"]
    assert data["mode"] == "whole_document"
    assert "本节讲工具结果" in data["text"]


@pytest.mark.parametrize("source", ["learning.md", "learning.txt", "learning.py"])
def test_reuse_reads_lines_and_delivers_with_shared_state_without_copying_chat_to_worker(tmp_path, source):
    (tmp_path / source).write_text("# 阅读笔记\n工具结果用调用 ID 配对。\n", encoding="utf-8")
    memory = ConversationMemory(system_prompt="旧系统提示", strategy="summary")
    memory.summary = "仍需区分消息和回执。"
    memory.add_user("用中文讲解。")
    memory.add_ai("可以。")
    state = LearningState(learning_goal="理解工具调用", plan_markdown="先阅读再口述",
                          conversation_memory=memory.export_state())
    original = state.model_dump()

    async def ask(role, stage, messages, tools=()):
        if stage in {"planner", "deliver"}:
            assert "旧系统提示" not in messages[0].content
            assert messages[1].type == "human" and "仍需区分" in messages[1].content
            assert messages[2].content == "用中文讲解。"
            assert json.loads(messages[-2].content)["plan_markdown"] == "先阅读再口述"
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [{
                "id": "read", "instruction": "用中文核对第 2 行", "sources": [source], "depends_on": [],
            }]}))
        if stage == "deliver":
            (location,) = json.loads(messages[-1].content)["evidence_catalog"].values()
            assert location == {"source": source, "lines": [2, 2]}
            return AIMessage(content="工具结果用调用 ID 配对。")
        assert "仍需区分" not in messages[1].content
        if len(messages) == 2:
            return AIMessage(content="", tool_calls=[{
                "name": "read_material_lines", "id": "read-lines", "args": {
                    "relative_path": source, "start_line": 2, "end_line": 2,
                },
            }])
        (item,) = json.loads(messages[-1].content)["evidence"]
        return AIMessage(content=json.dumps({"findings": [{
            "statement": item["text"], "evidence_ids": [item["evidence_id"]],
        }], "gaps": []}))

    graph = probe.build_graph(tmp_path, {source: None}, ask, [], roomy=True, reuse_retrieval=True,
                              learning_state=state, source_footer=True)
    result = asyncio.run(graph.ainvoke({"question": "按当前计划读笔记第 2 行，再用中文说明。"}))
    assert result["delivery_status"] == "completed"
    assert f"[source={source}, lines=2-2]" in result["answer"]
    assert state.model_dump() == original


def test_structure_only_worker_delivers_short_article_with_traceable_full_text(tmp_path):
    source = "lesson.md"
    original = "# 工具结果\n结果带有工具调用对应的 ID。"
    (tmp_path / source).write_text(original, encoding="utf-8")
    stages = []

    async def ask(role, stage, messages, tools=()):
        stages.append(stage)
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [{
                "id": "A", "instruction": "用短文全文解释工具结果", "sources": [source], "depends_on": [],
            }]}))
        if stage == "deliver":
            (item,) = json.loads(messages[-1].content)["evidence_catalog"].values()
            assert item["kind"] == "document_structure"
            assert item["mode"] == "whole_document"
            return AIMessage(content="结果带有对应的调用 ID。")
        if len(messages) == 2:
            return AIMessage(content="", tool_calls=[{
                "id": "structure", "name": "inspect_document_structure", "args": {"relative_path": source},
            }])
        (item,) = json.loads(messages[-1].content)["evidence"]
        assert item["text"] == original
        return AIMessage(content=json.dumps({"findings": [{
            "statement": "结果带有对应的调用 ID", "evidence_ids": [item["evidence_id"]],
        }], "gaps": []}))

    trace = []
    result = asyncio.run(probe.build_graph(
        tmp_path, {source: None}, ask, trace, roomy=True, structure_lookup=True, source_footer=True,
    ).ainvoke({"question": "只用这篇短文解释工具结果与调用的关系。"}))
    assert stages == ["planner", "A", "A", "deliver"]
    assert [row["tool"] for row in trace] == ["inspect_document_structure"]
    assert result["results"]["A"]["status"] == "done"
    assert "[source=lesson.md, lines=1-2]" in result["answer"]


def test_structure_rejects_unverified_bookmarks_as_evidence(tmp_path, monkeypatch):
    class Reader:
        def invoke(self, args):
            return json.dumps({"ok": True, "data": {
                "source": "book.pdf", "mode": "sections", "sections": [
                    {"title": "未定位书签", "boundary_status": "located_in_metadata"},
                    {"title": "已定位标题", "start": {"page": 3}, "boundary_status": "located_in_extracted_text"},
                ], "sources": [{"citation": "[source=book.pdf, page=3]"}],
            }})

    monkeypatch.setattr(runtime, "build_inspect_document_structure_tool", lambda *args: Reader())
    store, acquired = {}, set()
    tool = probe.build_scoped_structure_reader(tmp_path, {"book.pdf": None}, store, acquired)
    (item,) = json.loads(tool.invoke({"relative_path": "book.pdf"}))["evidence"]
    assert json.loads(item["text"])[0]["title"] == "已定位标题"
    assert len(json.loads(item["text"])) == 1
    assert acquired == {item["evidence_id"]}


@pytest.mark.parametrize("strategy", ["bm25", "semantic"])
def test_history_only_worker_searches_then_reads_with_snapshot_limit(tmp_path, monkeypatch, strategy):
    settings = replace(probe.get_settings(), history_retrieval_strategy=strategy,
                       history_semantic_min_similarity=0.5)
    monkeypatch.setattr(runtime, "get_settings", lambda: settings)

    class Embeddings:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for text in texts]

        def embed_query(self, query):
            return [1.0, 0.0]

    monkeypatch.setattr(runtime, "create_embeddings", Embeddings)
    root = tmp_path / "archives"
    old_text = "更正：今天最多35分钟，只阅读和口头说明。"
    capture = ArchiveCapture(root, None, True)
    capture.capture(old_text, "已记住本轮条件。")
    bound_snapshot = capture.state.model_copy(deep=True)
    capture.capture("不属于旧快照的消息。", "了解。")
    memory = ConversationMemory(system_prompt="当前会话")
    memory.history_archive = bound_snapshot
    archive_file = next(root.glob("*.sqlite3"))
    before = archive_file.read_bytes()

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [{
                "id": "A", "instruction": "从历史中找出时间更正原话", "sources": [], "depends_on": [],
            }]}))
        if stage == "deliver":
            (item,) = json.loads(messages[-1].content)["evidence_catalog"].values()
            assert item["kind"] == "archived_conversation"
            return AIMessage(content=f"{old_text} {item['citation_ref']} [history-ref=1:2]")
        calls = [
            ("search_conversation_turns", {"query": "更正 今天最多35分钟"}),
            ("read_conversation_turn", {"source_index": 1, "turn_number": 2}),
            ("read_conversation_turn", {"source_index": 1, "turn_number": 1}),
        ]
        step = sum(isinstance(message, ToolMessage) for message in messages)
        if step:
            receipt = json.loads(messages[-1].content)
            if step == 1:
                assert "evidence" not in receipt and receipt["data"]["matches"][0]["turn_number"] == 1
                expected = "char_bigram_bm25" if strategy == "bm25" else "semantic"
                assert receipt["data"]["retrieval_strategy"] == expected
            elif step == 2:
                assert receipt["error"]["code"] == "turn_out_of_scope"
            else:
                (item,) = receipt["evidence"]
                assert json.loads(item["text"])["user_text"] == old_text
                return AIMessage(content=json.dumps({"findings": [{
                    "statement": old_text, "evidence_ids": [item["evidence_id"]],
                }], "gaps": []}))
        name, args = calls[step]
        assert name in {item.name for item in tools}
        return AIMessage(content="", tool_calls=[{"id": f"history-{step}", "name": name, "args": args}])

    graph = probe.build_graph(tmp_path, {}, ask, [], roomy=True, conversation_memory=memory,
                              history_archive_root=root, source_footer=True)
    result = asyncio.run(graph.ainvoke({"question": "逐字引用我当时更正学习时间的原话。"}))
    assert result["delivery_status"] == "completed"
    assert old_text in result["answer"]
    assert "[history=" in result["answer"] and "【历史引用未核实】" in result["answer"]
    assert "不属于旧快照的消息" not in result["answer"]
    assert archive_file.read_bytes() == before


def test_fallback_prioritizes_actual_receipt_with_requested_identifier():
    pages = [{"page": 34, "text": "Agent tool 介绍"},
             {"page": 36, "text": "tool_call_id 与工具请求关联"}]
    ranked = probe.rank_fallback_evidence("解释 tool_call_id", pages)
    assert [item["page"] for item in ranked] == [36, 34]


def test_roomy_worker_can_keep_reading_and_then_submit_in_a_fresh_request(tmp_path, monkeypatch):
    book = probe.BOOKS[0]
    monkeypatch.setattr(runtime, "read_pdf_pages", lambda *args: {
        "pages": [{"page": 36, "text": "tool_call_id 将工具结果关联到调用。"}],
        "truncated": False,
    })
    trace = []
    worker_calls = 0

    async def ask(role, stage, messages, tools=()):
        nonlocal worker_calls
        if stage == "planner":
            assert role == "standard"
            return AIMessage(content=json.dumps({"tasks": [{
                "id": "A", "instruction": "核对 tool_call_id", "sources": [book], "depends_on": [],
            }]}))
        if stage == "deliver":
            assert json.loads(messages[1].content)["reports"]["A"]["status"] == "done"
            return AIMessage(content="完整交付")
        worker_calls += 1
        if worker_calls <= 13:
            assert {"read_pdf", "recall_evidence", "submit_report"} <= {t.name for t in tools}
            return AIMessage(content="", tool_calls=[{
                "id": f"read-{worker_calls}", "name": "read_pdf",
                "args": {"relative_path": book, "start_page": 36, "end_page": 36},
            }])
        if worker_calls == 14:
            assert tools
            return AIMessage(content="未按格式提交")
        assert not tools
        assert len(messages) == 2
        (item,) = json.loads(messages[1].content)["evidence"]
        return AIMessage(content=json.dumps({"findings": [{
            "statement": "结果按 ID 配对", "evidence_ids": [item["evidence_id"]],
        }], "gaps": []}))

    graph = probe.build_graph(tmp_path, {book: {36}}, ask, trace,
                              planner_role="standard", bounded=True, roomy=True)
    result = asyncio.run(graph.ainvoke({"question": "解释 tool_call_id"}))
    assert worker_calls == 15
    assert sum(row["tool"] == "read_pdf" for row in trace) == 13
    assert result["answer"] == "完整交付"


def test_scoped_worker_rejects_extra_pages_and_has_no_locator(tmp_path):
    book = probe.BOOKS[0]
    (read,) = probe.build_tools(tmp_path, {book: {36}}, {}, set())
    assert json.loads(read.invoke({"relative_path": book, "start_page": 36, "end_page": 37})) == {
        "error": "page_scope"
    }
    assert json.loads(read.invoke({"relative_path": probe.BOOKS[1], "start_page": 36, "end_page": 36})) == {
        "error": "source_scope"
    }


def test_roomy_read_accepts_a_range_larger_than_the_underlying_batch(tmp_path, monkeypatch):
    book = probe.BOOKS[0]
    batches = []

    def read_page(root, source, start, end):
        if end - start >= 3:
            raise ValueError("超过单次底层页数")
        batches.append((start, end))
        return {"pages": [{"page": page, "text": f"正文 {page}"} for page in range(start, end + 1)],
                "truncated": False, "total_pages": 9}

    monkeypatch.setattr(runtime, "read_pdf_pages", read_page)
    read, _ = probe.build_tools(tmp_path, {book: None}, {}, set(), roomy=True)
    receipt = json.loads(read.invoke({"relative_path": book, "start_page": 2, "end_page": 8}))
    assert batches == [(2, 4), (5, 7), (8, 8)]
    assert [item["page"] for item in receipt["evidence"]] == list(range(2, 9))


def test_history_projection_preserves_call_pairing_and_original_evidence():
    def request(identifier):
        return AIMessage(content="", tool_calls=[{"name": "read_pdf", "args": {}, "id": identifier}])

    def receipt(identifier):
        return ToolMessage(tool_call_id=identifier, content=json.dumps({
            "evidence": [{"evidence_id": identifier, "text": "正文"}]
        }))

    history = [HumanMessage(content="任务"), request("one"), receipt("one"), request("two"), receipt("two")]
    projected = probe.project_history(history)
    assert json.loads(projected[2].content) == {"evidence": [{"evidence_id": "one"}]}
    assert projected[2].tool_call_id == "one"
    assert projected[4].content == history[4].content
    assert json.loads(history[2].content)["evidence"][0]["text"] == "正文"


@pytest.mark.parametrize("worker_full_question", [None, True])
def test_unknown_evidence_id_is_not_sent_to_delivery(tmp_path, worker_full_question):
    book = probe.BOOKS[0]
    stages = []

    async def ask(role, stage, messages, tools=()):
        stages.append(stage)
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [{
                "id": "A", "instruction": "核对第36页", "sources": [book], "depends_on": []
            }]}))
        if stage == "A":
            payload = json.loads(messages[1].content)
            assert payload["assignment"]["instruction"] == "核对第36页"
            if worker_full_question:
                assert payload["user_request"] == "只读第36页"
            else:
                assert "user_request" not in payload
                assert probe.WORKER_ASSIGNMENT_RULE in messages[0].content
            return AIMessage(content=json.dumps({"findings": [{
                "statement": "未核验的结论", "evidence_ids": ["invented"]
            }], "gaps": [], "deliverables": [{"content": "这项纯取证任务未取得证据。"}]}))
        payload = json.loads(messages[1].content)
        assert payload["question"] == "只读第36页"
        assert payload["results"]["A"]["status"] == "unavailable"
        assert payload["evidence"] == []
        return AIMessage(content="缺少可核验证据。")

    options = {"worker_full_question": True} if worker_full_question else {}
    graph = probe.build_graph(tmp_path, {book: {36}}, ask, [], **options)
    result = asyncio.run(graph.ainvoke({"question": "只读第36页"}))
    assert stages == ["planner", "A", "deliver"]
    assert result["answer"] == "缺少可核验证据。"


def test_dependent_worker_receives_referenced_text_once(tmp_path, monkeypatch):
    book = probe.BOOKS[0]
    monkeypatch.setattr(runtime, "read_pdf_pages", lambda *args: {
        "pages": [{"page": 36, "text": "工具结果使用调用 ID 配对"}], "truncated": False
    })
    tasks = [
        {"id": "A", "instruction": "核对配对方式", "sources": [book], "depends_on": []},
        {"id": "B", "instruction": "解释配对作用", "sources": [], "depends_on": ["A"]},
    ]

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": tasks}))
        if stage == "A" and len(messages) == 2:
            return AIMessage(content="", tool_calls=[{
                "id": "read", "name": "read_pdf",
                "args": {"relative_path": book, "start_page": 36, "end_page": 36},
            }])
        if stage == "A":
            items = json.loads(messages[-1].content)["evidence"]
        elif stage == "B":
            payload = json.loads(messages[1].content)
            (report,) = payload["upstream_reports"].values()
            assert "text" not in report["findings"][0]
            items = payload["upstream_evidence"]
            assert [item.name for item in tools] == ["read_pdf", "submit_report"]
        else:
            payload = json.loads(messages[1].content)
            assert len(payload["evidence"]) == 1
            return AIMessage(content="交付")
        assert len(items) == 1
        return AIMessage(content="", tool_calls=[{"id": "report", "name": "submit_report", "args": {
            "findings": [{"statement": "配对依据", "evidence_ids": [items[0]["evidence_id"]]}],
            "gaps": [],
        }}])

    result = asyncio.run(probe.build_graph(tmp_path, {book: {36}}, ask, []).ainvoke({"question": "解释关联"}))
    assert result["results"]["B"]["status"] == "done"


def test_invalid_submission_retains_raw_read_receipt(tmp_path, monkeypatch):
    book = probe.BOOKS[0]
    monkeypatch.setattr(runtime, "read_pdf_pages", lambda *args: {
        "pages": [{"page": 36, "text": "已读取正文"}], "truncated": False,
    })

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [{
                "id": "A", "instruction": "核对一页", "sources": [book], "depends_on": [],
            }]}))
        if stage == "A" and len(messages) == 2:
            return AIMessage(content="", tool_calls=[{
                "id": "read", "name": "read_pdf",
                "args": {"relative_path": book, "start_page": 36, "end_page": 36},
            }])
        if stage == "A":
            return AIMessage(content="这是无效交付")
        payload = json.loads(messages[1].content)
        assert payload["results"]["A"]["status"] == "unavailable"
        assert payload["evidence"][0]["text"] == "已读取正文"
        return AIMessage(content="阅读已成功，执行交付不可用。")

    state = asyncio.run(probe.build_graph(tmp_path, {book: {36}}, ask, []).ainvoke({"question": "只查第36页"}))
    assert state["answer"] == "阅读已成功，执行交付不可用。"


def test_worker_inherits_user_scope_beyond_planner_sources(tmp_path, monkeypatch):
    first, second = probe.BOOKS
    reads = []

    def read_page(_root, source, start, end):
        reads.append((source, start, end))
        return {"pages": [{"page": start, "text": "已取得另一份授权资料"}], "truncated": False}

    monkeypatch.setattr(runtime, "read_pdf_pages", read_page)
    scope = {first: {36}, second: {85}}

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [{
                "id": "A", "instruction": "从第一本开始核对", "sources": [first], "depends_on": [],
            }]}))
        if stage == "A" and len(messages) == 2:
            request = json.loads(messages[1].content)
            assert request["assignment"]["sources"] == [first]
            assert request["authorized_scope"] == {first: [36], second: [85]}
            return AIMessage(content="", tool_calls=[
                {"id": str(i), "name": "read_pdf",
                 "args": {"relative_path": source, "start_page": page, "end_page": page}}
                for i, (source, page) in enumerate(((second, 85), (second, 86), ("other.pdf", 1)))
            ])
        if stage == "A":
            found, outside_page, outside_file = [json.loads(message.content) for message in messages[-3:]]
            assert outside_page["error"] == "page_scope"
            assert outside_file["error"] == "source_scope"
            (item,) = found["evidence"]
            return AIMessage(content=json.dumps({"findings": [{
                "statement": "已核对第二本", "evidence_ids": [item["evidence_id"]],
            }], "gaps": []}))
        return AIMessage(content="已交付")

    result = asyncio.run(probe.build_graph(tmp_path, scope, ask, [], roomy=True)
                         .ainvoke({"question": "允许读第一本第36页和第二本第85页"}))
    assert result["results"]["A"]["status"] == "done"
    assert reads == [(second, 85, 85)]


def test_pdf_range_failure_reaches_delivery_without_worker_report(tmp_path, monkeypatch):
    book = probe.BOOKS[0]

    def out_of_bounds(_path, _source, first, _last):
        raise probe.PdfPageOutOfBounds(first, 720)

    monkeypatch.setattr(runtime, "read_pdf_pages", out_of_bounds)
    calls = 0

    async def ask(role, stage, messages, tools=()):
        nonlocal calls
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [{
                "id": "A", "instruction": "只查第9999页", "sources": [book], "depends_on": [],
            }]}))
        if stage == "A":
            calls += 1
            if calls <= 2:
                return AIMessage(content="", tool_calls=[{
                    "id": f"read-{calls}", "name": "read_pdf",
                    "args": {"relative_path": book, "start_page": 9999, "end_page": 9999},
                }])
            return AIMessage(content="无法解析的子任务交付")
        payload = json.loads(messages[1].content)
        assert payload["evidence_catalog"] == {}
        assert payload["reports"]["A"]["status"] == "unavailable"
        assert payload["reports"]["A"]["read_failures"] == [{
            "source": book, "code": "pdf_page_out_of_bounds", "message": "起始页超出PDF范围",
            "requested_start_page": 9999, "total_pages": 720, "retryable": False,
        }]
        return AIMessage(content="指定页越界")

    state = asyncio.run(probe.build_graph(tmp_path, {book: {9999}}, ask, [], roomy=True).ainvoke(
        {"question": "只读指定页，失败时说明原因"}
    ))
    assert state["answer"] == "指定页越界"
