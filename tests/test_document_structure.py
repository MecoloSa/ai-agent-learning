"""Structure decisions use real positions and preserve document coverage."""

import json

import pytest
from langchain_core.messages import AIMessage

from ai_agent.resources.document_structure import inspect_document_structure
from ai_agent.runtime.source_delivery import TurnSourceLedger


class Model:
    def __init__(self, decide):
        self.decide = decide
        self.contexts = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        context = json.loads(messages[-1].content)
        self.contexts.append(context)
        return AIMessage(content="", tool_calls=[{
            "name": "StructureDecision", "id": "structure", "args": self.decide(context),
        }])


def inspect(tmp_path, name, model):
    return inspect_document_structure(str(tmp_path), name, model_factory=lambda: model)


def test_short_article_is_returned_complete_without_model(tmp_path):
    text = "# 文章\n正文。\n## 末节\n末节独有事实3617。"
    (tmp_path / "article.md").write_text(text, encoding="utf-8")
    result = inspect(tmp_path, "article.md", None)
    assert result["text"] == text and result["mode"] == "whole_document"
    assert result["invocations"] == []
    ledger = TurnSourceLedger()
    ledger.observe("inspect_document_structure", json.dumps({"ok": True, "data": result}), executed=True)
    assert "lines=1-4" in ledger.render()


def test_long_markdown_keeps_major_chapters_and_skips_code_headings(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMENT_DIRECT_READ_MAX_CHARS", "10")
    (tmp_path / "article.md").write_text(
        "# 文章\n## 第一章\n内容\n### 小节\n内容\n## 第二章\n内容\n```md\n# 代码中的标题\n```", encoding="utf-8")
    model = Model(lambda context: {"action": "submit", "sections": [
        {"location_id": item["location_id"], "level": item["native_level"]}
        for item in context["locations"]
    ]})
    result = inspect(tmp_path, "article.md", model)
    assert [item["title"] for item in result["sections"]] == ["第一章", "第二章"]
    assert result["sections"][0]["end_exclusive"]["line"] == 6
    assert all("代码中的标题" != item["title"] for item in model.contexts[0]["locations"])
    assert len(result["invocations"]) == 1


def test_unknown_location_and_path_escape_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMENT_DIRECT_READ_MAX_CHARS", "1")
    (tmp_path / "article.md").write_text("# 内容\n文章", encoding="utf-8")
    model = Model(lambda _: {"action": "submit", "sections": [{"location_id": "invented", "level": 1}]})
    with pytest.raises(ValueError, match="未提供"):
        inspect(tmp_path, "article.md", model)
    with pytest.raises(ValueError):
        inspect(tmp_path, "../outside.md", model)


def test_single_chapter_is_not_expanded_into_its_subsections(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMENT_DIRECT_READ_MAX_CHARS", "1")
    (tmp_path / "chapter.md").write_text(
        "# 第一章 工具\n## 定义\n内容\n## 例子\n内容", encoding="utf-8")
    model = Model(lambda context: {"action": "submit", "sections": [
        {"location_id": item["location_id"], "level": item["native_level"]} for item in context["locations"]
    ]})
    result = inspect(tmp_path, "chapter.md", model)
    assert [item["title"] for item in result["sections"]] == ["第一章 工具"]


def test_exhausted_structure_budget_reports_incomplete(tmp_path, monkeypatch):
    from pypdf import PdfWriter
    monkeypatch.setenv("DOCUMENT_STRUCTURE_MAX_ROUNDS", "1")
    monkeypatch.setenv("DOCUMENT_DIRECT_READ_MAX_CHARS", "1")
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    writer.write(tmp_path / "blank.pdf")
    model = Model(lambda _: {"action": "read_more", "requested_pages": [1]})
    result = inspect(tmp_path, "blank.pdf", model)
    assert not result["inspection_complete"]
    assert not result["sections"] and result["limitations"]


def test_empty_structure_submission_does_not_claim_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMENT_DIRECT_READ_MAX_CHARS", "1")
    (tmp_path / "plain.txt").write_text("没有明确标题的长正文。", encoding="utf-8")
    result = inspect(tmp_path, "plain.txt", Model(lambda _: {"action": "submit", "sections": []}))
    assert not result["inspection_complete"]


@pytest.mark.parametrize(("configured_rounds", "expected_rounds"), [(None, 3), ("10", 10)])
def test_structure_budget_allows_submission_in_final_round(
    tmp_path, monkeypatch, configured_rounds, expected_rounds,
):
    from pypdf import PdfWriter
    monkeypatch.delenv("DOCUMENT_STRUCTURE_MAX_ROUNDS", raising=False)
    if configured_rounds is not None:
        monkeypatch.setenv("DOCUMENT_STRUCTURE_MAX_ROUNDS", configured_rounds)
    monkeypatch.setenv("DOCUMENT_STRUCTURE_SCAN_MAX_PAGES", "1")
    writer = PdfWriter()
    for _ in range(expected_rounds):
        writer.add_blank_page(width=300, height=400)
    writer.add_outline_item("第一章", 0)
    writer.write(tmp_path / "rounds.pdf")
    count = 0

    def decide(context):
        nonlocal count
        count += 1
        if count < expected_rounds:
            return {"action": "read_more", "requested_pages": [count + 1]}
        return {"action": "submit", "sections": [
            {"location_id": context["locations"][0]["location_id"], "level": 1},
        ]}

    result = inspect(tmp_path, "rounds.pdf", Model(decide))
    assert result["scanned_pages"] == list(range(1, expected_rounds + 1))
    assert len(result["invocations"]) == expected_rounds
    assert result["sections"][0]["title"] == "第一章"


def test_pdf_bookmarks_use_physical_pages_and_bounded_read_more(tmp_path, monkeypatch):
    from pypdf import PdfWriter
    monkeypatch.setenv("DOCUMENT_DIRECT_READ_MAX_CHARS", "1")
    writer = PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=300, height=400)
    writer.add_outline_item("第一章", 1)
    writer.add_outline_item("第二章", 3)
    writer.write(tmp_path / "book.pdf")
    count = 0

    def decide(context):
        nonlocal count
        count += 1
        if count == 1:
            return {"action": "read_more", "requested_pages": [2]}
        return {"action": "submit", "sections": [
            {"location_id": item["location_id"], "level": item["native_level"]}
            for item in context["locations"] if "native_level" in item
        ]}

    result = inspect(tmp_path, "book.pdf", Model(decide))
    assert [item["start"]["page"] for item in result["sections"]] == [2, 4]
    assert len(result["invocations"]) == 2
    assert result["sections"][0]["boundary_status"] == "located_in_metadata"
    assert result["sources"] == []  # 书签不会被登记为已读正文。


@pytest.mark.parametrize("outline", ["verified", "wrong_page", "same_page", "missing"])
def test_pdf_outline_fast_path_requires_every_title_at_its_destination(tmp_path, monkeypatch, outline):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    from ai_agent.resources.pdf_materials import read_pdf_pages

    monkeypatch.setenv("DOCUMENT_DIRECT_READ_MAX_CHARS", "1")
    monkeypatch.setenv("DOCUMENT_STRUCTURE_SCAN_MAX_PAGES", "1")
    writer = PdfWriter()
    for title in ("Cover", "Chapter One", "Supporting text", "Chapter Two"):
        page = writer.add_blank_page(width=300, height=400)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 24 350 Td ({title}) Tj ET".encode("ascii"))
        page.replace_contents(stream)
    writer.set_page_label(1, 3, style="/D", start=1)
    if outline != "missing":
        writer.add_outline_item("Chapter One", 1)
        writer.add_outline_item("Chapter Two", {"verified": 3, "wrong_page": 2, "same_page": 1}[outline])
    writer.write(tmp_path / "book.pdf")

    model = Model(lambda _: {"action": "submit", "sections": []})
    result = inspect(tmp_path, "book.pdf", model)
    if outline == "verified":
        assert result["mode"] == "verified_outline"
        assert result["outline_verified"] and not result["inspection_complete"]
        assert result["scanned_pages"] == [1, 2, 4]
        assert [(s["start"]["page"], s["start"]["pdf_page_label"]) for s in result["sections"]] == [(2, "1"), (4, "3")]
        assert result["sections"][0]["end_exclusive"]["page"] == 4
        assert not model.contexts
        assert read_pdf_pages(str(tmp_path), "book.pdf", 4, 4)["pages"][0]["pdf_page_label"] == "3"
    else:
        assert not result.get("outline_verified")
        assert len(model.contexts) == 1
        if outline == "wrong_page":
            assert result["scanned_pages"] == [1, 2, 3]


def test_workspace_main_agent_can_read_short_article_through_structure_tool(tmp_path, monkeypatch):
    import ai_agent.main as app
    from ai_agent.learning.learning_state import LearningState
    from ai_agent.learning.workspace_service import WorkspaceCommand, WorkspaceService
    from ai_agent.learning.workspace_store import WorkspaceStore
    from ai_agent.persistence.session_materials import create_material_binding
    from uuid import uuid4

    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "article.md").write_text("# 文章\n## 第一节\n定义\n## 末节\n独有事实3617", encoding="utf-8")
    state = LearningState(learning_goal="读文章", plan_markdown="阅读文章",
                          material_binding=create_material_binding(str(materials)))
    store = WorkspaceStore(tmp_path / "workspace.sqlite3", state)
    monkeypatch.setattr(app, "create_llm", lambda: object())
    monkeypatch.setattr(app, "create_llm_with_tools", lambda _: object())
    messages_seen = []
    activity = []

    def respond(llm, messages, *, stage, **kwargs):
        messages_seen.append(messages)
        if len(messages_seen) == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "inspect_document_structure", "id": "structure",
                "args": {"relative_path": "article.md"},
            }, {
                "name": "read_pdf_pages", "id": "wide-range",
                "args": {"relative_path": "book.pdf", "start_page": 8, "end_page": 12},
            }])
        assert any("独有事实3617" in str(item.content) for item in messages)
        assert any("pdf_page_range_limit" in str(item.content) for item in messages)
        return AIMessage(content="文章包含定义与末节，末节的事实编号是3617。")

    monkeypatch.setattr(app, "invoke_llm_response", respond)
    event = WorkspaceService(store, tmp_path).execute(WorkspaceCommand(
        action="chat", request_id=uuid4().hex, version=1,
        payload={"text": "请按原文顺序列出文章所有章节，并解释每章讲了什么。"},
    ), stream_observer=lambda kind, data: activity.append(data) if kind == "tool_activity" else None)
    assert "3617" in event.message and "程序整理的本轮读取来源" in event.message
    assert [(row["name"], row["status"]) for row in activity] == [
        ("inspect_document_structure", "running"), ("inspect_document_structure", "completed"),
        ("read_pdf_pages", "running"), ("read_pdf_pages", "failed"),
    ]
    assert activity[-1]["reason"] == "请求页数超过单次上限，请缩小范围。"
    assert len(messages_seen) == 2 and store.load().version == 1
