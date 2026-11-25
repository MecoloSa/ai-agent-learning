import json

from langchain_core.messages import AIMessage, ToolMessage

from tests.helpers.interactive_receipt_compare import project_receipts


def pair(key, start, payload):
    return [
        AIMessage(content="", tool_calls=[{"name": "read_pdf_pages", "id": key,
            "args": {"relative_path": "book.pdf", "start_page": start, "end_page": start}}]),
        ToolMessage(content=json.dumps(payload), tool_call_id=key),
    ]


def success(page, text):
    return {"ok": True, "data": {"pages": [{"citation": f"book.pdf:{page}", "text": text, "partial": False}]}}


def test_only_exact_duplicates_and_resolved_errors_are_compacted():
    error = {"ok": False, "error": {"code": "temporary_read_error"}}
    history = [
        *pair("corrected", 1, error),
        *pair("first", 1, success(1, "same text")),
        *pair("different-content", 1, success(1, "updated text")),
        *pair("unresolved", 2, error),
        *pair("last", 1, success(1, "same text")),
    ]

    projected = project_receipts(history)
    assert [message.tool_call_id for message in projected[1::2]] == [
        "corrected", "first", "different-content", "unresolved", "last",
    ]
    assert json.loads(projected[1].content) == {"ok": False, "superseded_by": "first"}
    cleaned = json.loads(projected[3].content)["data"]["pages"][0]
    assert "text" not in cleaned
    assert cleaned["text_in_receipt"] == "last"
    assert [projected[index] for index in (5, 7, 9)] == [history[index] for index in (5, 7, 9)]
    assert json.loads(history[3].content) == success(1, "same text")
