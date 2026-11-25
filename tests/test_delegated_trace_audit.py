from tests.helpers.delegated_trace_audit import audit_branch


def test_distinguishes_new_receipts_changed_args_and_optional_recall():
    item = {"source": "AI-Agents-in-Depth.pdf", "page": 41, "evidence_id": "page-41",
            "text": "tool_call_id 关联对应调用"}

    def row(name, args):
        return {"tool": name, "args": args, "result": {"evidence": [item]}}

    result = audit_branch([
        row("read_pdf", {"start_page": 41}),
        row("recall_evidence", {"evidence_id": "page-41"}),
        *(row("read_pdf", {"start_page": page}) for page in (40, 39, 38, 37)),
    ], [])
    assert result["repeated_args_including_recall"] == 0
    assert result["acquisitions_without_new_receipt"] == 4
    assert result["four_stall_checkpoint"] == {"acquisition": 5, "anchors": ["tool_call_id"]}


def test_new_receipt_on_same_page_counts_and_no_text_locator_is_progress():
    rows = [
        {"tool": "locate_pdf", "args": {"relative_path": "book.pdf", "query": "a"},
         "result": {"candidates": [{"page": 3}]}},
        {"tool": "read_pdf", "args": {"relative_path": "book.pdf"},
         "result": {"evidence": [{"evidence_id": "p3", "source": "book.pdf", "page": 3, "text": "正文"}]}},
    ]
    assert audit_branch(rows, [])["acquisitions_without_new_receipt"] == 0
