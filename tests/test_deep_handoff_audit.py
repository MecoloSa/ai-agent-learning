from tests.helpers.deep_handoff_audit import project_context, separate_recorded_draft
from tests.helpers.delegated_reading import footer_evidence


def test_selects_findings_and_generated_artifacts_without_changing_original():
    payload = {
        "assignment": {"required_artifacts": {"reader": ["verified"], "writer": ["draft"]}},
        "upstream_reports": {
            "reader": {"status": "done", "gaps": ["图未提取"], "findings": [
                {"artifact": "verified", "evidence_ids": ["page"]},
                {"artifact": "background", "evidence_ids": ["other"]},
            ]},
            "writer": {"status": "done", "deliverables": [
                {"artifact": "draft", "content": "题干"}, {"artifact": "notes", "content": "备注"},
            ]},
        },
        "upstream_evidence": [{"evidence_id": "page", "page": 36, "text": "原文"},
                              {"evidence_id": "other", "page": 37, "text": "背景"}],
        "authorized_scope": {"book.pdf": [36, 37]},
    }
    selected = project_context(payload)
    assert selected["upstream_reports"]["reader"]["gaps"] == ["图未提取"]
    assert selected["upstream_reports"]["writer"]["deliverables"] == [{"artifact": "draft", "content": "题干"}]
    assert selected["upstream_evidence"] == [{"evidence_id": "page", "page": 36, "text": "原文"}]
    assert project_context(payload, references=True)["upstream_evidence"] == [{"evidence_id": "page", "page": 36}]
    assert selected["authorized_scope"] == payload["authorized_scope"]
    assert len(payload["upstream_evidence"]) == 2
    fallback = {key: value for key, value in payload.items() if key != "upstream_evidence"}
    fallback["evidence"] = payload["upstream_evidence"]
    assert project_context(fallback, references=True)["evidence"] == payload["upstream_evidence"]


def test_ordinary_dependency_keeps_full_context():
    payload = {"assignment": {"depends_on": ["reader"]}, "upstream_reports": {"reader": {}}}
    assert project_context(payload, references=True) == payload


def test_footer_does_not_expand_to_unused_reads_and_keeps_report_fallback():
    store = {key: {"evidence_id": key, "source": "book.pdf", "page": page}
             for key, page in (("candidate", 35), ("report", 36), ("rechecked", 37))}
    results = {"reader": {"evidence_ids": list(store), "findings": [{"evidence_ids": ["report"]}]}}
    assert footer_evidence(results, store, {"rechecked"}) == [store["rechecked"]]
    assert footer_evidence(results, store, set()) == [store["report"]]
    assert results["reader"]["evidence_ids"] == list(store)


def test_fixture_packaging_can_remove_notes_without_changing_evidence_view():
    content = "【最终输出：口头练习题干（仅题干）】\n题干正文\n【以下为审核用说明，不属于最终输出】\n审核备注"
    evidence = [{"evidence_id": "p36", "text": "原文"}]
    payload = {
        "assignment": {"required_artifacts": {"read": ["role"], "draft": ["stem"]}},
        "upstream_reports": {
            "read": {"findings": [{"artifact": "role", "evidence_ids": ["p36"]}]},
            "draft": {"deliverables": [{"artifact": "stem", "content": content}]},
        },
        "upstream_evidence": evidence,
    }
    split = separate_recorded_draft(payload)
    assert split["upstream_reports"]["draft"]["deliverables"][1]["content"] == "审核备注"
    view = project_context(split)
    assert view["upstream_reports"]["draft"]["deliverables"] == [{"artifact": "stem", "content": "题干正文"}]
    assert view["upstream_evidence"] == evidence
    assert payload["upstream_reports"]["draft"]["deliverables"][0]["content"] == content
