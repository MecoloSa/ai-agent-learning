import json
from types import SimpleNamespace

import pytest

import ai_agent.evaluation.review as review_module
from ai_agent.evaluation.review import REVIEW_ITEMS, build_review
import ai_agent.evaluation.source_evidence as source_module


def test_review_does_not_treat_execution_success_as_quality_pass():
    record = {
        "run_id": "run-001",
        "case_id": "case-001",
        "mode": "pipeline",
        "success": True,
    }

    review = build_review(record, "file-hash")

    assert review["run_id"] == "run-001"
    assert review["review_status"] == "pending_review"
    assert set(review["checks"]) == set(REVIEW_ITEMS)

    # 程序跑通，不代表内容质量通过；初始判定必须全部待审核。
    assert all(value is None for value in review["checks"].values())


def test_failed_execution_is_not_marked_as_quality_pass():
    record = {
        "run_id": "run-002",
        "case_id": "case-001",
        "mode": "agent",
        "success": False,
    }

    review = build_review(record, "file-hash")

    assert review["review_status"] == "not_evaluated"
    assert all(value is None for value in review["checks"].values())

def test_judge_cannot_confirm_citations_without_source(monkeypatch):
    fake_result = {
        "checks": {
            item: True for item in review_module.REVIEW_ITEMS
        },
        "evidence_notes": {
            item: "模拟评价理由"
            for item in review_module.REVIEW_ITEMS
        },
    }

    def fake_invoke(llm, messages, *, stage):
        return SimpleNamespace(
            content=json.dumps(fake_result, ensure_ascii=False)
        )

    monkeypatch.setattr(
        review_module, "invoke_llm_response", fake_invoke
    )

    result = review_module.judge_plan(
        {"plan": "第1天学习计划。"},
        {"learning_goal": "学习Agent", "days": 1},
        llm=object(),
    )

    # 即使模型声称引用正确，没有原文也不能确认支持度。
    assert result["checks"]["citation_support"] is None


def test_judge_receives_original_text_and_can_reject_citation(monkeypatch):
    pack = {"documents": [{"source": "book.txt", "page": None,
                           "text": "原文：模型回复不是用户消息。"}],
            "coverage_complete": True}
    fake_result = {
        "checks": {item: False for item in REVIEW_ITEMS},
        "evidence_notes": {item: "论断与提供的原文不符" for item in REVIEW_ITEMS},
    }

    def fake_invoke(llm, messages, *, stage):
        payload = json.loads(messages[1].content)
        assert payload["source_evidence"] == pack
        return SimpleNamespace(content=json.dumps(fake_result))

    monkeypatch.setattr(review_module, "invoke_llm_response", fake_invoke)
    result = review_module.judge_plan(
        {"plan": "模型回复是用户消息。[source=book.txt]"},
        {"learning_goal": "学习消息结构", "days": 1}, object(), evidence=pack,
    )
    assert result["checks"]["citation_support"] is False


def test_incomplete_evidence_cannot_receive_overall_citation_pass(monkeypatch):
    result = {"checks": {item: True for item in REVIEW_ITEMS},
              "evidence_notes": {item: "模拟理由" for item in REVIEW_ITEMS}}
    monkeypatch.setattr(review_module, "invoke_llm_response", lambda *a, **kw:
                        SimpleNamespace(content=json.dumps(result)))
    checked = review_module.judge_plan(
        {"plan": "计划"}, {"learning_goal": "目标", "days": 1}, object(),
        evidence={"documents": [{"text": "部分原文"}], "coverage_complete": False},
    )
    assert checked["checks"]["citation_support"] is None


def test_source_loader_reads_actual_text_and_reports_missing_file(tmp_path):
    (tmp_path / "book.txt").write_text("原始资料正文", encoding="utf-8")
    pack = source_module.load_source_evidence(
        tmp_path, "[source=book.txt, chunk=0][source=missing.txt, chunk=1]",
    )
    assert pack["documents"][0]["text"] == "原始资料正文"
    assert pack["files"][0]["sha256"]
    assert pack["issues"] and pack["coverage_complete"] is False


def test_pdf_loader_uses_one_based_pages_and_keeps_neighbor_labels(tmp_path, monkeypatch):
    (tmp_path / "book.pdf").write_bytes(b"fake-pdf")
    monkeypatch.setattr(source_module, "PdfReader", lambda path: SimpleNamespace(
        pages=[SimpleNamespace(extract_text=lambda text=text: text)
               for text in ("第一页", "第二页", "第三页", "第四页")],
    ))
    pack = source_module.load_source_evidence(tmp_path, "[source=book.pdf, page=2, chunk=5]")
    assert [(d["page"], d["text"]) for d in pack["documents"]] == [
        (1, "第一页"), (2, "第二页"), (3, "第三页"),
    ]
    assert pack["coverage_complete"] is True
    assert pack["chunk_validation"] == "not_verified"
    invalid = source_module.load_source_evidence(tmp_path, "[source=book.pdf, page=99]")
    assert invalid["issues"] and not invalid["documents"]


def test_source_budget_rejects_overflow_without_silent_truncation(tmp_path):
    (tmp_path / "book.txt").write_text("正文" * 200, encoding="utf-8")
    with pytest.raises(ValueError, match="超过字符预算"):
        source_module.load_source_evidence(tmp_path, "[source=book.txt]", max_chars=100)


def test_full_mode_reads_uncited_supported_files(tmp_path):
    (tmp_path / "a.txt").write_text("第一份资料", encoding="utf-8")
    (tmp_path / "b.md").write_text("第二份资料", encoding="utf-8")
    pack = source_module.load_source_evidence(tmp_path, "[source=a.txt]", mode="full")
    assert {d["source"] for d in pack["documents"]} == {"a.txt", "b.md"}


def test_source_loader_does_not_follow_citation_outside_directory(tmp_path):
    pack = source_module.load_source_evidence(tmp_path, "[source=../private.txt]")
    assert not pack["documents"]
    assert pack["issues"]


def test_malformed_citation_does_not_count_as_complete_coverage(tmp_path):
    (tmp_path / "book.txt").write_text("原文", encoding="utf-8")
    pack = source_module.load_source_evidence(
        tmp_path, "[source=book.txt][source=book.pdf, page=1-3]",
    )
    assert pack["documents"]
    assert pack["coverage_complete"] is False
    assert any("引用格式无法定位" in issue for issue in pack["issues"])
