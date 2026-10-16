"""评审校准回归测试：验证程序保护，不声称验证真实模型语义能力。

使用假PDF/假LLM覆盖正确定位、缺失、错页、证据不足和错误响应。
书籍JSONL用于真实Judge校准；其人工标签不会进入模型上下文。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import ai_agent.evaluation.calibration as calibration
import ai_agent.evaluation.review as review
import ai_agent.evaluation.source_evidence as source


def make_pack(tmp_path, monkeypatch, plan):
    """伪造三页PDF，隔离磁盘内容解析，专注引用与支持页的对应关系。"""
    (tmp_path / "book.pdf").write_bytes(b"fake")
    monkeypatch.setattr(source, "PdfReader", lambda _: SimpleNamespace(pages=[
        SimpleNamespace(extract_text=lambda t=t: t)
        for t in ("前页背景", "本页原文", "后页实现")
    ]))
    return source.load_source_evidence(tmp_path, plan)


def fake_judge(monkeypatch, pack, *, supporting_pages=None, supported=True):
    result = {
        "checks": {item: True for item in review.REVIEW_ITEMS},
        "evidence_notes": {item: "模拟总体通过" for item in review.REVIEW_ITEMS},
        "citation_reviews": [{"citation": c["citation"], "supported": supported,
                              "supporting_pages": supporting_pages if supporting_pages is not None else [2],
                              "reason": "模拟逐引用判定"} for c in pack["citation_checks"]],
    }
    def invoke(llm, messages, *, stage):
        # 断言实际模型输入遵守统一通过语义，而不只是观察最终输出。
        assert "true=符合，false=确认不符合" in messages[0].content
        assert "概念解释是否准确" in messages[0].content
        assert "是否遵守私有思维链边界" in messages[0].content

        # 检查传入模型的实际规则，避免只更新文档而遗漏提示词。
        assert "内部矛盾不自动导致本项失败" in messages[0].content
        assert "此处true表示没有违反要求，不代表进行了实践" in messages[0].content
        assert "只有缺少判断所必需的信息" in messages[0].content
        return SimpleNamespace(content=json.dumps(result, ensure_ascii=False))
    monkeypatch.setattr(review, "invoke_llm_response", invoke)
    return result


def judge(pack):
    return review.judge_plan({"plan": "待评审计划"},
                             {"learning_goal": "学习引用", "days": 1}, object(), evidence=pack)


def test_source_marks_original_page_and_neighbors(tmp_path, monkeypatch):
    pack = make_pack(tmp_path, monkeypatch, "[source=book.pdf, page=2]")
    assert [d["evidence_role"] for d in pack["documents"]] == ["neighbor", "cited", "neighbor"]
    assert pack["citation_checks"][0]["status"] == "located"


@pytest.mark.parametrize("marker", [
    "[source=missing.pdf, page=1]", "[source=book.pdf, page=99]",
    "[source=book.pdf]", "[source=book.pdf, page=1-3]",
])
def test_invalid_citation_cannot_pass_even_if_model_says_true(tmp_path, monkeypatch, marker):
    pack = make_pack(tmp_path, monkeypatch, marker)
    fake_judge(monkeypatch, pack)
    assert judge(pack)["checks"]["citation_support"] is False


def test_neighbor_support_does_not_validate_cited_page(tmp_path, monkeypatch):
    pack = make_pack(tmp_path, monkeypatch, "[source=book.pdf, page=2]")
    fake_judge(monkeypatch, pack, supporting_pages=[3])
    result = judge(pack)
    assert result["checks"]["citation_support"] is False
    assert "不含被引用页" in result["citation_reviews"][0]["reason"]


def test_original_page_support_can_pass_without_claiming_chunk_verified(tmp_path, monkeypatch):
    pack = make_pack(tmp_path, monkeypatch, "[source=book.pdf, page=2, chunk=999]")
    fake_judge(monkeypatch, pack)
    result = judge(pack)
    assert result["checks"]["citation_support"] is True
    assert pack["chunk_validation"] == "not_verified"
    assert "chunk编号和图像未验证" in result["evidence_notes"]["citation_support"]


def test_unprovided_support_page_cannot_pass(tmp_path, monkeypatch):
    pack = make_pack(tmp_path, monkeypatch, "[source=book.pdf, page=2]")
    fake_judge(monkeypatch, pack, supporting_pages=[2, 100])
    assert judge(pack)["checks"]["citation_support"] is None


def test_no_citation_stays_unknown(tmp_path, monkeypatch):
    pack = make_pack(tmp_path, monkeypatch, "没有出处")
    fake_judge(monkeypatch, pack)
    assert judge(pack)["checks"]["citation_support"] is None


def test_empty_cited_page_cannot_borrow_neighbor_text(tmp_path, monkeypatch):
    pack = make_pack(tmp_path, monkeypatch, "[source=book.pdf, page=2]")
    monkeypatch.setattr(source, "PdfReader", lambda _: SimpleNamespace(pages=[
        SimpleNamespace(extract_text=lambda t=t: t) for t in ("前页", "", "后页")
    ]))
    pack = source.load_source_evidence(tmp_path, "[source=book.pdf, page=2]")
    fake_judge(monkeypatch, pack)
    assert pack["citation_checks"][0]["status"] == "incomplete"
    assert judge(pack)["checks"]["citation_support"] is None


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_type"])
def test_malformed_per_citation_response_is_rejected(tmp_path, monkeypatch, change):
    pack = make_pack(tmp_path, monkeypatch, "[source=book.pdf, page=2]")
    result = fake_judge(monkeypatch, pack)
    if change == "missing":
        result["citation_reviews"] = []
    elif change == "duplicate":
        result["citation_reviews"] *= 2
    else:
        result["citation_reviews"][0]["supporting_pages"] = [True]
    with pytest.raises(ValueError):
        judge(pack)


def test_one_wrong_citation_prevents_overall_pass(tmp_path, monkeypatch):
    pack = make_pack(tmp_path, monkeypatch, "[source=book.pdf, page=2][source=missing.pdf, page=1]")
    fake_judge(monkeypatch, pack)
    assert judge(pack)["checks"]["citation_support"] is False


def test_book_cases_cover_both_books_and_paired_page_correction():
    cases = calibration.load_calibration_cases(Path(__file__).parent / "fixtures/evaluation/judge_calibration_cases.jsonl")
    by_id = {c["case_id"]: c for c in cases}
    assert len(cases) == 16

    # 防止案例数量增加了，但关键边界仍未纳入人工评分。
    assert by_id["judge-book-007"]["expected_checks"] == {
        "goal_alignment": False,
        "internal_consistency": False,
    }
    assert by_id["judge-book-010"]["expected_checks"] == {
        "practice_executable": True,
        "citation_support": None,
    }
    assert "AI-Agents-in-Depth-zh-CN.pdf" in by_id["judge-book-001"]["plan"]
    assert "Hello-Agents-V1.0.3-20260717.pdf" in by_id["judge-book-002"]["plan"]
    assert by_id["judge-book-002"]["plan"].replace("page=91", "page=92") == by_id["judge-book-003"]["plan"]


def test_calibration_reports_false_pass_false_reject_unknown_and_failed():
    records = [
        {"status": "judged", "expected_checks": {"citation_support": False}, "checks": {"citation_support": True}},
        {"status": "judged", "expected_checks": {"concept_accuracy": True}, "checks": {"concept_accuracy": False}},
        {"status": "judged", "expected_checks": {"citation_support": None}, "checks": {"citation_support": None}},
        {"status": "judge_failed", "expected_checks": {"citation_support": False}},
    ]
    summary = calibration.summarize_calibration(records)
    assert summary["false_pass_rate"] == 1.0
    assert summary["false_reject_rate"] == 1.0
    assert summary["unknown_rate"] == pytest.approx(1 / 3)
    assert summary["agreement_rate"] == pytest.approx(1 / 3)
    assert summary["judge_failed"] == 1
    assert calibration.summarize_calibration([])["agreement_rate"] is None


def test_calibration_does_not_send_human_labels_to_judge(tmp_path, monkeypatch):
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(sys, "argv", ["calibration", "--cases", str(Path(__file__).parent / "fixtures/evaluation/judge_calibration_cases.jsonl"), "--output", str(output), "--llm"])
    monkeypatch.setattr(calibration, "load_source_evidence", lambda *a, **kw: {})
    monkeypatch.setattr(calibration, "create_llm", lambda: SimpleNamespace(model_name="fake"))
    def invoke(record, case, llm, *, evidence):
        assert set(record) == {"plan"}
        assert set(case) == {"learning_goal", "days"}
        return {"checks": {item: None for item in review.REVIEW_ITEMS}}
    monkeypatch.setattr(calibration, "judge_plan", invoke)
    calibration.main()
    assert len(output.read_text(encoding="utf-8").splitlines()) == 16

def test_trace_preserves_raw_pass_before_invalid_citation_rejection(
    tmp_path, monkeypatch
):
    pack = make_pack(
        tmp_path, monkeypatch, "[source=missing.pdf, page=1]"
    )
    fake_judge(monkeypatch, pack)
    result = judge(pack)

    assert result["raw_model_result"]["checks"]["citation_support"] is True
    assert result["checks"]["citation_support"] is False
    assert result["raw_model_result"]["citation_reviews"][0]["supported"] is True
    assert result["citation_reviews"][0]["supported"] is False

    assert {change["scope"] for change in result["rule_changes"]} == {
        "dimension", "citation"
    }

    # 验证快照独立：修改最终结果不能污染原始结果。
    result["citation_reviews"][0]["supporting_pages"].append(99)
    assert result["raw_model_result"]["citation_reviews"][0][
        "supporting_pages"
    ] == [2]


def test_trace_does_not_count_reason_rewrite_as_verdict_change(
    tmp_path, monkeypatch
):
    pack = make_pack(
        tmp_path, monkeypatch, "[source=book.pdf, page=2]"
    )
    fake_judge(monkeypatch, pack)
    result = judge(pack)

    assert result["checks"]["citation_support"] is True
    assert result["rule_changes"] == []
    assert result["trace_version"] == "judge-trace-v1"


def test_trace_records_unknown_when_no_citation_exists(
    tmp_path, monkeypatch
):
    pack = make_pack(tmp_path, monkeypatch, "没有出处")
    fake_judge(monkeypatch, pack)
    result = judge(pack)

    assert result["raw_model_result"]["checks"]["citation_support"] is True
    assert result["checks"]["citation_support"] is None
    assert len(result["rule_changes"]) == 1
    assert result["rule_changes"][0]["after"] is None


def test_trace_summary_separates_raw_error_from_final_correction():
    records = [{
        "status": "judged",
        "expected_checks": {"citation_support": False},
        "checks": {"citation_support": False},
        "suggestion": {
            "raw_model_result": {
                "checks": {"citation_support": True}
            },
            "rule_changes": [
                {"scope": "dimension"},
                {"scope": "citation"},
            ],
        },
    }]

    summary = calibration.summarize_judge_trace(records)

    assert summary["raw_model"]["false_pass_rate"] == 1.0
    assert summary["final_same_cases"]["false_pass_rate"] == 0.0
    assert summary["cases_with_verdict_changes"] == 1
    assert summary["dimension_verdict_changes"] == 1
    assert summary["citation_verdict_changes"] == 1


def test_trace_summary_does_not_invent_raw_results_for_old_files():
    records = [{
        "status": "judged",
        "expected_checks": {"citation_support": False},
        "checks": {"citation_support": False},
    }]

    summary = calibration.summarize_judge_trace(records)

    assert summary["trace_case_count"] == 0
    assert summary["judged_cases_without_trace"] == 1
    assert summary["raw_model"]["agreement_rate"] is None
    assert summary["final_same_cases"]["agreement_rate"] is None

@pytest.mark.parametrize(
    "positive_id,negative_id,dimension",
    [
        ("judge-book-011", "judge-book-012", "goal_alignment"),
        ("judge-book-013", "judge-book-014", "practice_executable"),
        ("judge-book-015", "judge-book-016", "reasoning_boundary"),
    ],
)
def test_new_calibration_pairs_share_goal_and_opposite_labels(
    positive_id, negative_id, dimension
):
    """检查配对设计，避免同时更换目标而无法解释评分差异。"""
    cases = calibration.load_calibration_cases(
        Path(__file__).parent
        / "fixtures/evaluation/judge_calibration_cases.jsonl"
    )
    by_id = {case["case_id"]: case for case in cases}
    positive = by_id[positive_id]
    negative = by_id[negative_id]

    assert positive["learning_goal"] == negative["learning_goal"]
    assert positive["days"] == negative["days"]
    assert positive["expected_checks"] == {dimension: True}
    assert negative["expected_checks"] == {dimension: False}

@pytest.mark.parametrize("online", [False, True])
def test_calibration_reports_execution_stages(
    tmp_path, monkeypatch, capsys, online
):
    output = tmp_path / "progress.jsonl"
    cases = Path(__file__).parent / "fixtures/evaluation/judge_calibration_cases.jsonl"
    argv = ["calibration", "--cases", str(cases), "--output", str(output)]
    if online:
        argv.append("--llm")
    monkeypatch.setattr(sys, "argv", argv)

    # 隔离真实PDF和网络，只验证进度输出及结果保存。
    monkeypatch.setattr(
        calibration, "load_source_evidence", lambda *a, **kw: {}
    )
    monkeypatch.setattr(
        calibration, "create_llm",
        lambda: SimpleNamespace(model_name="fake"),
    )
    monkeypatch.setattr(
        calibration, "judge_plan",
        lambda *a, **kw: {
            "checks": {item: None for item in review.REVIEW_ITEMS}
        },
    )

    calibration.main()
    progress = capsys.readouterr().err

    assert "[准备]" in progress
    assert "[证据读取]" in progress
    assert "[证据完成]" in progress
    assert "[结果保存]" in progress
    assert "[全部完成]" in progress
    assert len(output.read_text(encoding="utf-8").splitlines()) == 16

    if online:
        assert "[模型评审]" in progress
        assert "[评审完成]" in progress
        assert "[最终判定]" not in progress
    else:
        assert "[离线完成]" in progress
        assert "[模型评审]" not in progress

def test_case_progress_shows_changes_without_printing_payloads(capsys):
    record = {
        "case_id": "judge-book-009",
        "expected_checks": {"citation_support": False},
        "checks": {item: False for item in review.REVIEW_ITEMS},
        "plan": "不应打印的完整计划",
        "source_evidence": {"text": "不应打印的资料正文"},
        "suggestion": {
            "rule_changes": [{
                "scope": "dimension",
                "target": "citation_support",
                "before": None,
                "after": False,
            }],
        },
    }

    calibration.show_case_result(record, 1.25, verbose=True)
    progress = capsys.readouterr().err

    assert "标注项一致=1/1" in progress
    assert "[规则变化]" in progress
    assert "未判定 → 不符合" in progress
    assert "不应打印的完整计划" not in progress
    assert "不应打印的资料正文" not in progress
