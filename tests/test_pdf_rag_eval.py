"""PDF 页级诊断测试：区分原标注与可独立回答的替代页，不连接模型。"""

from pathlib import Path

import pytest

from langchain_core.documents import Document
from ai_agent.evaluation.pdf_rag_eval import (
    evaluate_case,
    load_retrieval_cases,
)
from ai_agent.evaluation.rerank_probe import RerankOutcome

def test_pdf_eval_reports_gold_missing_from_raw_candidates():
    class EmptyRAG:
        def retrieve(self, **kwargs):
            assert kwargs["candidate_audit"] == []
            return []

    case = {
        "case_id": "missing-page",
        "question": "第四部分是什么？",
        "gold_evidence": [
            {"source": "book.pdf", "pages": [4]}
        ],
    }

    record = evaluate_case(EmptyRAG(), case, k=8)

    assert record["ranked_ids"] == []
    assert record["raw_candidate_count"] == 0
    assert record["gold_diagnostics"]["book.pdf#page=4"] == {
        "best_raw_rank": None,
        "best_raw_score": None,
        "decisions": [],
    }

def test_pdf_eval_separates_pool_coverage_from_top_k():
    """正确页在候选池中但排第二：候选覆盖成功，前一名选择失败。"""
    documents = [
        Document(
            page_content="distractor",
            metadata={"source": "book.pdf", "page": 9, "chunk_index": 1},
        ),
        Document(
            page_content="target",
            metadata={"source": "book.pdf", "page": 4, "chunk_index": 2},
        ),
    ]

    class FixedRAG:
        def retrieve(self, **kwargs):
            kwargs["candidate_audit"].extend([
                {
                    "source": "book.pdf", "page": 9, "chunk_index": 1,
                    "raw_rank": 1, "score": 0.8, "decision": "selected",
                },
                {
                    "source": "book.pdf", "page": 4, "chunk_index": 2,
                    "raw_rank": 13, "score": 0.4, "decision": "reject_k_limit",
                },
            ])
            return [documents[0]]

    class FixedLexical:
        def __init__(self):
            self.documents = documents

        def search(self, query, *, k):
            return documents

    record = evaluate_case(
        FixedRAG(),
        {
            "case_id": "pool-vs-rank",
            "question": "question",
            "gold_evidence": [{"source": "book.pdf", "pages": [4]}],
            "acceptable_evidence": [{"source": "book.pdf", "pages": [9]}],
        },
        k=1,
        lexical=FixedLexical(),
    )

    assert record["hybrid_candidate_recall"] == 1.0
    assert record["hybrid_candidate_count"] == 2
    assert record["hybrid_recall_at_k"] == 0.0
    # 旧指标仍判未命中原页，新指标承认人工标注的替代页。
    assert record["hybrid_acceptable_rr_at_k"] == 1.0
    assert record["acceptable_ids"] == [
        "book.pdf#page=4", "book.pdf#page=9"
    ]
    assert record["hybrid_hit_at_1"] is False
    assert record["hybrid_hit_at_3"] is True

    def fake_reranker(query, candidates):
        assert query == "question"
        assert len(candidates) == 2
        return RerankOutcome(
            documents=[candidates[1], candidates[0]],
            total_tokens=12,
            request_id="offline-test",
        )

    reranked = evaluate_case(
        FixedRAG(),
        {
            "case_id": "pool-vs-rank",
            "question": "question",
            "gold_evidence": [{"source": "book.pdf", "pages": [4]}],
            "acceptable_evidence": [{"source": "book.pdf", "pages": [9]}],
        },
        k=1,
        lexical=FixedLexical(),
        reranker=fake_reranker,
        answer_probe=lambda question, rrf_pool, reordered: {
            "question": question,
            "rrf_first": rrf_pool[0].page_content,
            "rerank_first": reordered[0].page_content,
        },
    )
    assert reranked["rerank_status"] == "succeeded"
    assert reranked["answer_pair"] == {
        "status": "succeeded",
        "question": "question",
        "rrf_first": "distractor",
        "rerank_first": "target",
    }
    assert reranked["rerank_baseline_recall_at_k"] == 0.0
    assert reranked["rerank_baseline_rr_at_k"] == 0.0
    assert reranked["rerank_baseline_acceptable_rr_at_k"] == 1.0
    assert reranked["rerank_recall_at_k"] == 1.0
    assert reranked["rerank_acceptable_rr_at_k"] == 1.0
    assert reranked["rerank_pool_recall"] == 1.0
    # 页级命中之外，还应能定位实际被重排到首位的片段。
    top_evidence = reranked["rerank_top_evidence"]
    assert top_evidence[0]["page_id"] == "book.pdf#page=4"
    assert top_evidence[0]["chunk_index"] == 2
    assert top_evidence[0]["preview"] == "target"
    assert len(top_evidence[0]["preview"]) <= 120

    # 正确页虽在完整 RRF 池中，但上限为 1 时被截掉。
    truncated = evaluate_case(
        FixedRAG(),
        {
            "case_id": "pool-vs-rank",
            "question": "question",
            "gold_evidence": [{"source": "book.pdf", "pages": [4]}],
        },
        k=1,
        lexical=FixedLexical(),
        rerank_candidate_limit=1,
        reranker=lambda query, candidates: RerankOutcome(
            documents=candidates,
            total_tokens=3,
            request_id="offline-truncated",
        ),
    )
    assert truncated["hybrid_candidate_recall"] == 1.0
    assert truncated["rerank_pool_recall"] == 0.0
    assert truncated["rerank_recall_at_k"] == 0.0


    def failed_reranker(query, candidates):
        raise RuntimeError("offline simulated failure")

    failed = evaluate_case(
        FixedRAG(),
        {
            "case_id": "pool-vs-rank",
            "question": "question",
            "gold_evidence": [{"source": "book.pdf", "pages": [4]}],
        },
        k=1,
        lexical=FixedLexical(),
        reranker=failed_reranker,
    )
    assert failed["rerank_status"] == "failed"
    assert failed["rerank_recall_at_k"] is None
    assert failed["rerank_acceptable_rr_at_k"] is None
    assert failed["hybrid_recall_at_k"] == 0.0


def test_pdf_eval_rejects_invalid_alternative_page(tmp_path: Path):
    """bool 属于 int 的子类，但不能被当成 PDF 第 1 页。"""
    cases_path = tmp_path / "bad_alternative.jsonl"
    cases_path.write_text(
        '{"case_id":"bad","question":"q",'
        '"gold_evidence":[{"source":"book.pdf","pages":[4]}],'
        '"acceptable_evidence":[{"source":"book.pdf","pages":[true]}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="替代证据需要"):
        load_retrieval_cases(cases_path)


def test_audited_book_cases_are_not_accidentally_removed():
    """固定实验保留原 15 例，后续仍允许继续扩充。"""
    path = (
        Path(__file__).parent
        / "fixtures/evaluation/book_retrieval_cases_audited_v2.jsonl"
    )
    cases = load_retrieval_cases(path)
    ids = {case["case_id"] for case in cases}

    assert len(cases) >= 15
    assert len(ids) == len(cases)
    assert {
        "book-rag-part4-actual-title",
        "book-rag-failure-attribution-toc",
        "book-rag-langgraph-query-state",
    } <= ids

def test_holdout_cases_are_valid_and_separate_from_development_cases():
    """留出题用于检验泛化，不能与此前反复调参的题目混在一起。"""
    fixture_dir = Path(__file__).parent / "fixtures/evaluation"
    holdout = load_retrieval_cases(
        fixture_dir / "book_retrieval_holdout_v1.jsonl"
    )
    development = load_retrieval_cases(
        fixture_dir / "book_retrieval_cases_audited_v2.jsonl"
    )

    holdout_ids = [case["case_id"] for case in holdout]
    development_ids = {case["case_id"] for case in development}

    # 至少保留当前三道起始题；以后扩充题集无需改动这个测试。
    assert len(holdout) >= 3
    assert len(holdout_ids) == len(set(holdout_ids))
    assert not (set(holdout_ids) & development_ids)

    for case in holdout:
        assert case["question"].strip()
        for evidence in case["gold_evidence"]:
            assert evidence["source"] in {
                "AI-Agents-in-Depth-zh-CN.pdf",
                "Hello-Agents-V1.0.3-20260717.pdf",
            }
            # bool 在 Python 中属于 int 的子类，因此用 type 排除 True/False。
            assert evidence["pages"]
            assert all(
                type(page) is int and page > 0
                for page in evidence["pages"]
            )

def test_answer_summary_preserves_unknown_usage_and_pair_failures():
    """失败配对不混入用量，缺失 token 不伪装成零。"""
    from ai_agent.evaluation.answer_pair_probe import summarize_answer_pairs

    records = [
        {
            "answer_pair": {
                "status": "succeeded",
                "rrf": {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                },
                "rerank": {"usage": None},
            },
        },
        {"answer_pair": {"status": "failed"}},
        {"answer_pair": {"status": "skipped_rerank_failed"}},
        {},  # 兼容未启用回答实验的历史记录。
    ]

    summary = summarize_answer_pairs(records)

    assert summary["succeeded_cases"] == 1
    assert summary["failed_cases"] == 1
    assert summary["skipped_cases"] == 1
    assert summary["not_recorded_cases"] == 1
    assert summary["rrf_chat_usage"]["total_tokens"] == 15
    assert summary["rerank_chat_usage"]["total_tokens"] is None
    assert summary["rerank_chat_usage"]["usage_complete"] is False
    assert summary["quality_assessment"] == "not_computed"