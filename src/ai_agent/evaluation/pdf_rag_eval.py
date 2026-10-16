"""
PDF 资料检索评测：评估 RAG 系统对给定问题的页级召回质量。

1. load_retrieval_cases：从 JSONL 加载测试案例（含问题与金标准证据页）；
2. evaluate_case：对单案例执行检索，页级去重后计算 recall@k 与 reciprocal rank；
3. main：CLI 入口，同步向量索引 → 批量评测 → 输出 JSONL 结果及汇总指标。
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document

from ai_agent.config import create_llm, get_settings, resolve_project_path
from ai_agent.evaluation.answer_pair_probe import (
    run_answer_pair,
    summarize_answer_pairs,
)
from ai_agent.rag import LearningRAG
from ai_agent.evaluation.metrics import recall_at_k, reciprocal_rank
from ai_agent.evaluation.lexical_probe import (
    CharBigramBM25,
    rrf_candidate_documents,
)
from ai_agent.evaluation.rerank_probe import (
    RerankOutcome,
    rerank_candidates,
    configured_rerank_options,
)
# 这个版本专门用于你指定的 PDF 书籍目录，因此要求命中有 page。
# 以后评测 .md/.py 时再增加相应证据 ID 规则。
def page_id(document: Document) -> str:
    """把命中的 Document 转为稳定的页级证据 ID。"""
    source = document.metadata["source"]
    page = document.metadata["page"]

    # 使用页码而不是 chunk 编号，避免切分参数改变后金标准失效。
    return f"{source}#page={page}"

def load_retrieval_cases(path: Path) -> list[dict]:
    cases = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            case = json.loads(line)
            if not isinstance(case["question"], str):
                raise ValueError("question 必须是字符串")
            if not case["gold_evidence"]:
                raise ValueError("gold_evidence 不能为空")
            case["case_id"]
            alternatives = case.get("acceptable_evidence", [])
            if not isinstance(alternatives, list):
                raise ValueError("acceptable_evidence 必须是列表")

            for item in alternatives:
                if not isinstance(item, dict):
                    raise ValueError("替代证据必须是对象")
                source = item.get("source")
                pages = item.get("pages")
                if (
                    not isinstance(source, str) or not source
                    or not isinstance(pages, list) or not pages
                    or any(type(page) is not int or page < 1 for page in pages)
                ):
                    raise ValueError("替代证据需要 source 和正整数页码")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{line_number} 检索案例格式错误：{exc}"
            ) from exc

        cases.append(case)

    if not cases:
        raise ValueError("检索案例文件不能为空")

    return cases

def evaluate_case(
    rag: LearningRAG,
    case: dict,
    k: int,
    *,
    lexical: CharBigramBM25 | None = None,
    reranker: Callable[[str, list[Document]], RerankOutcome] | None = None,
    rerank_candidate_limit: int = 64,
    answer_probe: Callable[
        [str, list[Document], list[Document]], dict
    ] | None = None,
) -> dict:
    settings = get_settings()
    started = time.perf_counter()

    # 诊断与命中来自同一次查询，不额外调用 Embedding 服务。
    # 本评测只覆盖单条查询的直接召回，不等于完整在线 Agent 轨迹。
    candidate_audit: list[dict[str, object]] = []
    hits = rag.retrieve(
        query=case["question"],
        k=k,
        min_relevance=settings.rag_min_relevance,
        relative_margin=settings.rag_relative_margin,
        min_hits=settings.rag_min_hits_per_query,
        candidate_audit=candidate_audit,
    )
    retrieval_ms = round((time.perf_counter() - started) * 1000)

    # 保留检索顺序，并在页级去重。
    ranked_ids = list(dict.fromkeys(
        page_id(document)
        for document in hits
    ))

    # 金标准也转换成相同格式，才能比较两边是否一致。
    gold_ids = {
        f"{item['source']}#page={page}"
        for item in case["gold_evidence"]
        for page in item["pages"]
    }

    # 旧金标准保持不变；新集合只用于“任一充分证据页”的单独指标。
    # 不把替代页混入 recall@k 的分母，否则会误把“或”算成“且”。
    acceptable_ids = gold_ids | {
        f"{item['source']}#page={page}"
        for item in case.get("acceptable_evidence", [])
        for page in item["pages"]
    }

    probe: dict[str, object] = {}
    if lexical is not None:
        probe_started = time.perf_counter()
        sparse_hits = lexical.search(case["question"], k=32)
        sparse_ids = list(dict.fromkeys(
            page_id(document) for document in sparse_hits
        ))[:k]
        candidates = rrf_candidate_documents(
            candidate_audit,
            sparse_hits,
            documents=lexical.documents,
        )

        # dict.fromkeys 保留首次出现的顺序，并去掉同页重复片段。
        candidate_page_ids = list(dict.fromkeys(
            page_id(document) for document in candidates
        ))
        hybrid_ids = candidate_page_ids[:k]

        # 候选覆盖率是固定候选池下的召回上限；不是最终前 k 的质量。
        # 金标准只参与评分，不能传入候选生成或未来的重排模型。
        candidate_recall = recall_at_k(
            candidate_page_ids,
            gold_ids,
            max(1, len(candidate_page_ids)),
        )

        # 保留现有向量结果作基线；不能只报告修复的失败案例。
        probe = {
            "sparse_ranked_ids": sparse_ids,
            "hybrid_ranked_ids": hybrid_ids,
            "sparse_recall_at_k": recall_at_k(
                sparse_ids, gold_ids, k
            ),
            "hybrid_recall_at_k": recall_at_k(
                hybrid_ids, gold_ids, k
            ),
            "hybrid_rr_at_k": reciprocal_rank(
                hybrid_ids, gold_ids
            ),
            "hybrid_acceptable_rr_at_k": reciprocal_rank(
                hybrid_ids, acceptable_ids
            ),
            "hybrid_candidate_count": len(candidates),
            "hybrid_candidate_page_count": len(candidate_page_ids),
            "hybrid_candidate_page_ids": candidate_page_ids,
            "hybrid_candidate_recall": candidate_recall,
            "hybrid_hit_at_1": bool(
                set(candidate_page_ids[:1]) & gold_ids
            ),
            "hybrid_hit_at_3": bool(
                set(candidate_page_ids[:3]) & gold_ids
            ),
            "hybrid_probe_ms": round(
                (time.perf_counter() - probe_started) * 1000
            ),
        }

        logging.getLogger(__name__).info(
            "重排候选准备：chunks=%d pages=%d "
            "pool_recall=%.3f hit@1=%s hit@3=%s",
            len(candidates),
            len(candidate_page_ids),
            candidate_recall,
            probe["hybrid_hit_at_1"],
            probe["hybrid_hit_at_3"],
        )

    if reranker is not None:
        # 截止位置固定，前后两种排序都使用同一批候选。
        # 动态评估：同时观察输入数、成功率、页级 recall/MRR、耗时和 token。
        pool = candidates[:rerank_candidate_limit]

        # 截取是按片段进行的；先转换成页，才能判断正确证据页是否仍在池中。
        pool_page_ids = list(dict.fromkeys(
            page_id(doc) for doc in pool
        ))
        baseline_ids = pool_page_ids[:k]

        probe.update({
            "rerank_input_count": len(pool),
            "rerank_pool_page_count": len(pool_page_ids),
            # 动态评估：小候选池若丢失正确页，任何重排模型都无法补救。
            "rerank_pool_recall": recall_at_k(
                pool_page_ids, gold_ids, max(1, len(pool_page_ids))
            ),
            "rerank_baseline_ranked_ids": baseline_ids,
            "rerank_baseline_recall_at_k": recall_at_k(
                baseline_ids, gold_ids, k
            ),
            "rerank_baseline_rr_at_k": reciprocal_rank(
                baseline_ids, gold_ids
            ),
            "rerank_baseline_acceptable_rr_at_k": reciprocal_rank(
                baseline_ids, acceptable_ids
            ),
            "rerank_acceptable_rr_at_k": None,
            "rerank_ranked_ids": None,
            # 仅保留每个入选页对应的首个片段，便于人工核对“页对了、内容是否也对”。
            "rerank_top_evidence": None,
            "rerank_recall_at_k": None,
            "rerank_rr_at_k": None,
            "rerank_total_tokens": None,
            "rerank_request_id": None,
            "rerank_error_type": None,
            "rerank_ms": None,
        })

        if not pool:
            probe["rerank_status"] = "skipped_empty"
        else:
            rerank_started = time.perf_counter()
            try:
                outcome = reranker(case["question"], pool)

                top_evidence: list[dict[str, object]] = []
                seen_pages: set[str] = set()
                for doc in outcome.documents:
                    evidence_page_id = page_id(doc)
                    if evidence_page_id in seen_pages:
                        continue

                    # 重排的是片段，评分却按页计算；首次出现的片段代表该页的排序位置。
                    seen_pages.add(evidence_page_id)
                    top_evidence.append({
                        "page_id": evidence_page_id,
                        "chunk_index": doc.metadata.get("chunk_index"),
                        "rerank_score": doc.metadata.get("rerank_score"),
                        "rerank_rank": doc.metadata.get("rerank_rank"),
                        # 只保存短预览供本地核对，不把整本资料正文写进评测记录。
                        "preview": " ".join(doc.page_content.split())[:120],
                    })
                    if len(top_evidence) >= k:
                        break

                reranked_ids = [item["page_id"] for item in top_evidence]
                probe.update({
                    "rerank_status": "succeeded",
                    "rerank_ranked_ids": reranked_ids,
                    "rerank_top_evidence": top_evidence,
                    "rerank_recall_at_k": recall_at_k(
                        reranked_ids, gold_ids, k
                    ),
                    "rerank_rr_at_k": reciprocal_rank(
                        reranked_ids, gold_ids
                    ),
                    # 与旧口径并列：差距提示原标注可能遗漏了可独立作答的页。
                    "rerank_acceptable_rr_at_k": reciprocal_rank(
                        reranked_ids, acceptable_ids
                    ),
                    "rerank_total_tokens": outcome.total_tokens,
                    "rerank_request_id": outcome.request_id,
                    "rerank_backend": outcome.backend,
                    "rerank_completed_missing_count": outcome.completed_missing_count,
                })
            except Exception as exc:
                # 失败不能伪装成“重排后等于 RRF”；记录类型供排查。
                # 不记录异常正文，避免接口错误带出密钥或资料内容。
                probe.update({
                    "rerank_status": "failed",
                    "rerank_error_type": type(exc).__name__,
                })
            finally:
                probe["rerank_ms"] = round(
                    (time.perf_counter() - rerank_started) * 1000
                )

    if answer_probe is not None:
        if probe.get("rerank_status") != "succeeded":
            probe["answer_pair"] = {"status": "skipped_rerank_failed"}
        else:
            try:
                # pool 与 outcome.documents 含同一批片段，仅顺序不同。
                # 回答失败单独记录，不能把成功的重排误报为失败。
                probe["answer_pair"] = {
                    "status": "succeeded",
                    **answer_probe(
                        case["question"], pool, outcome.documents
                    ),
                }
            except Exception as exc:
                probe["answer_pair"] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }

    gold_diagnostics = {}
    for gold_id in sorted(gold_ids):
        rows = [
            row
            for row in candidate_audit
            if row["page"] is not None
            and f"{row['source']}#page={row['page']}" == gold_id
        ]
        gold_diagnostics[gold_id] = {
            "best_raw_rank": min(
                (int(row["raw_rank"]) for row in rows),
                default=None,
            ),
            "best_raw_score": max(
                (float(row["score"]) for row in rows),
                default=None,
            ),
            "decisions": sorted(
                {str(row["decision"]) for row in rows}
            ),
        }

    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "k": k,
        "ranked_ids": ranked_ids,
        "gold_ids": sorted(gold_ids),
        "acceptable_ids": sorted(acceptable_ids),
        "raw_candidate_count": len(candidate_audit),
        "gold_diagnostics": gold_diagnostics,
        "hit_count": len(ranked_ids),
        "recall_at_k": recall_at_k(ranked_ids, gold_ids, k),
        "rr_at_k": reciprocal_rank(ranked_ids, gold_ids),
        # 目前只评估任一页可独立作答；多页联合证据要另设计证据组。
        "acceptable_rr_at_k": reciprocal_rank(
            ranked_ids, acceptable_ids
        ),
        "retrieval_ms": retrieval_ms,
        **probe,
    }

def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="评测 PDF 资料的页级直接召回"
    )
    parser.add_argument("--materials", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细调用、检索与预算日志")
    parser.add_argument(
        "--compare-hybrid",
        action="store_true",
        help="同时评测关键词检索和 RRF 融合；不修改线上检索策略",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=settings.rag_rerank_enabled,
        help="默认随配置启用重排及融合对照；--no-rerank 恢复纯向量评测",
    )
    parser.add_argument("--rerank-backend", choices=["chat", "dashscope"],
                        default=settings.rag_rerank_backend,
                        help="chat 默认用 DeepSeek；dashscope 使用专用付费重排")
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="覆盖当前后端模型；不填时继承相应环境配置",
    )
    parser.add_argument(
        "--rerank-candidate-limit",
        type=int,
        default=settings.rag_rerank_candidate_limit,
    )
    parser.add_argument(
        "--answer-pair",
        action="store_true",
        help="用相同问题和预算生成 RRF/rerank 两份回答；需同时开启重排",
    )
    parser.add_argument(
        "--rerank-base-url",
        default=None,
        help="覆盖选中后端地址：chat 用 OpenAI兼容根地址，dashscope 用 DashScope API 根地址",
    )
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k 必须大于 0")
    if args.rerank:
        # 重排需要融合候选；自动开启对照，但顶层向量指标仍保留原基线。
        args.compare_hybrid = True
    if args.answer_pair and not args.rerank:
        parser.error("--answer-pair 需要同时指定 --compare-hybrid --rerank")
    if args.rerank_candidate_limit < args.k:
        parser.error("--rerank-candidate-limit 不能小于 --k")
    rerank_options = None
    if args.rerank:
        try:
            rerank_options = configured_rerank_options(settings, backend=args.rerank_backend,
                model=args.rerank_model, base_address=args.rerank_base_url)
        except ValueError as exc:
            parser.error(str(exc))
        if not rerank_options["api_key"] or not rerank_options["base_address"]:
            parser.error("所选重排后端缺少 API Key 或服务地址")

    from ai_agent.console_logging import configure_console_logging
    configure_console_logging(verbose=args.verbose, log_level=settings.log_level)

    material_dir = resolve_project_path(args.materials)
    case_path = resolve_project_path(args.cases)
    output_path = resolve_project_path(args.output)

    cases = load_retrieval_cases(case_path)
    rag = LearningRAG(str(material_dir))

    # 首次同步可能需要对两本书分块并调用 Embedding，单独计时。
    # 动态评估：索引耗时与查询耗时分开记录，避免冷启动影响比较。
    started = time.perf_counter()
    rag.sync()
    index_sync_ms = round((time.perf_counter() - started) * 1000)

    lexical = None
    if args.compare_hybrid:
        stored = rag.store.get(
            include=["documents", "metadatas"]
        )
        pdf_documents = [
            Document(
                page_content=content,
                metadata=dict(metadata),
            )
            for content, metadata in zip(
                stored["documents"] or [],
                stored["metadatas"] or [],
            )
            if content and metadata and metadata.get("page") is not None
        ]
        # 实验只建一次本地词项表；不调用额外 Embedding。
        lexical = CharBigramBM25(pdf_documents)

    reranker = None
    if args.rerank:
        def reranker(query: str, candidates: list[Document]) -> RerankOutcome:
            return rerank_candidates(
                query,
                candidates,
                **rerank_options,
            )

    answer_probe = None
    if args.answer_pair:
        answer_llm = create_llm(role="standard")

        def answer_probe(
            question: str,
            rrf_pool: list[Document],
            reranked: list[Document],
        ) -> dict:
            return run_answer_pair(
                question,
                rrf_pool,
                reranked,
                rag=rag,
                llm=answer_llm,
            )

    records = [
        evaluate_case(
            rag,
            case,
            args.k,
            lexical=lexical,
            reranker=reranker,
            rerank_candidate_limit=args.rerank_candidate_limit,
            answer_probe=answer_probe,
        )
        for case in cases
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in records
        ) + "\n",
        encoding="utf-8",
    )

    summary = {
        "case_count": len(records),
        "k": args.k,
        "mean_recall_at_k": (
            sum(r["recall_at_k"] for r in records) / len(records)
        ),
        "mrr_at_k": (
            sum(r["rr_at_k"] for r in records) / len(records)
        ),
        "acceptable_mrr_at_k": (
            sum(r["acceptable_rr_at_k"] for r in records) / len(records)
        ),
        "index_sync_ms": index_sync_ms,
        "results_path": str(output_path),
    }

    if args.answer_pair:
        summary["answer_pair"] = summarize_answer_pairs(records)

    if args.compare_hybrid:
        # 顶层原有指标仍表示向量基线；融合指标单独展示。
        summary["hybrid"] = {
            "mean_candidate_recall": (
                    sum(r["hybrid_candidate_recall"] for r in records)
                    / len(records)
            ),
            "hit_at_1": (
                    sum(r["hybrid_hit_at_1"] for r in records)
                    / len(records)
            ),
            "hit_at_3": (
                    sum(r["hybrid_hit_at_3"] for r in records)
                    / len(records)
            ),
            "mean_recall_at_k": (
                    sum(r["hybrid_recall_at_k"] for r in records)
                    / len(records)
            ),
            "mrr_at_k": (
                    sum(r["hybrid_rr_at_k"] for r in records)
                    / len(records)
            ),
            "acceptable_mrr_at_k": (
                    sum(r["hybrid_acceptable_rr_at_k"] for r in records)
                    / len(records)
            ),
        }

    if args.rerank:
        paired = [
            row for row in records
            if row["rerank_status"] == "succeeded"
        ]

        def paired_mean(key: str) -> float | None:
            return (
                sum(row[key] for row in paired) / len(paired)
                if paired else None
            )

        # 只对调用成功的相同案例做配对比较；失败另计。
        summary["rerank"] = {
            "model": rerank_options["model"],
            "backend": rerank_options["backend"],
            "candidate_limit": args.rerank_candidate_limit,
            # 所有案例都计入候选覆盖率，接口失败也不掩盖截取造成的丢失。
            "mean_pool_recall": (
                sum(row["rerank_pool_recall"] for row in records)
                / len(records)
            ),
            "mean_input_count": (
                sum(row["rerank_input_count"] for row in records)
                / len(records)
            ),
            "succeeded_cases": len(paired),
            "failed_cases": sum(
                row["rerank_status"] == "failed" for row in records
            ),
            "skipped_empty_cases": sum(
                row["rerank_status"] == "skipped_empty"
                for row in records
            ),
            "same_pool_rrf_recall_at_k": paired_mean(
                "rerank_baseline_recall_at_k"
            ),
            "rerank_recall_at_k": paired_mean("rerank_recall_at_k"),
            "same_pool_rrf_mrr_at_k": paired_mean(
                "rerank_baseline_rr_at_k"
            ),
            "rerank_mrr_at_k": paired_mean("rerank_rr_at_k"),
            # 同一批成功案例、同一候选池比较，避免失败改变两侧分母。
            "same_pool_rrf_acceptable_mrr_at_k": paired_mean(
                "rerank_baseline_acceptable_rr_at_k"
            ),
            "rerank_acceptable_mrr_at_k": paired_mean(
                "rerank_acceptable_rr_at_k"
            ),
            "mean_rerank_ms": paired_mean("rerank_ms"),
            "total_rerank_tokens": sum(
                row["rerank_total_tokens"] or 0 for row in paired
            ),
            "token_usage_complete": all(
                row["rerank_total_tokens"] is not None for row in paired
            ),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
