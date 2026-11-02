"""历史检索策略对照。默认预检；--embed冻结在线向量；--replay完全本地重放。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from langchain_core.documents import Document
from ai_agent.config import create_embeddings, get_settings
from ai_agent.retrieval.lexical_retrieval import CharBigramBM25
from ai_agent.retrieval.retrieval_fusion import fuse_rrf_candidates

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES = [ROOT / "tests/fixtures/evaluation" / name for name in
                    ("history_retrieval_coverage.json", "history_semantic_followup.json")]


def documents_from_fixture(fixture):
    documents = []
    for source_index, source in enumerate(fixture["sources"], 1):
        cap = source["through_turn"]
        if type(cap) is not int or not 1 <= cap <= len(source["user_turns"]):
            raise ValueError("非法范围")
        for turn_number, text in enumerate(source["user_turns"][:cap], 1):
            # 仅适配RRF的位置契约；source是实验内的位置标识，不是可打开的路径。
            documents.append(Document(page_content=text, metadata={
                "source_index": source_index, "turn_number": turn_number,
                "source": f"history-source-{source_index}", "chunk_index": turn_number - 1,
            }))
    return documents


def unit(vector):
    values = tuple(float(v) for v in vector)
    length = math.hypot(*values)
    if not values or not all(math.isfinite(v) for v in values) or not math.isfinite(length) or not length:
        raise ValueError("非法向量")
    return tuple(v / length for v in values)


def compare(documents, document_vectors, query_vector, query, *, dense_min=None):
    if len(documents) != len(document_vectors):
        raise ValueError("消息和向量数量不一致")
    lexical = CharBigramBM25(documents).search(query, k=5)
    q = unit(query_vector)
    scored = []
    for index, vector in enumerate(document_vectors):
        v = unit(vector)
        if len(v) != len(q):
            raise ValueError("向量维数不一致")
        score = sum(a * b for a, b in zip(q, v))
        if dense_min is None or score >= dense_min:
            scored.append((index, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    dense = [(documents[index], score) for index, score in scored[:5]]
    # 复用项目现有融合算法；融合分数不承担相关性或空结果判断。
    hybrid = fuse_rrf_candidates(dense, lexical, limit=3)
    policies = {"bm25": lexical[:3], "dense": [d for d, _ in dense[:3]],
                "hybrid_rrf": hybrid,
                "empty_fallback": lexical[:3] if lexical else [d for d, _ in dense[:3]]}
    positions = {name: [{"source_index": d.metadata["source_index"], "turn_number": d.metadata["turn_number"]}
                        for d in values] for name, values in policies.items()}
    return {"policies": positions, "lexical_nonempty": bool(lexical),
            "dense_scores": [{"source_index": d.metadata["source_index"], "turn_number": d.metadata["turn_number"],
                              "cosine": score} for d, score in dense],
            "query_embedding_needed": {"bm25": False, "dense": True, "hybrid_rrf": True,
                                       "empty_fallback": not bool(lexical)}}


def replay(frozen):
    documents = [Document(**d) for d in frozen["documents"]]
    rows = []
    for item in frozen["queries"]:
        case = item["case"]
        variants = []
        # 这是灵敏度观察，不据此选择生产阈值；原始无筛选结果单独保留。
        for threshold in (None, 0.3, 0.4, 0.5):
            outcome = compare(documents, frozen["document_vectors"], item["vector"], case["query"], dense_min=threshold)
            targets = {tuple(t) for t in case["targets"]}
            metrics = {}
            for policy, hits in outcome["policies"].items():
                positions = [(h["source_index"], h["turn_number"]) for h in hits]
                metrics[policy] = {"hit_at_1": bool(positions and positions[0] in targets) if targets else None,
                                   "hit_at_3": bool(set(positions) & targets) if targets else None,
                                   "negative_empty": not positions if not targets else None}
            variants.append({"dense_min": threshold, **outcome, "metrics": metrics})
        rows.append({**case, "fixture": item["fixture"], "query_embedding_ms": item["query_embedding_ms"],
                     "variants": variants})
    return {"embedding_model": frozen["embedding_model"], "documents": len(documents),
            "frozen_query_embedding_count": len(rows), "document_embedding_ms": frozen["document_embedding_ms"],
            "rows": rows, "assessment": "not_evaluated"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--embed", action="store_true")
    mode.add_argument("--replay", type=Path)
    parser.add_argument("--rerank", action="store_true", help="仅配合--replay，对已公布的两题执行现有聊天重排")
    parser.add_argument("--output", type=Path, default=ROOT / ".evaluation-results/history-strategy-comparison-v1.json")
    args = parser.parse_args(argv)
    if args.rerank and not args.replay:
        parser.error("--rerank需要--replay，避免无意重复生成向量")
    if args.replay:
        frozen = json.loads(args.replay.read_text(encoding="utf-8"))
        frozen_path = args.replay.resolve()
    else:
        fixtures = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in DEFAULT_FIXTURES]
        documents = documents_from_fixture(fixtures[0][1])
        if any(documents_from_fixture(f) != documents for _, f in fixtures):
            raise ValueError("对照资料不一致，不能共享文档向量")
        if not args.embed:
            print(json.dumps({"mode": "preflight_only", "scoped_user_messages": len(documents),
                              "fixtures": [{"name": p.name, "data": f} for p, f in fixtures]}, ensure_ascii=False, indent=2))
            return 0
        embedding = create_embeddings()
        started = time.monotonic()
        document_vectors = embedding.embed_documents([d.page_content for d in documents])
        document_ms = round((time.monotonic() - started) * 1000)
        queries = []
        for path, fixture in fixtures:
            for case in fixture["cases"]:
                started = time.monotonic()
                vector = embedding.embed_query(case["query"])
                queries.append({"fixture": path.name, "case": case, "vector": vector,
                                "query_embedding_ms": round((time.monotonic() - started) * 1000)})
                print("已编码：" + case["case_id"], flush=True)
        frozen = {"embedding_model": get_settings().embedding_model,
                  "documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in documents],
                  "document_vectors": document_vectors, "document_embedding_ms": document_ms,
                  "queries": queries, "fixtures_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in fixtures}}
        frozen_path = args.output.parent / "history-strategy-artifacts" / uuid4().hex / "frozen.json"
        frozen_path.parent.mkdir(parents=True)
        frozen_path.write_text(json.dumps(frozen, ensure_ascii=False), encoding="utf-8")
    result = replay(frozen)
    result["query_embedding_client_calls_this_run"] = 0 if args.replay else len(frozen["queries"])
    if args.rerank:
        from ai_agent.retrieval.reranking import configured_rerank_options, rerank_candidates
        documents = [Document(**d) for d in frozen["documents"]]
        by_position = {(d.metadata["source_index"], d.metadata["turn_number"]): d for d in documents}
        options = configured_rerank_options(get_settings(), backend="chat")
        rerank_rows = []
        for item in frozen["queries"]:
            case = item["case"]
            if case["case_id"] not in {"semantic-followup-001", "coverage-010"}:
                continue
            outcome = compare(documents, frozen["document_vectors"], item["vector"], case["query"])
            dense = [(by_position[(h["source_index"], h["turn_number"])], h["cosine"]) for h in outcome["dense_scores"]]
            lexical = CharBigramBM25(documents).search(case["query"], k=5)
            candidates = fuse_rrf_candidates(dense, lexical, limit=5)
            started = time.monotonic()
            row = {"case_id": case["case_id"], "query": case["query"], "targets": case["targets"],
                   "candidate_positions": [{"source_index": d.metadata["source_index"], "turn_number": d.metadata["turn_number"]} for d in candidates]}
            try:
                ordered = rerank_candidates(case["query"], candidates, **options)
                row.update(status="completed", positions=[{"source_index": d.metadata["source_index"], "turn_number": d.metadata["turn_number"]} for d in ordered.documents],
                           total_tokens=ordered.total_tokens, backend=ordered.backend, model=options["model"])
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__)
            row["latency_ms"] = round((time.monotonic() - started) * 1000)
            rerank_rows.append(row)
            print("已重排：" + case["case_id"], flush=True)
        result["chat_rerank_probe"] = rerank_rows
    result.update(frozen_path=str(frozen_path), mode="offline_replay" if args.replay else "online_embedding_then_local_comparison")
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["dependency_sha256"] = {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for relative in (
        "src/ai_agent/retrieval/lexical_retrieval.py", "src/ai_agent/retrieval/retrieval_fusion.py",
        "src/ai_agent/retrieval/reranking.py", "src/ai_agent/retrieval/embeddings.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "frozen_path": str(frozen_path), "queries": len(result["rows"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
