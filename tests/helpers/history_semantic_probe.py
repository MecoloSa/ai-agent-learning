"""合成历史的语义召回对照；默认预检，--embed才调用现有Embedding客户端。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from langchain_core.documents import Document
from ai_agent.config import create_embeddings, get_settings
from ai_agent.retrieval.lexical_retrieval import CharBigramBM25

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "tests/fixtures/evaluation/history_retrieval_coverage.json"


def scoped_documents(fixture):
    documents = []
    for index, source in enumerate(fixture["sources"], 1):
        cap = source["through_turn"]
        if type(cap) is not int or not 1 <= cap <= len(source["user_turns"]):
            raise ValueError("非法来源范围")
        for turn, text in enumerate(source["user_turns"][:cap], 1):
            documents.append(Document(page_content=text, metadata={"source_index": index, "turn_number": turn}))
    return documents


def cosine(left, right):
    if not left or len(left) != len(right):
        raise ValueError("向量维数无效")
    if any(not math.isfinite(v) for v in [*left, *right]):
        raise ValueError("向量含非有限数值")
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if not left_norm or not right_norm:
        raise ValueError("不能计算零向量相似度")
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def probe(fixture, embedding):
    # 这是候选检索诊断，不是生产SQLite范围过滤或历史回读的集成验收。
    # 只编码合成用户消息的可读部分，不编码助手消息或上限外消息。
    documents = scoped_documents(fixture)
    vectors = embedding.embed_documents([d.page_content for d in documents])
    if len(vectors) != len(documents):
        raise ValueError("返回的向量数量与用户消息不一致")
    lexical = CharBigramBM25(documents)
    rows = []
    for case in fixture["cases"]:
        started = time.monotonic()
        query_vector = embedding.embed_query(case["query"])
        scored = sorted(enumerate(vectors), key=lambda item: (-cosine(query_vector, item[1]), item[0]))
        dense = [{**documents[i].metadata, "rank": rank, "cosine": cosine(query_vector, vector)}
                 for rank, (i, vector) in enumerate(scored[:3], 1)]
        bm25 = [{**d.metadata, "rank": rank} for rank, d in enumerate(lexical.search(case["query"], k=3), 1)]
        targets = {tuple(t) for t in case["targets"]}
        def hit(items):
            return bool(items and (items[0]["source_index"], items[0]["turn_number"]) in targets)
        rows.append({**case, "dense_top3": dense, "bm25_top3": bm25,
                     "dense_hit_at_1": hit(dense) if targets else None,
                     "bm25_hit_at_1": hit(bm25) if targets else None,
                     "latency_ms": round((time.monotonic() - started) * 1000)})
        print("已完成查询：" + case["case_id"], flush=True)
    return {"scoped_user_turns": len(documents), "query_count": len(rows), "rows": rows,
            "warning": "向量候选未做相关性阈值筛选；负例仍可能返回候选，返回不代表有关。"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=ROOT / ".evaluation-results/history-semantic-probe-v1.json")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args(argv)
    fixture = json.loads(args.cases.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != "history-retrieval-coverage-v1":
        raise ValueError("不支持的案例格式")
    if not args.embed:
        print(json.dumps({"mode": "preflight_only", "fixture": fixture,
                          "scoped_user_turns": len(scoped_documents(fixture))}, ensure_ascii=False, indent=2))
        return 0
    started = time.monotonic()
    result = probe(fixture, create_embeddings())
    source_hash = hashlib.sha256()
    for relative in ("tests/helpers/history_semantic_probe.py", "src/ai_agent/config.py",
                     "src/ai_agent/retrieval/embeddings.py", "src/ai_agent/retrieval/lexical_retrieval.py"):
        source_hash.update(relative.encode("utf-8"))
        source_hash.update((ROOT / relative).read_bytes())
    result.update(mode="online_embedding_probe", assessment="not_evaluated",
                  embedding_model=get_settings().embedding_model,
                  cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                  source_py_sha256=source_hash.hexdigest(), total_latency_ms=round((time.monotonic() - started) * 1000))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
