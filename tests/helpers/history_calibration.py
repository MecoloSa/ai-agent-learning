"""固定开发/验收查询的阈值诊断；默认预检，--embed编码一次，--replay本地复现。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from langchain_core.documents import Document
from ai_agent.config import create_embeddings, get_settings
from ai_agent.retrieval.semantic_history import SemanticHistoryIndex

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/evaluation/history_calibration_v1.json"
GRID = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]


def documents(fixture):
    return [Document(page_content=text, metadata={"source_index": s, "turn_number": t})
            for s, source in enumerate(fixture["sources"], 1)
            for t, text in enumerate(source["turns"][:source["through_turn"]], 1)]


def score(index, cases, threshold):
    rows = []
    for case in cases:
        hits = index.search(case["query"], k=3, min_similarity=threshold)
        positions = [[d.metadata["source_index"], d.metadata["turn_number"]] for d, _ in hits]
        rows.append({**case, "positions": positions, "scores": [v for _, v in hits],
                     "target_hit": case["target"] in positions if case["target"] else None,
                     "unsupported_candidates": bool(hits) if case["target"] is None else None})
    return {"threshold": threshold, "positive_hits": sum(r["target_hit"] is True for r in rows),
            "negative_with_candidates": sum(r["unsupported_candidates"] is True for r in rows), "rows": rows}


def choose(dev_results):
    # 预先固定：优先负例不返回候选，再最大化正例召回，同分选较低阈值。
    clean = [r for r in dev_results if r["negative_with_candidates"] == 0]
    if not clean:
        raise ValueError("候选阈值均不能通过开发集负例约束")
    return max(clean, key=lambda r: (r["positive_hits"], -r["threshold"]))["threshold"]


def evaluate(frozen):
    fixture = frozen["fixture"]
    docs = documents(fixture)
    class FrozenEmbedding:
        def embed_documents(self, texts):
            if texts != [d.page_content for d in docs]:
                raise ValueError("冻结向量与正文不一致")
            return frozen["document_vectors"]
        def embed_query(self, query):
            return frozen["query_vectors"][query]
    index = SemanticHistoryIndex(docs, FrozenEmbedding())
    dev = [score(index, fixture["dev"], threshold) for threshold in GRID]
    selected = choose(dev)
    # 验收集只用选定阈值，不参与选择或再次遍历阈值。
    return {"dev_grid": dev, "selected_threshold": selected,
            "selected_dev": next(r for r in dev if r["threshold"] == selected),
            "holdout": score(index, fixture["holdout"], selected),
            "existing_04_dev": score(index, fixture["dev"], 0.4),
            "existing_04_holdout": score(index, fixture["holdout"], 0.4)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--embed", action="store_true")
    mode.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".evaluation-results/history-calibration-v1.json")
    args = parser.parse_args(argv)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if not args.embed and not args.replay:
        print(json.dumps({"mode": "preflight_only", "fixture": fixture, "threshold_grid": GRID}, ensure_ascii=False, indent=2))
        return 0
    if args.replay:
        frozen = json.loads(args.replay.read_text(encoding="utf-8"))
        frozen_path = args.replay
    else:
        client = create_embeddings()
        started = time.monotonic()
        vectors = client.embed_documents([d.page_content for d in documents(fixture)])
        queries = {}
        for case in fixture["dev"] + fixture["holdout"]:
            queries[case["query"]] = client.embed_query(case["query"])
            print("已编码：" + case["id"], flush=True)
        frozen = {"fixture": fixture, "document_vectors": vectors, "query_vectors": queries,
                  "embedding_model": get_settings().embedding_model,
                  "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                  "encoding_ms": round((time.monotonic() - started) * 1000)}
        frozen_path = args.output.with_name(args.output.stem + "-frozen.json")
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_path.write_text(json.dumps(frozen, ensure_ascii=False), encoding="utf-8")
    result = {"assessment": "not_evaluated", "frozen_path": str(frozen_path),
              "embedding_model": frozen["embedding_model"], "fixture_sha256": frozen["fixture_sha256"],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "semantic_source_sha256": hashlib.sha256((ROOT / "src/ai_agent/retrieval/semantic_history.py").read_bytes()).hexdigest(),
              **evaluate(frozen)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_threshold": result["selected_threshold"],
                      "dev": {k: result['selected_dev'][k] for k in ('positive_hits', 'negative_with_candidates')},
                      "holdout": {k: result['holdout'][k] for k in ('positive_hits', 'negative_with_candidates')},
                      "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
