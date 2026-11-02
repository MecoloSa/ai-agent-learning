"""固定查询的历史检索覆盖诊断；默认预检，--run只执行本地工具，不请求模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from ai_agent.persistence.history_session import ArchiveCapture
from ai_agent.persistence.history_readback import HistoryReadAccess
from ai_agent.persistence.memory_state import HistoryArchiveState

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "tests/fixtures/evaluation/history_retrieval_coverage.json"


def evaluate(fixture, archive_root):
    refs = []
    for spec in fixture["sources"]:
        capture = ArchiveCapture(archive_root, None, True)
        for text in spec["user_turns"]:
            capture.capture(text, fixture["assistant_text"])
        refs.append(capture.state.sources[0].model_copy(update={"through_turn": spec["through_turn"]}))
    access = HistoryReadAccess(archive_root)
    access.start_turn(HistoryArchiveState(enabled=True, sources=tuple(refs)))
    search = next(t for t in access.tools if t.name == "search_conversation_turns")
    rows = []
    for case in fixture["cases"]:
        # 各查询独立重置单轮缓存；这里不模拟模型查询改写或整轮工具预算。
        access.start_turn(HistoryArchiveState(enabled=True, sources=tuple(refs)))
        result = json.loads(search.invoke({"query": case["query"], "k": 5}))
        matches = result.get("data", {}).get("matches", [])
        found = {(m["source_index"], m["turn_number"]): m["rank"] for m in matches}
        targets = {tuple(t) for t in case["targets"]}
        ranks = [found[t] for t in targets if t in found]
        ok = result.get("ok") is True
        rows.append({**case, "tool_ok": ok, "matches": matches,
                     "hit_at_1": ok and any(rank == 1 for rank in ranks) if targets else None,
                     "hit_at_5": ok and bool(ranks) if targets else None,
                     "reciprocal_rank_at_5": 1 / min(ranks) if ranks else (0.0 if targets else None),
                     "negative_empty": ok and not matches if not targets else None,
                     "error": result.get("error")})
    groups = {}
    for group in dict.fromkeys(r["group"] for r in rows):
        part = [r for r in rows if r["group"] == group]
        groups[group] = {"cases": len(part),
                         "hit_at_5_count": sum(r["hit_at_5"] is True for r in part),
                         "negative_empty_count": sum(r["negative_empty"] is True for r in part),
                         "tool_error_count": sum(not r["tool_ok"] for r in part)}
    return {"rows": rows, "groups": groups,
            "scope": [{"source_index": i, "through_turn": r.through_turn} for i, r in enumerate(refs, 1)]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=ROOT / ".evaluation-results/history-retrieval-coverage-v1.json")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    fixture = json.loads(args.cases.read_text(encoding="utf-8"))
    if fixture["schema_version"] != "history-retrieval-coverage-v1":
        raise ValueError("不支持的覆盖诊断格式")
    if not args.run:
        print(json.dumps({"mode": "preflight_only", "fixture": fixture}, ensure_ascii=False, indent=2))
        return 0
    output = args.output.resolve()
    run_id = uuid4().hex
    archives = output.parent / "history-coverage-artifacts" / run_id / "archives"
    source_hash = hashlib.sha256()
    for relative in ("src/ai_agent/persistence/history_readback.py",
                     "src/ai_agent/persistence/history_archive.py",
                     "src/ai_agent/retrieval/lexical_retrieval.py",
                     "tests/helpers/history_retrieval_coverage.py"):
        source_hash.update(relative.encode("utf-8"))
        source_hash.update((ROOT / relative).read_bytes())
    result = {"run_id": run_id, "mode": "offline_fixed_queries", "assessment": "not_evaluated",
              "source_py_sha256": source_hash.hexdigest(),
              "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
              "archives": str(archives), **evaluate(fixture, archives)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 召回未命中是需要保留的诊断结果；只有工具失败才令本地执行失败。
    return 0 if all(r["tool_ok"] for r in result["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
