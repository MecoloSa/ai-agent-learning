"""复用已采用快照，仅补验新进程完整回答交付；不重跑规划和写入。"""
import argparse
import json
from pathlib import Path

from tests.helpers.learning_closed_loop import ROOT, launch, RESUME, DISCUSSION, formal
from ai_agent.persistence.state_store import load_learning_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = [
        ("technical", ROOT / ".evaluation-results/closed-loop-fixed-20261103/technical/first.json",
         ROOT / ".evaluation-results/closed-loop-fixed-20261103/technical/archives"),
        ("literature", ROOT / ".evaluation-results/closed-loop-clarified-20261103/adopted.json",
         ROOT / ".evaluation-results/closed-loop-fixed-20261103/literature/archives"),
    ]
    (root / "manifest.json").write_text(json.dumps({"mode": "online" if args.llm else "preflight_only",
        "question": RESUME, "scope": "两次独立新进程恢复，不再次修改计划"}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.llm:
        results = []
        for name, source, archives in sources:
            directory = root / name
            directory.mkdir()
            before = load_learning_state(source)
            source_bytes = source.read_bytes()
            current = directory / "current.json"
            code = launch({"source": str(source), "current": str(current), "archives": str(archives),
                "events": [{"stage": "chat", "text": RESUME}, {"stage": "chat", "text": "exit"}],
                "observation": str(directory / "observation.json")}, directory, "resume")
            result = {"case_id": name, "process_code": code}
            if not code:
                after = load_learning_state(current)
                observation = json.loads((directory / "observation.json").read_text(encoding="utf-8"))
                answer = observation["turns"][0]["response"]
                result.update(business_unchanged=formal(before) == formal(after),
                    source_unchanged=source.read_bytes() == source_bytes,
                    quote_in_answer=DISCUSSION in answer, answer=answer,
                    tools=[t["tool_name"] for t in observation["turns"][0]["audit_trace"].get("tool_results", [])])
            results.append(result)
            (root / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{name}: code={code} unchanged={result.get('business_unchanged')} quote={result.get('quote_in_answer')}", flush=True)
    return 0


if __name__ == "__main__":
    main()
