"""复用原配对入口，只在测试进程注入补查故障；不修改业务或原实验数据。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from ai_agent.evaluation import acquisition_pair as experiment
from ai_agent.evaluation.acquisition_pair_report import write_report


def fault_factory(base):
    def factory(documents, calls=None):
        parent = base(documents, calls)

        class TimeoutRAG(parent):
            def retrieve_many(self, pairs, **kwargs):
                if calls is not None:
                    calls.append({"queries": list(pairs), "outcome": "timeout", "returned_documents": 0})
                raise TimeoutError("固定实验故障：补查超时")

        return TimeoutRAG

    return factory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args(argv)
    run_original = experiment.run_one
    rag_original = experiment.frozen_rag

    def run(case, root, arm, repeat):
        if case["supplement_outcome"] == "timeout":
            with patch.object(experiment, "frozen_rag", fault_factory(rag_original)):
                row = run_original(case, root, arm, repeat)
        else:
            row = run_original(case, root, arm, repeat)
        row["fault_injection"] = case["supplement_outcome"]
        return row

    # 注入只发生在补查阶段；初始资料和获取事实仍由原工具生成。
    with patch.object(experiment, "run_one", run):
        result = experiment.main([
            "--cases", "tests/fixtures/evaluation/acquisition_holdout/cases.json",
            "--output-dir", str(args.output_dir), "--repeats", "2",
            *(["--llm"] if args.llm else []),
        ])
    if result == 0:
        manifest_path = args.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["fault_injection_helper_sha256"] = experiment.digest(Path(__file__).read_text(encoding="utf-8"))
        manifest["cases_kind"] = "new_synthetic_holdout_not_real_user_materials"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.llm:
            write_report(args.output_dir)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
