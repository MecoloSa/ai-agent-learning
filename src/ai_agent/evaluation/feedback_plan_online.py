"""单案例真实交互评测；默认仅预检，--llm才执行模型调用。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ai_agent.config import get_settings
from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases
from ai_agent.evaluation.feedback_plan_runner import run_case
from ai_agent.evaluation.interactive_metrics import build_eval_config_snapshot
from ai_agent.persistence.checkpoint_guard import material_fingerprint


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_manifest(case, cases_path: Path) -> dict:
    # 从配置模块定位实际源码根目录，避免依赖当前工作目录。
    import ai_agent.config as config_module

    root = Path(config_module.__file__).resolve().parent
    files = []

    for path in sorted(root.rglob("*.py")):
        if not path.resolve().is_relative_to(root):
            raise ValueError("源码链接越出包目录")
        files.append((path.relative_to(root).as_posix(), file_sha256(path)))

    code_hash = hashlib.sha256(
        json.dumps(files, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    settings = get_settings()
    config = build_eval_config_snapshot(settings)

    # 只补充明确允许的字段，不序列化包含密钥的整个Settings。
    config["settings"].update({
        "feedback_model": settings.feedback_model,
        "enhanced_model": settings.enhanced_model,
        "llm_retry_min_wait": settings.llm_retry_min_wait,
        "llm_retry_max_wait": settings.llm_retry_max_wait,
        "memory_summary_trigger_chars": settings.memory_summary_trigger_chars,
        "memory_summary_max_chars": settings.memory_summary_max_chars,
        "memory_window_size": settings.memory_window_size,
    })

    return {
        "schema_version": "feedback-plan-online-v1",
        "profile": {
            "memory_strategy": "window",
            "read_transport": "local",
            "analysis": "empty",
            "preloaded_material_context": "empty",
        },
        "config": config,
        "chat_endpoint_sha256": hashlib.sha256(
            settings.chat_base_url.encode("utf-8")
        ).hexdigest(),
        "cases_sha256": file_sha256(cases_path),
        "initial_state_sha256": file_sha256(case.initial_state_file),
        "materials_fingerprint": material_fingerprint(str(case.material_dir)),
        "source_py_sha256": code_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()

    try:
        cases = load_feedback_plan_cases(args.cases)
        case = next(
            (item for item in cases if item.case_id == args.case_id),
            None,
        )
        if case is None:
            parser.error("case-id不存在")

        # 结果和console.txt等产物不得写进被评测资料，避免评测污染自己的输入。
        if args.output.resolve().is_relative_to(case.material_dir.resolve()):
            parser.error("输出必须位于案例资料目录外")

        if args.output.exists():
            parser.error("结果文件已存在，请换一个文件名")

        before = build_manifest(case, args.cases)

        if not args.llm:
            print(json.dumps({
                "mode": "preflight_only",
                "case_id": case.case_id,
                "manifest": before,
            }, ensure_ascii=False, indent=2))
            return

        args.output.parent.mkdir(parents=True, exist_ok=True)

        # 在模型执行前独占创建，避免先付出调用成本再发现结果不能保存。
        with args.output.open("x", encoding="utf-8") as stream:
            artifacts = args.output.parent / f"{args.output.stem}-artifacts"
            record = run_case(case, None, artifacts, online=True)

            after, condition_error = None, None
            try:
                after = build_manifest(case, args.cases)
            except (OSError, ValueError) as exc:
                # 后置核对失败仍保存实际回复及轨迹，不把异常正文写入摘要。
                condition_error = type(exc).__name__

            unchanged = before == after if after is not None else None
            record["experiment"] = {
                "before": before,
                "after": after,
                "conditions_unchanged": unchanged,
                "condition_error_type": condition_error,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    except (OSError, ValueError) as exc:
        parser.error(f"预检或结果写入失败：{type(exc).__name__}")

    print(
        f"{case.case_id}: {record['status']} "
        f"conditions_unchanged={unchanged}"
    )

    if (
        record["status"] != "completed"
        or unchanged is not True
        or any(
            check["category"] == "control" and check["value"] is False
            for check in record["checks"]
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()