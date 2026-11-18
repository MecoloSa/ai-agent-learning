"""v0.8.1替换文件实验：生成影子源码，独立进程加载，不覆盖src。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MODULES = {
    "ai_agent.runtime.prompt_rules": ROOT / "src/ai_agent/runtime/prompt_rules.py",
    "ai_agent.runtime.agents": ROOT / "src/ai_agent/runtime/agents.py",
}
PROFILES = ("baseline", "freedom", "recovery", "evidence", "combined", "applied")


def stage_evidence(block: str) -> str:
    """仅整理已知读取回执；未知结构原样保留，不合并范围或推测完整性。"""
    try:
        payload = json.loads(block)
    except (ValueError, TypeError):
        return block
    if not isinstance(payload, dict) or set(payload) != {"ok", "data"} or payload["ok"] is not True:
        return block
    data = payload["data"]
    if not isinstance(data, dict) or not isinstance(data.get("citation"), str):
        return block
    if not isinstance(data.get("source"), str) or type(data.get("truncated")) is not bool:
        return block
    prefix_keys = {"source", "citation", "scope", "returned_chars", "truncated", "text"}
    line_keys = {"source", "citation", "requested_start_line", "requested_end_line",
                 "returned_start_line", "returned_end_line", "truncated", "lines"}
    if set(data) == prefix_keys and data["scope"] == "prefix" and isinstance(data["text"], str):
        return (f"{data['citation']}\n读取范围：文件开头；截断={data['truncated']}\n"
                + data["text"])
    if set(data) != line_keys or not isinstance(data["lines"], list) or not data["lines"]:
        return block
    rows = data["lines"]
    if not all(isinstance(row, dict) and set(row) == {"line", "text", "partial"}
               and type(row["line"]) is int and isinstance(row["text"], str)
               and type(row["partial"]) is bool for row in rows):
        return block
    if (rows[0]["line"] != data["returned_start_line"]
            or rows[-1]["line"] != data["returned_end_line"]
            or any(right["line"] != left["line"] + 1 for left, right in zip(rows, rows[1:]))):
        return block
    # 保留实际范围和每行partial，不把请求结束位置当成实际取得的位置。
    header = (f"{data['citation']}\n实际行范围：{data['returned_start_line']}-"
              f"{data['returned_end_line']}；截断={data['truncated']}\n")
    return header + "\n".join(
        f"{row['line']}{'（部分）' if row['partial'] else ''}: {row['text']}" for row in rows
    )


def once(source: str, before: str, after: str) -> str:
    """定位不唯一即停止，避免新源码误套旧替换。"""
    if source.count(before) != 1:
        raise ValueError(f"替换锚点不唯一：{before[:70]!r}")
    return source.replace(before, after, 1)


def candidate_sources(profile: str) -> dict[str, str]:
    if profile not in PROFILES:
        raise ValueError(profile)
    if profile == "applied":
        return {name: path.read_text(encoding="utf-8") for name, path in MODULES.items()}
    # 实验始终从已固定的基线构建，不能随着正式落地改变AB的基线含义。
    sources = {
        name: subprocess.check_output(
            ["git", "show", f"v0.8.1:{path.relative_to(ROOT).as_posix()}"],
            cwd=ROOT,
        ).decode("utf-8")
        for name, path in MODULES.items()
    }
    a = sources["ai_agent.runtime.agents"]
    if profile in {"freedom", "combined"}:
        a = once(a, '"2. 每天最多两个任务，每项写明学习内容、具体行动和与目标匹配的完成标志。\\n"',
                 '"2. 根据目标与可用时间组织任务，每项写明学习内容、具体行动和与目标匹配的完成标志。\\n"')
        a = once(a, "per_tool_limit = ANALYSIS_TOOL_LIMITS.get(name, 4)", "per_tool_limit = total_limit")
        start = a.index("    remaining_by_tool = {", a.index("def _build_analysis_budget_status"))
        end = a.index("\n\n    notices =", start)
        a = a[:start] + "    remaining_by_tool = {name: remaining_total for name in tool_names}" + a[end:]
        a = once(a, '+ str(total_limit)\n        + " 次；各工具上限："\n        + json.dumps(ANALYSIS_TOOL_LIMITS, ensure_ascii=False)',
                 '+ str(total_limit)\n        + " 次；各工具共享总预算。"')
        a = a.replace("现在直接输出最终完整学习计划。", "现在输出证据支持的学习安排；不足以安排时说明缺口或提出必要澄清。")
        a = once(a, "4. 证据不足时说明限制；最终仅输出完整计划及必要说明，不输出内部判断过程。",
                 "4. 依据现有证据输出计划及必要说明；不足以完整安排时说明缺口、给出可支持的部分或提出必要澄清。")
        rules = sources["ai_agent.runtime.prompt_rules"]
        rules = once(rules, "失败时如实说明，未经授权不改读其他文件。",
                     "失败时如实说明；可在用户授权的资料范围内继续查找，遵守用户明确限定的文件范围。")
        sources["ai_agent.runtime.prompt_rules"] = rules
    if profile in {"recovery", "combined"}:
        a = once(a, "    executed_calls = 0\n    no_progress_rounds = 0", "    attempts_by_signature = {}\n    executed_calls = 0\n    no_progress_rounds = 0")
        a = once(a, "                executed_calls += 1\n                executed_this_round += 1",
                 "                attempts_by_signature[signature] = attempts_by_signature.get(signature, 0) + 1\n                executed_calls += 1\n                executed_this_round += 1")
        a = once(a, '                    result = f"工具执行出错：{exc}"',
                 '''                    retryable = isinstance(exc, (TimeoutError, ConnectionError)) and attempts_by_signature[signature] < 3
                    if retryable:
                        seen_signatures.discard(signature)
                    result = json.dumps({"ok": False, "error": {
                        "code": type(exc).__name__, "retryable": retryable,
                        "message": "本次未取得结果；可在剩余预算内按错误类型调整。",
                    }}, ensure_ascii=False)''')
        a = a.replace("。完全相同的工具名和参数不会重复执行，包括失败过的调用。",
                      "。成功或确定性失败不重复执行；暂时性异常最多尝试3次，仍计入总预算。")
        a = a.replace("同参请求仍会被拒绝；剩余额度不代表允许重复执行。",
                      "同参成功结果和确定性错误不重复执行，暂时性异常依据回执中的retryable决定。")
    if profile in {"evidence", "combined"}:
        a = once(a, "def _limit_evidence_context(", inspect.getsource(stage_evidence) + "\n\ndef _limit_evidence_context(")
        a = once(a, "    selected: list[str] = []\n    current_length = 0", '''    input_block_count = len(blocks)
    blocks = list(dict.fromkeys(blocks))
    duplicate_blocks = input_block_count - len(blocks)
    blocks = [stage_evidence(block) for block in blocks]

    selected: list[str] = []
    current_length = 0''')
        a = once(a, '"selection_policy": "ordered_whole_blocks",', '"selection_policy": "exact_unique_stage_reading_blocks",')
        a = once(a, '"input_blocks": len(blocks),', '"input_blocks": input_block_count,\n            "unique_blocks": len(blocks),\n            "duplicate_blocks": duplicate_blocks,')
    sources["ai_agent.runtime.agents"] = a
    return sources


def fingerprint() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in MODULES.values()}


@contextmanager
def replacement(profile: str, output: Path):
    """完整替换文件写入实验目录；临时加载到同名模块，结束后恢复模块字典。"""
    before = fingerprint()
    originals = {}
    sources = candidate_sources(profile)
    output.mkdir(parents=True, exist_ok=True)
    try:
        for name, source in sources.items():
            path = output / (name.replace(".", "/") + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
            module = importlib.import_module(name)
            originals[name] = module.__dict__.copy()
            # 编译真实替换文件，traceback可以定位候选；不覆盖模块源文件路径。
            exec(compile(source, str(path), "exec"), module.__dict__)
        yield
    finally:
        for name, state in reversed(list(originals.items())):
            module = importlib.import_module(name)
            module.__dict__.clear()
            module.__dict__.update(state)
        if fingerprint() != before:
            raise RuntimeError("正式源码发生变化，停止实验")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = args.output_dir
    if (output / "manifest.json").exists():
        parser.error("输出已存在，请使用新目录")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"profile": args.profile, "baseline_commit": subprocess.check_output(
        ["git", "rev-parse", "v0.8.1^{commit}"], cwd=ROOT, text=True).strip(),
        "source_sha256": fingerprint(), "mode": "offline_replacement"}
    with replacement(args.profile, output / "replacement-files"):
        import pytest
        test_args = args.pytest_args
        if test_args[:1] == ["--"]:
            test_args = test_args[1:]
        code = pytest.main(test_args)
    manifest.update(source_unchanged=manifest["source_sha256"] == fingerprint(), exit_code=int(code))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
