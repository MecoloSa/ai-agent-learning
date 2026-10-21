"""
AI 学习助手的批量评测运行器，用于自动化测试和评估学习助手在不同场景下的表现.
    流程：加载用例 → 逐条运行 → 记录结果 → 保存输出

    - 从 JSONL 文件加载评测用例（学习材料、目标、天数），调用学习助手执行并记录结果。
    - 支持 pipeline / agent 两种模式，每个 case × 每个 mode 笛卡尔积执行。
    - 单案例异常不中断批量评测，错误按类型记录便于聚合分析。
    - 用正则提取输出中的 [source=...] 引用，dict.fromkeys 去重保序。
    - "x" 模式 + 提前检查防止覆盖历史实验结果，避免浪费 API 费用。
    - perf_counter 精确计时，ensure_ascii=False 保证中文输出。
"""

from __future__ import annotations

import argparse
import json
import re
import time

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable
from uuid import uuid4

from ai_agent.runtime.agents import run_learning_assistant

CITATION_PATTERN = re.compile(r"\[source=[^\]\n]+\]")

@dataclass(frozen=True)
class EvalCase:
    """评测用例，包含输入、期望输出和元信息。"""
    case_id: str
    material_dir: str
    learning_goal: str
    days: int

@dataclass
class RunRecord:
    run_id: str
    case_id: str
    mode: str
    success: bool
    latency_ms: int
    citations: list[str]
    analysis: str
    plan: str
    error: str | None = None
    # 每个运行拥有独立字典；正文仅留在本地评测文件，便于追查错引。
    plan_evidence_trace: dict = field(default_factory=dict)

    # 默认值保持旧调用兼容；空轨迹表示未取得，不表示运行失败。
    orchestrator: str = "legacy"
    workflow_trace: dict = field(default_factory=dict)

def load_cases(path: Path) -> list[EvalCase]:
    # 函数签名：接收一个 JSONL 文件路径，返回 EvalCase 列表
    cases: list[EvalCase] = []

    for line_number, line in enumerate(
            # start=1)：同时给出行号和行内容，行号从 1 开始
            path.read_text(encoding="utf-8").splitlines(), start=1):
        # 跳过空行
        if not line.strip():
            continue

        try:
            # json.loads() 将一行 JSON 字符串解析为 Python 字典
            # EvalCase(**dict) — ** 是字典解包运算符
            cases.append(EvalCase(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"{path}:{line_number}不是合法的评测案例: {exc}"
            ) from exc

    return cases

# 运行单个测评案例
def run_case(
    case: EvalCase,
    mode: str,
    *,
    execute: Callable | None = None,
    orchestrator: str = "legacy",
) -> RunRecord:
    if orchestrator not in {"legacy", "langgraph"}:
        raise ValueError("orchestrator 必须为 legacy 或 langgraph")

    if execute is not None and orchestrator != "legacy":
        # 四参数替身不接收编排参数，不能冒充真实图运行。
        raise ValueError(
            "自定义 execute 仅支持 legacy；"
            "测试图时请替换具体业务步骤"
        )

    started = time.perf_counter()
    plan_evidence_trace: dict = {}
    workflow_receiver: dict = {}

    try:
        if execute is None:
            options = {
                "plan_evidence_trace": plan_evidence_trace,
            }

            if orchestrator == "langgraph":
                options.update(
                    orchestrator="langgraph",
                    workflow_trace=workflow_receiver,
                )

            # legacy不新增关键字，兼容已有严格签名的测试替身。
            analysis, plan, _ = run_learning_assistant(
                case.material_dir,
                case.learning_goal,
                case.days,
                mode,
                **options,
            )
        else:
            # 保持已有假执行函数的四参数接口；空轨迹不代表没有证据。
            analysis, plan, _ = execute(
                case.material_dir, case.learning_goal, case.days, mode,
            )
        error = None
    except Exception as exc:
        # 批量评测中单个案例失败不能中断剩余案例。
        # 动态评估：按错误类型聚合，区分模型、检索、配置和代码错误。
        analysis = ""
        plan = ""
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = round((time.perf_counter() - started) * 1000)
    # list(...)：把字典的键转回列表
    # dict.fromkeys(list) 创建一个字典，列表元素作为键，值全为 None。由于字典键必须唯一，自动去重
    citations = list(dict.fromkeys(
        # 用正则找出所有匹配 [source=...] 的子串
        CITATION_PATTERN.findall(f"{analysis}\n{plan}")
    ))

    return RunRecord(
        run_id=str(uuid4()),
        case_id=case.case_id,
        mode=mode,
        success=bool(plan) and error is None,
        latency_ms=latency_ms,
        citations=citations,
        analysis=analysis,
        plan=plan,
        error=error,
        plan_evidence_trace=plan_evidence_trace,
        orchestrator=orchestrator,
        # 接收字典外层的latest_run仅用于传递；
        # 结果文件直接保存该次图记录，减少一层嵌套。
        workflow_trace=workflow_receiver.get("latest_run", {}),
    )

def write_record(records: list[RunRecord], output: Path) -> None:
    """保存本次实验，不覆盖已有结果。"""
    # 这确保写入文件前目标目录一定存在
    output.parent.mkdir(parents=True, exist_ok=True)
    # 动态评估：不同实验使用不同文件，保留原始输出供对照。
    # x模式还可防止预检查后文件被其他进程创建时发生覆盖。
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(asdict(record), ensure_ascii=False) + "\n"
            )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--orchestrator",
        choices=["legacy", "langgraph"],
        default="legacy",
        help="外层编排方式；默认legacy，langgraph同时记录节点轨迹",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细调用、检索与预算日志")
    parser.add_argument(
        "--modes",
        nargs="+",                      # nargs="+" — 接受一个或多个值
        choices=["pipeline", "agent"],
        default=["pipeline", "agent"],
    )
    args = parser.parse_args()
    from ai_agent.config import get_settings
    from ai_agent.infrastructure.console_logging import configure_console_logging
    configure_console_logging(verbose=args.verbose, log_level=get_settings().log_level)
    # 提前检查，避免调用付费模型后才发现结果文件已经存在。
    if args.output.exists():
        parser.error(f"结果文件已存在，请换新文件名：{args.output}")
    cases = load_cases(args.cases)
    records = [
        run_case(case, mode, orchestrator=args.orchestrator)
        for case in cases
        for mode in args.modes
    ]
    write_record(records, args.output)

    for record in records:
        workflow_status = record.workflow_trace.get(
            "status", "not_recorded"
        )
        failed_node = record.workflow_trace.get("failed_node")

        print(
            f"{record.case_id} | mode={record.mode} | "
            f"orchestrator={record.orchestrator} | "
            f"success={record.success} | "
            f"workflow={workflow_status} | "
            f"failed_node={failed_node} | "
            f"耗时={record.latency_ms / 1000:.1f}秒 | "
            f"error={record.error}"
        )

    print(f"结果已保存：{args.output.resolve()}")

if __name__ == "__main__":
    main()
