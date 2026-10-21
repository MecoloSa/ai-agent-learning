"""用两本本地书籍的固定正反例校准评审器，而不是评测计划生成器。

默认只读取证据并检查案例；--llm才请求真实模型。人工expected_checks
仅用于比较，不传给judge_plan，避免把标准答案泄漏给评分者。
动态评估：同时观察错放率、误拒率、null比例和评审失败数，不能只报准确率。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

from ai_agent.config import create_llm, resolve_project_path
from ai_agent.evaluation.review import JUDGE_PROMPT_VERSION, REVIEW_ITEMS, judge_plan
from ai_agent.evaluation.source_evidence import load_source_evidence


def load_calibration_cases(path: Path) -> list[dict]:
    """只接受有明确人工判定的案例；null表示专门测试证据不足。"""
    cases = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            raise ValueError(f"第{number}行案例必须为对象")
        for key in ("case_id", "learning_goal", "plan", "rationale"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"第{number}行缺少{key}")
        if case["case_id"] in seen:
            raise ValueError("校准案例case_id重复")
        if type(case.get("days")) is not int or not 1 <= case["days"] <= 30:
            raise ValueError("校准案例days必须为1至30的整数")
        expected = case.get("expected_checks")
        if not isinstance(expected, dict) or not expected or not set(expected) <= set(REVIEW_ITEMS):
            raise ValueError("expected_checks必须包含已定义评审维度")
        if any(v is not None and type(v) is not bool for v in expected.values()):
            raise ValueError("人工判定只能是true、false或null")
        seen.add(case["case_id"])
        cases.append(case)
    if not cases:
        raise ValueError("校准案例不能为空")
    return cases


def summarize_calibration(records: list[dict]) -> dict:
    """按维度计数，比例分母排除评审失败，但单独展示失败与覆盖率。

    false_pass_rate是人工判错却被模型放过的比例；false_reject_rate反之。
    null不算通过，也不偷偷算成false；所有比例无样本时返回None。
    """
    observations = [(expected, r["checks"][item]) for r in records if "checks" in r
                    for item, expected in r["expected_checks"].items()]
    negatives = sum(expected is False for expected, _ in observations)
    positives = sum(expected is True for expected, _ in observations)
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    return {
        "case_count": len(records),
        "judged_cases": sum("checks" in r for r in records),
        "judge_failed": sum(r["status"] == "judge_failed" for r in records),
        "checked_dimensions": len(observations),
        "expected_fail_dimensions": negatives,
        "expected_pass_dimensions": positives,
        "agreement_rate": ratio(sum(a is b for a, b in observations), len(observations)),
        "false_pass_rate": ratio(sum(a is False and b is True for a, b in observations), negatives),
        "false_reject_rate": ratio(sum(a is True and b is False for a, b in observations), positives),
        "unknown_rate": ratio(sum(b is None for _, b in observations), len(observations)),
    }

def summarize_judge_trace(records: list[dict]) -> dict:
    """分别报告模型原始成绩与规则处理后的成绩。

    旧实验没有原始快照，不能用最终判断冒充原始判断。
    动态评估：比较相同案例上的前后成绩，同时查看覆盖数和变化原因。
    """
    raw_records = []
    paired_final_records = []
    dimension_changes = 0
    citation_changes = 0
    changed_cases = 0

    for record in records:
        if "checks" not in record:
            continue

        suggestion = record.get("suggestion", {})
        raw = suggestion.get("raw_model_result")

        # 缺少快照的旧文件不参与前后配对比较。
        if not isinstance(raw, dict) or "checks" not in raw:
            continue

        raw_records.append({
            "status": "judged",
            "expected_checks": record["expected_checks"],
            "checks": raw["checks"],
        })
        paired_final_records.append(record)

        changes = suggestion.get("rule_changes", [])
        changed_cases += bool(changes)
        dimension_changes += sum(
            change["scope"] == "dimension" for change in changes
        )
        citation_changes += sum(
            change["scope"] == "citation" for change in changes
        )

    return {
        "trace_case_count": len(raw_records),
        "judged_cases_without_trace": sum(
            "checks" in record for record in records
        ) - len(raw_records),
        "raw_model": summarize_calibration(raw_records),
        "final_same_cases": summarize_calibration(paired_final_records),
        "cases_with_verdict_changes": changed_cases,
        "dimension_verdict_changes": dimension_changes,
        "citation_verdict_changes": citation_changes,
    }

def show_progress(stage: str, message: str) -> None:
    """即时输出执行事实，与stdout中的指标分开，不输出资料正文。"""
    # flush避免输出缓冲，让等待模型时能看到当前停在哪个案例。
    print(f"[{stage}] {message}", file=sys.stderr, flush=True)


def show_case_result(record: dict, elapsed: float, *, verbose: bool = False) -> None:
    """展示最终判定、人工标签对照和规则变化，不把无标签项算入成绩。"""
    labels = {
        "goal_alignment": "目标对齐",
        "concept_accuracy": "概念准确",
        "practice_executable": "实践可执行",
        "citation_support": "引用支持",
        "reasoning_boundary": "推理边界",
        "internal_consistency": "内部一致",
    }
    checks = record["checks"]

    def verdict(value):
        return "符合" if value is True else (
            "不符合" if value is False else "未判定"
        )

    expected = record["expected_checks"]
    matched = sum(checks[key] is value for key, value in expected.items())
    changes = record["suggestion"].get("rule_changes", [])

    show_progress(
        "评审完成",
        f"{record['case_id']} | 耗时={elapsed:.2f}s | "
        f"标注项一致={matched}/{len(expected)} | "
        f"判定变化记录={len(changes)}",
    )
    if not verbose:
        return
    show_progress(
        "最终判定",
        "；".join(f"{labels[key]}={verdict(checks[key])}" for key in REVIEW_ITEMS),
    )

    # 动态评估：变化不等于改进，需要结合人工标准看方向。
    for change in changes:
        target = labels.get(change["target"], change["target"])
        show_progress(
            "规则变化",
            f"{change['scope']}:{target} | "
            f"{verdict(change['before'])} → {verdict(change['after'])}",
        )

    # 只展示有人工标签的分歧；无标签项仍需要人工抽检。
    for key, value in expected.items():
        if checks[key] is not value:
            show_progress(
                "标注分歧",
                f"{record['case_id']} | {labels[key]} | "
                f"人工={verdict(value)}，最终={verdict(checks[key])}",
            )

def main() -> None:
    parser = argparse.ArgumentParser(description="校准书籍计划评审器，默认离线证据检查")
    parser.add_argument("--cases", default="tests/fixtures/evaluation/judge_calibration_cases.jsonl")
    parser.add_argument("--materials", default="src/ai_agent/eval/ai_agent_book")
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm", action="store_true", help="调用配置中的真实聊天模型")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示逐项判定、规则变化和调试日志")
    args = parser.parse_args()
    from ai_agent.config import get_settings
    from ai_agent.infrastructure.console_logging import configure_console_logging
    configure_console_logging(verbose=args.verbose, log_level=get_settings().log_level)
    output = resolve_project_path(args.output)
    if output.exists():
        parser.error("输出已存在，请使用新实验文件名")
    run_started = perf_counter()
    case_path = resolve_project_path(args.cases)
    materials = resolve_project_path(args.materials)

    show_progress("准备", "加载并校验案例")
    cases = load_calibration_cases(case_path)
    show_progress(
        "准备",
        f"案例数={len(cases)} | "
        f"模式={'在线评审' if args.llm else '离线证据检查'} | "
        f"版本={JUDGE_PROMPT_VERSION}",
    )

    # 在付费请求前完成全部证据读取；定位负例不等同于读取流程异常。
    packs = []
    for index, case in enumerate(cases, 1):
        started = perf_counter()
        show_progress(
            "证据读取",
            f"{index}/{len(cases)} | {case['case_id']} | 开始",
        )
        try:
            evidence = load_source_evidence(materials, case["plan"])
        except Exception as exc:
            # 超预算等异常仍终止流程；只打印类型，不暴露正文或异常载荷。
            show_progress(
                "证据异常",
                f"{case['case_id']} | {type(exc).__name__} | "
                "停止，尚未请求评审模型",
            )
            raise

        packs.append(evidence)
        statuses = [
            item["status"]
            for item in evidence.get("citation_checks", [])
        ]
        show_progress(
            "证据完成",
            f"{case['case_id']} | "
            f"定位成功={statuses.count('located')} | "
            f"无效引用={statuses.count('invalid')} | "
            f"定位未完成={sum(s not in ('located', 'invalid') for s in statuses)} | "
            f"正文块={len(evidence.get('documents', []))} | "
            f"读取问题={len(evidence.get('issues', []))} | "
            f"耗时={perf_counter() - started:.2f}s",
        )

    llm = None
    if args.llm:
        show_progress("模型准备", "创建评审模型客户端")
        llm = create_llm()
        show_progress(
            "模型准备",
            f"模型={llm.model_name} | 客户端已创建，尚未确认请求成功",
        )

    records = []
    for index, (case, evidence) in enumerate(zip(cases, packs), 1):
        record = {"case_id": case["case_id"], "expected_checks": case["expected_checks"],
                  "rationale": case["rationale"], "source_evidence": evidence,
                  "cases_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
                  "judge_prompt_version": JUDGE_PROMPT_VERSION, "status": "evidence_ready"}
        if llm is not None:
            started = perf_counter()
            show_progress(
                "模型评审",
                f"{index}/{len(cases)} | {case['case_id']} | "
                "开始请求，等待响应与规则校验",
            )
            # 输出进度不会改变模型输入，人工标签仍只用于事后比较。
            try:
                judged = judge_plan(
                    {"plan": case["plan"]},
                    {
                        "learning_goal": case["learning_goal"],
                        "days": case["days"],
                    },
                    llm,
                    evidence=evidence,
                )
                record.update(
                    status="judged",
                    checks=judged["checks"],
                    suggestion=judged,
                    judge_model=llm.model_name,
                )
            except Exception as exc:
                record.update(
                    status="judge_failed",
                    error=type(exc).__name__,
                )
                show_progress(
                    "评审失败",
                    f"{case['case_id']} | {type(exc).__name__} | "
                    f"耗时={perf_counter() - started:.2f}s | 继续下一条",
                )
            else:
                show_case_result(record, perf_counter() - started, verbose=args.verbose)
        else:
            show_progress(
                "离线完成",
                f"{index}/{len(cases)} | {case['case_id']} | "
                "仅准备证据，未进行语义评审",
            )

        records.append(record)
    show_progress("结果保存", f"开始写入 {len(records)} 条记录")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    # 保留原有最终成绩，便于与之前的实验结果比较。
    print(json.dumps(
        summarize_calibration(records),
        ensure_ascii=False,
        indent=2,
    ))
    if args.llm:
        print("模型原始判断与规则处理后的配对比较：")
        print(json.dumps(
            summarize_judge_trace(records),
            ensure_ascii=False,
            indent=2,
        ))
    print(f"校准记录已保存：{output}")
    show_progress(
        "全部完成",
        f"已保存={len(records)}条 | "
        f"总耗时={perf_counter() - run_started:.2f}s",
    )


if __name__ == "__main__":
    main()
