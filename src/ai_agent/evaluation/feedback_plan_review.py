"""反馈计划结果的人工评审辅助：导出、校验、统计，不调用模型或自动判内容质量。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from ai_agent.evaluation.feedback_plan_cases import FeedbackPlanCase


def _read_rows(path: Path) -> tuple[bytes, list[dict]]:
    raw = path.read_bytes()
    rows = []
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path.name}:{number}: JSON无效") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: 每行必须是对象")
        rows.append(row)
    if not rows:
        raise ValueError("JSONL不能为空")
    return raw, rows


def prepare_packets(results: Path) -> list[dict]:
    """只使用本次运行的快照，不重读后来可能变动的案例、回复或最终计划。"""
    raw, rows = _read_rows(results)
    digest = hashlib.sha256(raw).hexdigest()
    packets = []
    seen = set()
    for row in rows:
        required = {"case_spec", "initial_context", "turns", "plan_changes", "checks"}
        if not required <= row.keys():
            raise ValueError("结果缺少观察快照，请重新回放；不能反推旧版候选")
        case = FeedbackPlanCase.model_validate(row["case_spec"])
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in seen:
            raise ValueError("run_id缺失或重复")
        seen.add(run_id)
        mode = row.get("execution_mode")
        if row.get("case_id") != case.case_id or mode not in {"offline_stub", "online_model"}:
            raise ValueError("案例不一致或执行模式无效")

        if packets and packets[0]["execution_mode"] != mode:
            raise ValueError("固定回复与真实模型结果不能混合评审，请拆分文件")

        if mode == "online_model":
            experiment = row.get("experiment")
            required = {
                "before", "after", "conditions_unchanged", "condition_error_type",
            }
            if not isinstance(experiment, dict) or not required <= experiment.keys():
                raise ValueError("在线结果缺少运行条件记录")

            unchanged = experiment["conditions_unchanged"]
            if (
                not isinstance(experiment["before"], dict)
                or not isinstance(experiment["after"], (dict, type(None)))
                or (unchanged is not None and type(unchanged) is not bool)
            ):
                raise ValueError("在线运行条件记录格式无效")
        if row.get("status") not in {"completed", "failed", "script_mismatch"}:
            raise ValueError("运行状态无效")
        if not isinstance(row["initial_context"], (dict, type(None))) or any(
            not isinstance(row[key], list) for key in ("turns", "plan_changes", "checks")
        ):
            raise ValueError("观察快照格式无效")
        if row["status"] == "completed" and not isinstance(row["initial_context"], dict):
            raise ValueError("完成记录缺少初始上下文")
        if any(not isinstance(check, dict) for check in row["checks"]):
            raise ValueError("控制检查格式无效")
        packets.append({
            "schema_version": "feedback-plan-review-v1",
            # 哈希用于绑定原始字节，不代表真实性认证或内容质量通过。
            "source_sha256": digest,
            "run_id": run_id,
            "case_id": case.case_id,
            "split": case.split,
            "execution_mode": row["execution_mode"],
            "run_status": row["status"],
            # 在线表保留原实验条件；离线表不新增字段，兼容已有标注。
            **(
                {"experiment": deepcopy(row["experiment"])}
                if mode == "online_model" else {}
            ),
            "case_spec": deepcopy(row["case_spec"]),
            "initial_context": deepcopy(row["initial_context"]),
            # 旧结果不补造最终状态，保持已有评审表可继续核对。
            **(
                {"final_context": deepcopy(row["final_context"])}
                if "final_context" in row else {}
            ),
            "turns": deepcopy(row["turns"]),
            "plan_changes": deepcopy(row["plan_changes"]),
            "control_checks": deepcopy([
                check for check in row["checks"] if check.get("category") == "control"
            ]),
            "source_evidence_status": "not_collected",
            "reviewer": "",
            "checks": [
                {"check_id": check.check_id, "category": check.category,
                 "criterion": check.criterion, "basis": check.basis,
                 "status": "pending", "value": None, "reason": "", "evidence_refs": []}
                for check in case.checks if check.category != "control"
            ],
        })
    return packets


def _resolve_ref(packet: dict, ref: str):
    """解析JSON Pointer，如/plan_changes/0/proposed_plan；不访问文件或网络。"""
    if not isinstance(ref, str) or not ref.startswith("/"):
        raise ValueError("依据引用必须是以/开头的JSON Pointer")
    parts = ref[1:].split("/")
    if parts[0] not in {"initial_context", "final_context", "turns", "plan_changes"}:
        raise ValueError("依据引用只能指向初始上下文或实际执行观察")
    current = packet
    try:
        for part in parts:
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, list):
                if not part.isascii() or not part.isdecimal() or str(int(part)) != part:
                    raise ValueError("列表索引必须是非负整数")
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                raise ValueError("引用经过了不可索引的值")
    except (KeyError, IndexError) as exc:
        raise ValueError("依据引用不存在") from exc
    if current is None or current == "" or current == [] or current == {}:
        raise ValueError("依据引用不能为空")
    return current


def summarize_reviews(results: Path, reviews: Path) -> dict:
    expected = prepare_packets(results)
    _, actual = _read_rows(reviews)
    by_id = {packet["run_id"]: packet for packet in expected}
    if len(actual) != len(expected):
        raise ValueError("评审表必须覆盖原结果的每次运行且不能多出记录")
    counts = {category: {"pending": 0, "true": 0, "false": 0, "unknown": 0}
              for category in ("quality", "evidence")}
    seen = set()
    for packet in actual:
        run_id = packet.get("run_id")
        if not isinstance(run_id, str) or run_id in seen or run_id not in by_id:
            raise ValueError("评审run_id未知或重复")
        seen.add(run_id)
        baseline = by_id[run_id]
        # 仅允许编辑reviewer与各检查的四个标注字段；观察和判据不可随标注改写。
        if packet.keys() != baseline.keys() or any(
            packet[key] != value for key, value in baseline.items()
            if key not in {"reviewer", "checks"}
        ):
            raise ValueError("来源、判据或观察已变化，请使用对应原始结果重新导出")
        reviewer = packet["reviewer"]
        if not isinstance(reviewer, str):
            raise ValueError("reviewer必须是字符串")
        checks = packet["checks"]
        if not isinstance(checks, list) or len(checks) != len(baseline["checks"]):
            raise ValueError("检查项缺失或多出")
        for check, original in zip(checks, baseline["checks"], strict=True):
            editable = {"status", "value", "reason", "evidence_refs"}
            if not isinstance(check, dict) or check.keys() != original.keys() or any(
                check[key] != value for key, value in original.items() if key not in editable
            ):
                raise ValueError("检查项顺序、身份或判据已变化")
            status, value = check["status"], check["value"]
            reason, refs = check["reason"], check["evidence_refs"]
            if (status not in ("pending", "reviewed")
                    or (value is not None and type(value) is not bool)
                    or not isinstance(reason, str) or not isinstance(refs, list)):
                raise ValueError("标注字段格式无效；value仅允许true/false/null")
            for ref in refs:
                _resolve_ref(packet, ref)
            if status == "pending":
                if value is not None or reason or refs:
                    raise ValueError("待评审项不能同时携带结论")
                key = "pending"
            else:
                if packet["run_status"] != "completed":
                    raise ValueError("未完成回放不能提交内容结论")
                if (
                    packet["execution_mode"] == "online_model"
                    and value is not None
                    and packet["experiment"]["conditions_unchanged"] is not True
                ):
                    raise ValueError("运行条件变化或未核实，只能待评审或说明无法判断")

                if not reviewer.strip() or not reason.strip():
                    raise ValueError("已评审项必须填写评审人和理由")
                if value is not None and not refs:
                    raise ValueError("true/false结论必须指向实际观察")
                # 能定位只代表引用有效；这里不验证理由是否充分或标签是否正确。
                key = "unknown" if value is None else str(value).lower()
            counts[check["category"]][key] += 1
    return {
        "execution_mode": expected[0]["execution_mode"], "runs": len(expected),
        "source_sha256": expected[0]["source_sha256"],
        "annotation_status": "pending" if any(c["pending"] for c in counts.values()) else "complete",
        "manual_checks": counts,
        "assessment": "annotation_summary_only",
        # 仅表示真实模型结果已有至少一项人工布尔标签，不代表整例或总体合格。
        "model_quality_evaluated": (
            expected[0]["execution_mode"] == "online_model"
            and any(c["true"] + c["false"] > 0 for c in counts.values())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "report"):
        child = sub.add_parser(command)
        child.add_argument("--results", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if command == "report":
            child.add_argument("--reviews", type=Path, required=True)
    args = parser.parse_args()
    try:
        # 全量校验通过后才创建输出；独占创建，防止覆盖原结果或已有人工标注。
        if args.command == "prepare":
            packets = prepare_packets(args.results)
            content = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in packets)
        else:
            report = summarize_reviews(args.results, args.reviews)
            content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.error(f"评审文件处理失败：{exc}")
    print(f"{args.command}: 已保存 {args.output}")


if __name__ == "__main__":
    main()
