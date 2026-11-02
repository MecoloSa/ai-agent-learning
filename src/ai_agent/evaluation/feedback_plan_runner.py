"""离线固定回复回放：复用业务控制流，不评价模型或计划质量。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import time
from unittest.mock import patch
from uuid import uuid4

import ai_agent.main as app
import ai_agent.learning.confirmation as confirmation
from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases
from ai_agent.evaluation.feedback_plan_input import ReplayInput, ReplayMismatch
from ai_agent.persistence.state_store import load_learning_state
from ai_agent.learning.learning_state import LearningState


def evaluate_adoption(record: dict) -> tuple[bool | None, str]:
    """只判一次计划采用及记录保留，不判断候选的教学质量。"""
    if record.get("status") != "completed":
        return None, "回放未完成，不判采用成功"

    before_data = record.get("initial_context")
    after_data = record.get("final_context")
    excluded = {"conversation_memory", "material_binding"}
    required = set(LearningState.model_fields) - excluded

    # 不能让模型默认值补齐缺失观察，再将其误判为记录已保留。
    if any(
        not isinstance(data, dict) or not required <= data.keys()
        for data in (before_data, after_data)
    ):
        return None, "缺少完整初始或落盘业务状态"

    try:
        before = LearningState.model_validate(before_data)
        after = LearningState.model_validate(after_data)
    except ValueError:
        return None, "业务状态快照格式无效"

    events = record.get("plan_confirmations")
    changes = record.get("plan_changes")
    if not isinstance(events, list) or not isinstance(changes, list):
        return None, "缺少确认或候选观察"

    if type(record.get("source_unchanged")) is not bool:
        return None, "缺少源存档比较结果"

    if len(events) != 1 or len(changes) != 1:
        return False, "本规则要求恰好一次候选和一次确认"

    event, change = events[0], changes[0]
    if (
        not isinstance(event, dict)
        or not {"accepted", "reason"} <= event.keys()
        or type(event["accepted"]) is not bool
        or not isinstance(event["reason"], str)
        or not isinstance(change, dict)
        or not {
            "decision", "confirmation_reason", "proposed_plan", "proposed_goal",
        } <= change.keys()
    ):
        return None, "确认或候选观察不完整"

    conditions = {
        "明确采用": (
            event["accepted"] is True
            and event["reason"] == "user_confirmed"
            and change["decision"] == "confirmed"
            and change["confirmation_reason"] == "user_confirmed"
        ),
        "候选计划生效": (
            after.plan_markdown == change["proposed_plan"]
            and after.plan_markdown != before.plan_markdown
        ),
        "目标未变": change["proposed_goal"] == before.learning_goal,
        "修订号只增1": after.context_revision == before.context_revision + 1,
        "任务对应关系提醒": (
            after.tasks_need_review
            == (before.tasks_need_review or bool(before.tasks))
        ),
        "源存档未变": record["source_unchanged"],
    }

    # 独立比较允许变化之外的字段，不复用被测业务更新函数来生成评分答案。
    permitted = {"plan_markdown", "context_revision", "tasks_need_review"}
    initial = before.model_dump(mode="json", exclude=excluded)
    final = after.model_dump(mode="json", exclude=excluded)

    for key, value in initial.items():
        if key not in permitted:
            conditions[f"保留{key}"] = final[key] == value

    failed = [name for name, passed in conditions.items() if not passed]
    return (
        (False, "不满足：" + "、".join(failed))
        if failed
        else (True, "明确采用，落盘候选一致且学习记录保留")
    )

def evaluate_checks(case, record: dict) -> list[dict]:
    """只判声明的控制规则；不从案例编号、描述或模型自述猜测结果。"""
    results = []
    for check in case.checks:
        value = None
        reason = "本步未评估此项"
        if check.category == "control" and check.rule == "plan_adopted":
            value, reason = evaluate_adoption(record)
        elif check.category == "control" and check.rule is not None:
            unchanged = record.get("business_state_unchanged")
            if record.get("status") != "completed":
                reason = "回放未完成，不据局部记录判通过"
            elif type(unchanged) is not bool:
                reason = "缺少业务状态比较结果"
            elif check.rule == "state_unchanged":
                value = unchanged
                reason = "业务状态未变" if value else "业务状态发生变化"
            else:
                events = record.get("plan_confirmations")
                if not isinstance(events, list):
                    reason = "缺少真实计划确认记录"
                elif not events:
                    value, reason = False, "未执行计划确认，不能声称已拒绝"
                elif any(
                    not isinstance(e, dict)
                    or type(e.get("accepted")) is not bool
                    or not isinstance(e.get("reason"), str)
                    for e in events
                ):
                    reason = "确认记录不完整"
                else:
                    # 读到n不等于业务拒绝；依据原确认函数的实际返回和原因码。
                    value = unchanged and all(
                        e["accepted"] is False and e["reason"] == "user_declined"
                        for e in events
                    )
                    reason = "明确拒绝且状态未变" if value else "未满足明确拒绝及状态不变"
        results.append({
            "check_id": check.check_id, "category": check.category,
            "rule": check.rule, "value": value, "reason": reason,
        })
    return results

def run_case(
    case, responses: list[str] | None, artifacts: Path, *, online: bool = False,
) -> dict:
    if type(online) is not bool:
        raise ValueError("online必须是布尔值")

    if online:
        # 禁止同时给固定回复，避免在线结果混入替身内容。
        if responses is not None:
            raise ValueError("在线分支必须传responses=None")
    elif not isinstance(responses, list) or any(not isinstance(x, str) for x in responses):
        raise ValueError("固定回复必须是字符串列表")
    run_id = uuid4().hex
    run_dir = Path(artifacts) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    current = run_dir / "current.json"
    reader = ReplayInput(case.inputs)
    remaining = iter(responses if responses is not None else [])
    response_count = 0
    started = time.perf_counter()
    record = {
        "run_id": run_id, "case_id": case.case_id,
        "execution_mode": "online_model" if online else "offline_stub", "status": "running",
        "error_type": None, "script_error": None,
        "business_state_unchanged": None, "source_unchanged": None,
        "assessment": "not_evaluated", "artifacts": str(run_dir.resolve()),
        "plan_confirmations": [],
        # 固化本次输入和判据，避免后续编辑案例后失去评审依据。
        "case_spec": case.model_dump(mode="json"),
        "initial_context": None,
        "final_context": None,
        "turns": [],
        "plan_changes": [],
    }

    def fake_answer(**kwargs):
        nonlocal response_count
        try:
            answer = next(remaining)
        except StopIteration:
            raise ReplayMismatch("固定回复耗尽") from None
        response_count += 1
        record["turns"].append({
            "turn": response_count,
            "user_input": kwargs["user_input"],
            "response": answer,
        })
        return answer

    # 必须先保存原函数，再安装patch，防止包装函数递归调用自己。
    original_turn = app._invoke_interactive_turn

    def live_answer(**kwargs):
        trace = {}
        event = {
            "turn": len(record["turns"]) + 1,
            "user_input": kwargs["user_input"],
            "response": None,
            "status": "running",
            "error_type": None,
            "audit_trace": trace,
            # 固化原调用方提供的消息内容；不读取additional_kwargs等额外元数据。
            "request_messages": [
                {"type": message.type, "content": deepcopy(message.content)}
                for message in kwargs["request_messages"]
            ],
        }
        # 调用前登记：中途异常时也保留原函数已收集的部分轨迹。
        record["turns"].append(event)

        try:
            answer = original_turn(**kwargs, audit_trace=trace)
            if not isinstance(answer, str):
                raise TypeError("交互执行函数必须返回字符串")
        except Exception as exc:
            event.update(status="failed", error_type=type(exc).__name__)
            raise

        event.update(status="completed", response=answer)
        return answer

    def chat_input(prompt):
        value = reader.read_chat()
        # 本步不回放状态命令，防止/save-state等绕过隔离的current路径。
        if value.lstrip().startswith("/"):
            raise ReplayMismatch("本步不支持状态命令")
        return value

    original_confirmation = app.read_confirmation

    def plan_confirmation(*args, **kwargs):
        # 保留真实预览、trace与确认语义，仅替换该入口的数据来源。
        with patch.object(confirmation, "input", lambda _: reader.read_plan_confirmation(), create=True):
            accepted = original_confirmation(*args, **kwargs)
        record["plan_confirmations"].append({
            "accepted": accepted,
            "reason": (kwargs.get("trace") or {}).get("reason"),
        })
        return accepted

    original_review = app._review_context_change

    def observe_change(state, **kwargs):
        # 记录传入真实校验入口的候选，不从控制台文本重新解析。
        event = {
            "turn": len(record["turns"]),
            "base_goal": state.learning_goal,
            "base_plan": state.plan_markdown,
            "proposed_goal": (
                state.learning_goal if kwargs.get("learning_goal") is None
                else kwargs["learning_goal"]
            ),
            "proposed_plan": (
                state.plan_markdown if kwargs.get("plan_markdown") is None
                else kwargs["plan_markdown"]
            ),
            "decision": None,
            "confirmation_reason": None,
            "error_type": None,
        }
        record["plan_changes"].append(event)
        try:
            next_state, decision = original_review(state, **kwargs)
        except Exception as exc:
            # 进入校验入口不代表候选合法，也不代表已展示或已采用。
            # 只记录异常类型，再交回原流程处理，不吞掉业务异常。
            event.update(decision="raised", error_type=type(exc).__name__)
            raise
        event["decision"] = decision
        event["confirmation_reason"] = (
            kwargs.get("confirmation_trace") or {}
        ).get("reason")
        return next_state, decision

    def unsupported_input(prompt=""):
        raise ReplayMismatch("出现本步未支持的输入入口")

    try:
        source = case.initial_state_file.read_bytes()
        initial = load_learning_state(case.initial_state_file)
        # 保存评审所需初始业务上下文，不复制聊天历史和资料绑定信息。
        record["initial_context"] = initial.model_dump(
            mode="json", exclude={"conversation_memory", "material_binding"},
        )
        with (run_dir / "console.txt").open("x", encoding="utf-8") as console:
            # patch仅用于评测且退出后恢复；禁止并发线程运行此适配器。
            # 正常工具绑定、确认、业务更新和磁盘保存均保留原实现。
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(console))
                stack.enter_context(patch("builtins.input", unsupported_input))
                stack.enter_context(patch.object(app, "input", chat_input, create=True))
                stack.enter_context(patch.object(app, "read_confirmation", plan_confirmation))
                stack.enter_context(patch.object(app, "_review_context_change", observe_change))
                if not online:
                    # 离线基线继续隔离模型与检索；在线分支保留业务原实现。
                    stack.enter_context(patch.object(app, "create_llm", lambda **_: object()))
                    stack.enter_context(patch.object(app, "create_llm_with_tools", lambda *a, **k: object()))
                    stack.enter_context(patch.object(app, "needs_rag", lambda _: False))

                stack.enter_context(patch.object(
                    app, "_invoke_interactive_turn", live_answer if online else fake_answer,
                ))
                app.run_interactive_session(
                    "", initial.plan_markdown, "", initial.learning_goal,
                    "window", str(case.material_dir), initial_state=initial,
                    current_state_path=current, read_transport="local",
                )
        reader.assert_finished()
        if not online and next(remaining, None) is not None:
            raise ReplayMismatch("存在未使用的固定回复")
        if not current.is_file():
            raise RuntimeError("未生成预期的当前存档")
        saved = load_learning_state(current)
        # 观察真实落盘状态；不能用内存候选代替保存成功的证据。
        record["final_context"] = saved.model_dump(
            mode="json", exclude={"conversation_memory", "material_binding"},
        )
        # 聊天历史正常增长，不纳入“拒绝未改变业务状态”的观察。
        record["business_state_unchanged"] = (
            initial.model_dump(exclude={"conversation_memory"})
            == saved.model_dump(exclude={"conversation_memory"})
        )
        record["source_unchanged"] = case.initial_state_file.read_bytes() == source
        record["status"] = "completed"
    except ReplayMismatch as exc:
        record.update(status="script_mismatch", error_type=type(exc).__name__, script_error=str(exc))
    except Exception as exc:
        # 不输出异常正文，避免将配置、资料或输入内容写入结果摘要。
        record.update(status="failed", error_type=type(exc).__name__)
    record.update(
        input_events=list(reader.events), stub_response_count=response_count,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    record["checks"] = evaluate_checks(case, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="仅离线固定回复回放，不请求模型")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_feedback_plan_cases(args.cases)
    replies = json.loads(args.responses.read_text(encoding="utf-8-sig"))
    if not isinstance(replies, dict) or any(
        c.case_id not in replies or not isinstance(replies[c.case_id], list)
        or any(not isinstance(x, str) for x in replies[c.case_id]) for c in cases
    ):
        parser.error("每个案例必须有对应的固定回复字符串列表")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 先独占创建结果文件；已存在时不运行任何案例，不覆盖旧结果。
        stream = args.output.open("x", encoding="utf-8")
    except FileExistsError:
        parser.error("结果文件已存在，请换一个文件名")
    artifacts = args.output.parent / f"{args.output.stem}-artifacts"
    batch_failed = False
    with stream:
        for case in cases:
            record = run_case(case, replies[case.case_id], artifacts)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            # 每案例交给操作系统缓冲，不保证强杀/断电时的完整持久化。
            stream.flush()
            print(f"{case.case_id}: {record['status']} assessment=not_evaluated")
            batch_failed = (
                batch_failed or record["status"] != "completed"
                or any(c["category"] == "control" and c["value"] is False for c in record["checks"])
            )
    if batch_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
