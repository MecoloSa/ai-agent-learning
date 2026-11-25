"""交互工具的执行要求与成功判定。

只使用程序真实执行产生的记录，不相信模型自述的“已经调用”。
不同工具可以返回不同格式，因此成功判定需要分别注册。

已为按行读取、资料清单和PDF按页读取注册成功判定。
检索工具暂仅确认真实调用正常返回，不据此判断是否命中或证据充分。
成功调用不等于参数完全符合用户要求，也不等于最终回答完全忠实。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Literal


@dataclass(frozen=True)
class ExecutionRequirement:
    """指定工具的确认要求；调用正常返回与业务成功分别处理。"""

    tool_name: str
    confirmation: Literal["succeeded", "returned"] = "succeeded"


@dataclass(frozen=True)
class ExecutionRecord:
    """程序真实执行记录；返回、成功及证据充分不是同一回事。"""

    tool_name: str
    succeeded: bool
    returned: bool = False
    failure_reason: str | None = None


def resolve_execution_requirement(
    user_input: str,
) -> ExecutionRequirement | None:
    """只识别句首明确点名工具的请求，不对所有问句进行意图分类。"""
    match = re.match(
        r"^(?:请\s*)?(?:实际|重新|再次)\s*调用\s*"
        r"(?P<tool_name>[A-Za-z_][A-Za-z0-9_]*)"
        r"(?=$|[\s，,。:：])",
        user_input.strip(),
    )
    if match is None:
        return None

    # 命名捕获组把工具名称单独提取出来，不在规则里固定某个工具。
    tool_name = match.group("tool_name")

    # 检索目前返回文本，只确认真实调用正常返回，不猜测语义成功。
    # 其他工具维持原来的成功校验；未知工具不会自动获得放行。
    return ExecutionRequirement(
        tool_name=tool_name,
        confirmation=(
            "returned"
            if tool_name == "retrieve_learning_context"
            else "succeeded"
        ),
    )

def _structured_result_succeeded(result: str) -> bool:
    """检查显式采用ok/data返回约定的工具，不用于任意工具。"""
    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return False

    return (
        isinstance(payload, dict)
        and payload.get("ok") is True
        and isinstance(payload.get("data"), dict)
    )


# 只有明确采用同一返回约定的工具才复用此判定。
# 检索工具仍返回普通文本，不在这里猜测其成功状态。
_SUCCESS_CHECKERS: dict[str, Callable[[str], bool]] = {
    "read_material_lines": _structured_result_succeeded,
    "list_material_files": _structured_result_succeeded,
    "inspect_document_structure": _structured_result_succeeded,
    "read_pdf_pages": _structured_result_succeeded,
    "delegate_learning_task": _structured_result_succeeded,
    "read_feedback_history": _structured_result_succeeded,
    "read_confirmation_history": _structured_result_succeeded,
    "list_conversation_sources": _structured_result_succeeded,
    "read_conversation_turn": _structured_result_succeeded,
    "search_conversation_turns": _structured_result_succeeded,

    # 此处的成功仅指提议被接收为pending，不代表确认或保存。
    "propose_learning_feedback": _structured_result_succeeded,
    "propose_task_creation": _structured_result_succeeded,
    "propose_task_cancellation": _structured_result_succeeded,
    "propose_plan_change": _structured_result_succeeded,
}


def execution_failure_reason(tool_name: str, result: str) -> str | None:
    """只解释已知工具的结构化错误码，不展示外部错误正文。"""
    if tool_name not in {"read_conversation_turn", "read_pdf_pages"}:
        return None
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not False:
        return None
    error = payload.get("error")
    if not isinstance(error, dict) or not isinstance(error.get("code"), str):
        return None
    # 文案由程序固定；工具message或模型回答可能包含路径、原文或指令，不能直接拼接。
    if tool_name == "read_pdf_pages":
        return {
            "pdf_invalid_page_range": "物理起止页无效，请检查页码和顺序。",
            "pdf_page_range_limit": "请求页数超过单次上限，请缩小范围。",
            "pdf_page_out_of_bounds": "起始物理页超出 PDF 实际页数。",
        }.get(error["code"])
    reasons = {
        "turn_out_of_scope": "请求轮次超过当前快照关联上限，本次未取得该轮原文。请使用已关联该轮的快照，或由你提供原话。",
        "source_out_of_scope": "请求来源不在本轮允许范围，请先查看当前已关联的历史来源。",
        "read_limit_exceeded": "完整问答超过本次读取字符预算，本次未返回截断原文；单次回读上限为4000字符。",
        "archive_unavailable": "关联归档缺失或无法读取，请检查归档是否仍可用；本次未取得原文。",
        "turn_missing": "关联范围内未找到该轮记录，不能据此断言这段对话从未发生。",
    }
    return reasons.get(error["code"])


def supports_execution_check(tool_name: str) -> bool:
    return tool_name in _SUCCESS_CHECKERS


def tool_result_succeeded(tool_name: str, result: str) -> bool:
    """缺少判定规则时保守返回 False，不猜测执行结果。"""
    checker = _SUCCESS_CHECKERS.get(tool_name)
    return checker(result) if checker is not None else False
