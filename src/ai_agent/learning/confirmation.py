"""读取终端确认结果，只记录可观察事实，不推断用户拒绝的心理原因。

不读写学习状态或快照；n与输入结束均不执行，但必须保留两者的区别。
"""
from __future__ import annotations
from collections.abc import Callable


def read_confirmation(
    prompt: str,
    *,
    trace: dict[str, str] | None = None,
    on_preview: Callable[[], None] | None = None,
) -> bool:
    """读取最终确认；可选预览只展示内容，不构成授权。"""
    if trace is not None:
        trace.clear()

    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            reason = "input_ended"
            break

        if answer == "v" and on_preview is not None:
            # 查看后仍须单独确认，不能把“看过”当成“同意”。
            on_preview()
            continue

        reason = (
            "user_confirmed" if answer == "y"
            else "user_declined" if answer == "n"
            else "user_not_confirmed"
        )
        break

    # 回执只记录最终决定；预览次数不影响确认原因。
    # 未提供预览函数的旧调用保持原行为，v仍属于未确认。
    if trace is not None:
        trace["reason"] = reason
    return reason == "user_confirmed"


def confirmation_details(decision: str, trace: dict[str, str] | None = None) -> dict[str, str]:
    """将程序结果解释为回执；旧记录或不一致数据不补造确认原因。"""
    reason = (trace or {}).get("reason", "not_recorded")
    allowed = {
        "confirmed": {"user_confirmed"},
        "cancelled": {"user_declined", "user_not_confirmed", "input_ended"},
        "invalid": set(),
        "unchanged": set(),
    }
    if reason not in allowed[decision]:
        reason = "not_recorded"
    notes = {
        "user_confirmed": "用户在确认环节输入y，候选已采用。",
        "user_declined": "用户在确认环节输入n，明确否决本次候选；未执行。",
        "user_not_confirmed": "确认环节收到非y/n输入，未采用；不推断用户原因。",
        "input_ended": "确认输入结束，未采用；不能说成用户明确否决。",
    }
    fallback = {
        "confirmed": "候选已确认；未保留具体确认输入类型。",
        "cancelled": "候选未采用；未保留具体确认输入类型，不能推断未采用原因。",
        "invalid": "提议未通过程序校验或已失效；不能说成用户否决。",
        "unchanged": "没有实际变更，未请求确认。",
    }
    return {"confirmation_reason": reason, "decision_note": notes.get(reason, fallback[decision])}
