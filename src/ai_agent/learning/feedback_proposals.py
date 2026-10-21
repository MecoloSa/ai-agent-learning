"""自然对话反馈的待确认提议契约。

模型只选择来源、范围和可选状态；完整用户消息由程序按来源ID附加。
本模块不调用模型、不更新LearningState、不保存文件。
动态评估：结构校验不等于语义正确，假设、转述和对象歧义仍需确认或澄清。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from ai_agent.learning.learning_state import (
    ActivityStatus,
    LearningState,
    StateModel,
    UnderstandingStatus,
)


class FeedbackProposalArgs(StateModel):
    """未来提议工具暴露给模型的参数，不包含完整状态或确认权限。"""

    # current为本轮；历史ID由程序本轮提供，不能任意引用摘要或助手回答。
    source_id: str = Field(default="current", min_length=1)
    scope: Literal["task", "current_context"]
    task_id: str | None = Field(min_length=1)
    activity_status: ActivityStatus | None
    understanding_status: UnderstandingStatus | None

    # 各字段先分别校验，再检查它们之间的组合是否合理。
    @model_validator(mode="after")
    def check_scope(self) -> FeedbackProposalArgs:
        if self.scope == "task":
            if self.task_id is None:
                raise ValueError("任务反馈必须指定已有task_id；对象不明应先澄清")
        else:
            if self.task_id is not None:
                raise ValueError("当前安排反馈不能同时指定task_id")
            if (
                self.activity_status is not None
                or self.understanding_status is not None
            ):
                raise ValueError("任务状态字段只能用于明确的任务反馈")
        # 两个状态均为None仍合法：原话的价值不以能否分类为前提。
        return self


@dataclass(frozen=True)
class PendingFeedback:
    """程序生成的临时结果，不是已经确认或持久化的正式记录。"""

    request: FeedbackProposalArgs
    user_message: str
    context_revision: int
    status: Literal["pending"] = field(default="pending", init=False)

    # 同一提议重复提交时复用ID，程序可拒绝重复写入。
    proposal_id: str = field(default_factory=lambda: uuid4().hex)


def prepare_feedback_proposal(
    state: LearningState,
    request: FeedbackProposalArgs,
    *,
    user_message: str,
) -> PendingFeedback:
    """核对参数与程序选定的用户原文，只生成候选，不执行更新。"""

    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("本轮用户消息不能为空")

    # 原文由收件箱按source_id解析；模型参数不再接收quote或user_message。
    # 来源合法不等于状态建议正确，仍须展示原文和变化并取得确认。

    if request.scope == "task" and not any(
        task.task_id == request.task_id for task in state.tasks
    ):
        raise ValueError(f"任务不存在：{request.task_id}")

    # 完整原话由程序提供，不采用模型生成的“用户原文”。
    # 原文可能含假设或转述，不能只因来源真实就自动写入。
    # current_context只定位当前安排，不表示永久偏好或时间约束一直有效。
    return PendingFeedback(
        request=request,
        user_message=user_message,
        context_revision=state.context_revision,
    )
