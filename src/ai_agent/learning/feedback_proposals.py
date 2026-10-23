"""自然对话反馈的待确认提议契约。

模型选择来源、范围、可选状态及更正正文候选；来源原话由程序按ID附加。
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
    learning_today,
)


class FeedbackProposalArgs(StateModel):
    """未来提议工具暴露给模型的参数，不包含完整状态或确认权限。"""

    # current为本轮；历史ID由程序本轮提供，不能任意引用摘要或助手回答。
    # 承接消息可能只授权记录，具体反馈事实仍来自此前用户原话。
    source_id: str = Field(
        default="current",
        min_length=1,
        description=(
            "包含拟记录事实的用户消息ID，取自feedback_proposal_sources。"
            "事实在上文且本轮仅授权记录时选previous_user_*；本轮已表达事实时选current。"
        ),
    )
    scope: Literal["task", "current_context"]
    task_id: str | None = Field(min_length=1)
    activity_status: ActivityStatus | None
    understanding_status: UnderstandingStatus | None
    supersedes_record_id: str | None = Field(
        default=None,
        min_length=1,
        description="仅用户明确更正当前安排反馈时，填写本轮记录中的旧record_id；普通补充填null，对象不明先澄清。",
    )

    # 只为更正提供完整候选正文；来源原话仍由程序按source_id获取。
    replacement_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
        strict=True,
        description=(
            "仅更正安排反馈时填写更正后的完整正文，普通反馈填null。"
            "结合旧正文与用户明确修改，保留未要求改变的条件；不确定时先澄清。"
            "这是待确认的整理结果，不是用户原话，不得新增用户未表达的限制。"
        ),
    )

    # 各字段先分别校验，再检查它们之间的组合是否合理。
    @model_validator(mode="after")
    def check_scope(self) -> FeedbackProposalArgs:
        if self.scope == "task":
            if self.supersedes_record_id is not None:
                raise ValueError("任务反馈暂不支持历史更正关系")
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
        # 更正必须明确给出替代整条记录的内容；普通反馈仍只保存真实原话。
        # 这里只校验协议完整性，不声称能判断自然语言是否遗漏或新增条件。
        if self.supersedes_record_id is not None:
            if self.replacement_text is None or not self.replacement_text.strip():
                raise ValueError("更正反馈必须提供非空的更正后完整正文")
        elif self.replacement_text is not None:
            raise ValueError("普通反馈不能提供更正正文")

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

    if request.supersedes_record_id is not None:
        target = next(
            (
                r for r in state.feedback_records
                if r.record_id == request.supersedes_record_id
            ),
            None,
        )
        if target is None or target.scope != "current_context":
            raise ValueError("找不到可更正的当前安排反馈")

        if (
            target.recorded_on != learning_today()
            or target.context_revision > state.context_revision
        ):
            raise ValueError("本阶段仅支持更正当天可见的安排反馈")

        if any(
            r.supersedes_record_id == target.record_id
            for r in state.feedback_records
        ):
            raise ValueError("该反馈已被更正，请选择最新记录")

    # 完整原话由程序提供，不采用模型生成的“用户原文”。
    # 原文可能含假设或转述，不能只因来源真实就自动写入。
    # current_context只定位当前安排，不表示永久偏好或时间约束一直有效。
    return PendingFeedback(
        request=request,
        user_message=user_message,
        context_revision=state.context_revision,
    )
