"""Explicit plan operations shared by the terminal and the local workspace."""

from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import Field, model_validator

from ai_agent.learning.feedback_runtime import sanitize_learning_action_reply
from ai_agent.learning.learning_state import StateModel
from ai_agent.learning.plan_proposals import PlanProposalInbox, WorkspacePlanProposalArgs
from ai_agent.learning.plan_tasks import TaskChange


class PlanChangeRequest(StateModel):
    mode: Literal["adjust", "replace", "extract_tasks", "split_task"]
    instructions: str = Field(min_length=1, max_length=12000, strict=True)
    task_id: str | None = None

    @model_validator(mode="after")
    def validate_scope(self):
        if (self.mode == "split_task") != bool(self.task_id):
            raise ValueError("细化任务时必须且只能指定一个任务。")
        return self

    def mode_message(self) -> HumanMessage:
        if self.mode == "extract_tasks":
            return HumanMessage(content="从当前正式计划提取尚未登记的任务，保持计划正文，用 task_changes 的 add 提交。")
        if self.mode == "split_task":
            return HumanMessage(content=f"保持计划正文，细化任务 {self.task_id}。归档该任务并新增至少两个子任务。")
        return HumanMessage(content=(
            "本轮局部调整现有计划，保留用户未要求变更的部分。"
            if self.mode == "adjust" else
            "本轮重新拟定整份计划，保留现有学习目标和用户明确的约束。"
        ))


def parse_plan_change_command(text: str) -> PlanChangeRequest | None:
    parts = text.split(maxsplit=1)
    if not parts:
        return None
    mode = {"/plan-adjust": "adjust", "/plan-replace": "replace"}.get(parts[0])
    if mode is None:
        return None
    if len(parts) == 1 or not parts[1].strip():
        raise ValueError(f"用法：{parts[0]} 具体要求")
    return PlanChangeRequest(mode=mode, instructions=parts[1])


@dataclass(frozen=True)
class PlanDraftResult:
    message: str
    confirmation_plan: str | None
    task_changes: tuple[TaskChange, ...] = ()

    @property
    def status(self) -> Literal["pending", "not_submitted"]:
        return "pending" if self.confirmation_plan is not None else "not_submitted"


def collect_plan_draft(message: str, inbox: PlanProposalInbox, markers: tuple[str, ...]) -> PlanDraftResult:
    """Use only the candidate delivered through the validated program channel."""
    sanitized = sanitize_learning_action_reply(message)
    if sanitized != message or inbox.conflicted:
        return PlanDraftResult(sanitized, None)
    if any(marker in message for marker in markers):
        return PlanDraftResult("本轮使用了旧文本变更标记。请重新提交完整候选。", None)
    pending = inbox.pending
    changes = pending.task_changes if inbox.confirmation_plan and isinstance(pending, WorkspacePlanProposalArgs) else ()
    return PlanDraftResult(message, inbox.confirmation_plan, changes)
