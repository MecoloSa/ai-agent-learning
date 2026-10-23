"""自然语言创建任务：参数、候选、确认与纯状态更新。

模型只选择任务标题和来源ID；原文与ID由程序提供，默认进度由LearningTask提供。
不修改Markdown计划、不执行学习任务、不自动判断完成，不在本模块保存文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import unicodedata
from uuid import uuid4
from pydantic import Field

from ai_agent.learning.confirmation import read_confirmation

from ai_agent.learning.learning_state import (
    LearningState,
    LearningTask,
    StateModel,
    register_learning_task,
)


class TaskCreationArgs(StateModel):
    """模型不能提交task_id、完成状态或任意目标/计划字段。"""

    title: str = Field(min_length=1)
    source_id: str = Field(default="current", min_length=1)


@dataclass(frozen=True)
class PendingTaskCreation:
    request: TaskCreationArgs
    user_message: str
    context_revision: int
    task_id: str = field(default_factory=lambda: f"task-{uuid4().hex}")


def _check_creation(
    state: LearningState,
    request: TaskCreationArgs,
    user_message: str,
) -> None:
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("指定用户消息不能为空")

    def normalized(title: str) -> str:
        # 仅处理全半角、大小写和连续空白，不进行语义相似度猜测。
        return " ".join(
            unicodedata.normalize("NFKC", title).split()
        ).casefold()

    existing = next(
        (
            task for task in state.tasks
            if normalized(task.title) == normalized(request.title)
        ),
        None,
    )
    if existing is not None:
        raise ValueError(
            f"已存在同名任务：{existing.task_id} / {existing.title}。"
            "请使用已有任务；确需新一轮练习时明确不同的任务标题。"
        )


def prepare_task_creation(
    state: LearningState,
    request: TaskCreationArgs,
    *,
    user_message: str,
) -> PendingTaskCreation:
    _check_creation(state, request, user_message)
    return PendingTaskCreation(
        request=request,
        user_message=user_message,
        context_revision=state.context_revision,
    )


def commit_task_creation(
    state: LearningState,
    pending: PendingTaskCreation,
) -> LearningState:
    if pending.context_revision != state.context_revision:
        raise ValueError("目标/计划已变化，请重新提出创建任务")

    _check_creation(state, pending.request, pending.user_message)

    # 同一候选始终使用同一ID；重复执行由状态的ID唯一性检查拒绝。
    # 不采用模型给出的进度，确保新任务默认not_started/unknown。
    return register_learning_task(
        state,
        LearningTask(
            task_id=pending.task_id,
            title=pending.request.title,
        ),
    )


def review_task_creation(
    state: LearningState,
    pending: PendingTaskCreation,
    *,
    confirmation_trace: dict[str, str] | None = None,
) -> tuple[LearningState, str]:
    if confirmation_trace is not None:
        confirmation_trace.clear()
    # 提前构造候选只做校验；未确认时绝不替换外层state。
    try:
        candidate = commit_task_creation(state, pending)
    except ValueError:
        print("[任务创建拒绝] 候选已失效或重复，当前状态未改变。")
        return state, "invalid"

    print("\n[创建任务确认] 尚未登记：", pending.request.title)
    print(f"候选ID：{pending.task_id}")
    print(
        f"所选用户原话：{pending.user_message}\n"
        f"来源：{pending.request.source_id}"
    )
    print(f"所属当前目标：{state.learning_goal}")
    print("默认进度=not_started，理解情况=unknown。")
    print("只增加登记任务，不修改学习计划，不表示已开始或完成。")

    if not read_confirmation(
        "输入y确认创建，其他输入取消：",
        trace=confirmation_trace,
    ):
        print("[任务创建取消] 未新增任务。")
        return state, "cancelled"

    print(
        f"[任务已登记] {pending.task_id}；"
        "磁盘保存以随后提示为准。"
    )
    return candidate, "confirmed"
