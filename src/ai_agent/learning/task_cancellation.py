"""取消已登记任务的底层状态操作。

只返回新状态，不调用模型、不确认、不写文件。调用方必须先取得用户授权；
后续聊天工具仍需沿用来源校验与确认，不把调用本函数当作授权证明。
"""
from dataclasses import dataclass
from pydantic import Field

from ai_agent.learning.confirmation import read_confirmation
from ai_agent.learning.learning_state import LearningState, StateModel


class TaskCancellation(StateModel):
    task_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


def cancel_learning_task(
    state: LearningState,
    request: TaskCancellation,
) -> LearningState:
    """保留任务和历史，仅标记取消；重复取消拒绝，避免覆盖原取消依据。"""
    target = next(
        (task for task in state.tasks if task.task_id == request.task_id),
        None,
    )
    if target is None:
        raise ValueError(f"任务不存在：{request.task_id}")
    if target.is_cancelled:
        raise ValueError(
            "任务已经取消，不重复写入；如需更正或恢复，应使用明确操作"
        )

    tasks = []
    for task in state.tasks:
        data = task.model_dump()
        if task.task_id == request.task_id:
            data.update(
                is_cancelled=True,
                cancellation_reason=request.text,
            )
        tasks.append(data)

    # 不覆盖latest_feedback、理解状态或原活动进度；取消不是一次学习完成。
    # 动态评估：比较前后状态，除这两个字段外其他业务数据应保持不变。
    return LearningState.model_validate({
        **state.model_dump(),
        "tasks": tasks,
    })

class TaskCancellationArgs(StateModel):
    """模型只选择已有任务与来源；不能提交取消原文或确认结果。"""
    task_id: str = Field(min_length=1)
    source_id: str = Field(default="current", min_length=1)


@dataclass(frozen=True)
class PendingTaskCancellation:
    request: TaskCancellationArgs
    user_message: str
    context_revision: int


def prepare_task_cancellation(
    state: LearningState,
    request: TaskCancellationArgs,
    *,
    user_message: str,
) -> PendingTaskCancellation:
    # 复用底层校验；返回的新状态只用于预检，这里不采用、不保存。
    cancel_learning_task(
        state,
        TaskCancellation(task_id=request.task_id, text=user_message),
    )
    return PendingTaskCancellation(
        request,
        user_message,
        state.context_revision,
    )


def commit_task_cancellation(
    state: LearningState,
    pending: PendingTaskCancellation,
) -> LearningState:
    """仅由取得确认的调用方采用返回值；版本检查不等于完整并发控制。"""
    if state.context_revision != pending.context_revision:
        raise ValueError("目标/计划版本已变化，请重新提出取消请求")

    return cancel_learning_task(
        state,
        TaskCancellation(
            task_id=pending.request.task_id,
            text=pending.user_message,
        ),
    )


def review_task_cancellation(
    state: LearningState,
    pending: PendingTaskCancellation,
    *,
    confirmation_trace: dict[str, str] | None = None,
) -> tuple[LearningState, str]:
    if confirmation_trace is not None:
        confirmation_trace.clear()
    try:
        candidate = commit_task_cancellation(state, pending)
    except ValueError:
        print("[取消任务拒绝] 提议已失效或任务不可取消，请重新提出；状态未改变。")
        return state, "invalid"

    task = next(
        item for item in state.tasks
        if item.task_id == pending.request.task_id
    )
    print(f"\n[取消任务确认] 尚未执行：{task.task_id} / {task.title}")
    print(
        f"所选用户原话：{pending.user_message}\n"
        f"来源：{pending.request.source_id}"
    )
    print(
        f"将标记为已取消；保留原进度={task.activity_status}、"
        f"理解={task.understanding_status}及反馈。"
    )
    print("不删除任务，不修改Markdown计划；取消不等于完成。")

    if state.tasks_need_review:
        print("注意：任务与当前计划的对应关系待核对，请确认所选任务。")

    if not read_confirmation(
        "确认取消该任务？输入y确认，其他输入放弃本次提议：",
        trace=confirmation_trace,
    ):
        print("[取消提议未采用] 任务原状态不变。")
        return state, "cancelled"

    print("[任务已取消] 本次会话内已生效；磁盘保存以随后提示为准。")
    return candidate, "confirmed"
