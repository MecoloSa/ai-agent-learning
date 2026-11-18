"""Validated task changes attached to a workspace candidate."""

from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from ai_agent.learning.learning_state import LearningState, LearningTask, StateModel


class TaskChange(StateModel):
    operation: Literal["add", "update", "archive"]
    task_id: str | None = None
    title: str | None = Field(default=None, min_length=1, max_length=200)
    estimated_minutes: int | None = Field(default=None, ge=1, le=1440, strict=True)
    instructions: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def validate_operation(self):
        details = self.model_dump(exclude_none=True, exclude={"operation", "task_id"})
        if self.operation == "add":
            if self.task_id is not None or self.title is None:
                raise ValueError("新增任务需要标题，ID 由程序生成。")
        elif not self.task_id:
            raise ValueError("更新或移除必须指定现有任务 ID。")
        elif self.operation == "archive" and details:
            raise ValueError("移除任务只需要任务 ID。")
        elif self.operation == "update" and not details:
            raise ValueError("更新需要提供要修改的字段，标题不能为空。")
        return self


def validate_task_changes(state: LearningState, changes: tuple[TaskChange, ...]) -> None:
    active_ids = {task.task_id for task in state.tasks if not task.is_cancelled}
    seen = set()
    for change in changes:
        if change.operation != "add":
            key = change.task_id
            if key in seen or key not in active_ids:
                raise ValueError("任务必须存在、未取消，且每份候选中只能操作一次。")
            seen.add(key)


def place_split_children(
    tasks: tuple[LearningTask, ...], parent_id: str, child_ids: tuple[str, ...],
) -> tuple[LearningTask, ...]:
    by_id = {task.task_id: task for task in tasks}
    child_set = set(child_ids)
    ordered = []
    for task in by_id.values():
        if task.task_id not in child_set:
            ordered.append(task)
        if task.task_id == parent_id:
            ordered.extend(by_id[child_id] for child_id in child_ids)
    return tuple(ordered)


def apply_task_changes(
    state: LearningState, changes: tuple[TaskChange, ...], *, split_from_task_id: str | None = None,
) -> LearningState:
    validate_task_changes(state, changes)
    if split_from_task_id is not None and not any(
        change.operation == "archive" and change.task_id == split_from_task_id for change in changes
    ):
        raise ValueError("细化任务必须归档原任务。")
    tasks = {task.task_id: task for task in state.tasks}
    from_plan = tasks[split_from_task_id].from_plan if split_from_task_id is not None else True
    new_ids = []
    for change in changes:
        if change.operation == "add":
            task = LearningTask(task_id=f"task-{uuid4().hex}", title=change.title,
                                estimated_minutes=change.estimated_minutes, instructions=change.instructions or "",
                                from_plan=from_plan)
            tasks[task.task_id] = task
            new_ids.append(task.task_id)
            continue
        key = change.task_id
        task = tasks[key]
        if change.operation == "archive":
            fields = {"is_cancelled": True, "cancellation_reason": "用户采用的计划任务变更。"}
        else:
            fields = change.model_dump(exclude_none=True, exclude={"operation", "task_id"})
            fields["from_plan"] = True
        tasks[key] = LearningTask.model_validate({**task.model_dump(), **fields})
    ordered = tuple(tasks.values())
    if split_from_task_id is not None:
        ordered = place_split_children(ordered, split_from_task_id, tuple(new_ids))
    return LearningState.model_validate({**state.model_dump(), "tasks": ordered})
