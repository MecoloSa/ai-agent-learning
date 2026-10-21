"""学习状态与用户反馈的最小模型。

本模块独立于 LLM、RAG 和聊天记忆：
- 记录任务活动进度，以及用户自报的理解情况；
- 根据明确反馈更新指定任务，不自动推断掌握程度；
- 本阶段仅处理内存中的数据，不负责解析 Markdown、保存文件或调整计划。

动态评估：重点检查任务定位是否准确、未反馈字段是否保留、
阅读完成是否被错误等同于理解，以及非法输入是否被拒绝。
"""

from __future__ import annotations

from typing import Literal
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
    field_validator,
)
from ai_agent.persistence.memory_state import MemorySnapshot


# 活动进度与理解情况分开：
# “阅读完成”不代表“理解清楚”；理解字段也只表示用户自报，不是测评结论。
ActivityStatus = Literal["not_started", "in_progress", "completed"]
UnderstandingStatus = Literal[
    "unknown",
    "self_reported_clear",
    "self_reported_unclear",
]


class StateModel(BaseModel):
    """为状态对象提供统一的数据检查规则。"""

    model_config = ConfigDict(
        extra="forbid",             # 拒绝拼错或未定义的字段，避免静默丢失反馈。
        frozen=True,                # 禁止直接改字段，更新统一走下面的函数。
        str_strip_whitespace=True,  # 先去除首尾空白，再检查字符串长度。
    )

class MaterialBinding(StateModel):
    """本地会话的资料关联，不是模型生成的结论，也不是资料副本。"""

    root_dir: str = Field(min_length=1)
    fingerprint: str = Field(
        pattern=r"^materials-v1:[0-9a-f]{64}$"
    )

    @field_validator("root_dir")
    @classmethod
    def require_absolute_root(cls, value: str) -> str:
        # 只校验路径形式，不在加载数据时扫描磁盘。
        # 即使目录失效，也应允许恢复学习进度。
        if not Path(value).is_absolute():
            raise ValueError("资料关联必须保存绝对路径")
        return value

class LearningTask(StateModel):
    task_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    activity_status: ActivityStatus = "not_started"
    understanding_status: UnderstandingStatus = "unknown"
    latest_feedback: str = ""
    feedback_source: Literal["none", "user_report"] = "none"    #Literal 类型：feedback_source 使用 typing.Literal，严格限制取值

def learning_today() -> date:
    """学习安排按北京时间判断当天，避免操作系统时区差异。"""
    return datetime.now(timezone(timedelta(hours=8))).date()


class FeedbackRecord(StateModel):
    """已确认原话及来源范围；不是客观掌握证明或计划变更授权。"""

    record_id: str = Field(min_length=1)
    scope: Literal["task", "current_context"]
    task_id: str | None
    quote: str = Field(min_length=1)
    text: str = Field(min_length=1)
    context_revision: int = Field(ge=1, strict=True)
    recorded_on: date
    activity_status: ActivityStatus | None
    understanding_status: UnderstandingStatus | None
    source: Literal["user_report"] = "user_report"

class LearningState(StateModel):
    # 保留原字段名，但语义明确为“当前生效值”。
    learning_goal: str = Field(min_length=1)
    plan_markdown: str = Field(min_length=1)

    initial_learning_goal: str = Field(min_length=1)
    initial_plan_markdown: str = Field(min_length=1)

    # 这是目标/计划的修订号，不是文件格式版本，也不随普通进度反馈递增。
    context_revision: int = Field(default=1, ge=1, strict=True)
    tasks_need_review: bool = False

    tasks: tuple[LearningTask, ...] = ()

    # 保留已确认原话；未确认提议不进入状态或快照。
    feedback_records: tuple[FeedbackRecord, ...] = ()

    # 恢复所需的本地元数据；旧快照没有该字段时默认为None。
    # 不将目录和指纹混入发给模型的任务状态消息。
    material_binding: MaterialBinding | None = None

    # 保存ConversationMemory的实际状态；
    # 旧快照缺失时不虚构历史。
    conversation_memory: MemorySnapshot | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_initial_context(cls, data: object) -> object:
        """新建状态或读取旧数据时，为缺失的初始记录补值。"""
        if not isinstance(data, dict):
            return data

        # 复制后再补字段，避免改动调用者传入的字典。
        result = dict(data)

        # setdefault 只补缺失字段；后续更新不能覆盖已有初始版本。
        result.setdefault(
            "initial_learning_goal", result.get("learning_goal")
        )
        result.setdefault(
            "initial_plan_markdown", result.get("plan_markdown")
        )
        return result

    @model_validator(mode="after")
    def check_unique_task_ids(self) -> LearningState:
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("任务 task_id 不能重复")
        return self

def register_learning_task(
    state: LearningState,
    task: LearningTask,
) -> LearningState:
    """共用登记入口：完整校验新状态，保留反馈、记忆和目标版本。"""
    return LearningState.model_validate({
        **state.model_dump(),
        "tasks": [*state.tasks, task],
    })

class UserFeedback(StateModel):
    """只接收用户明确表达或确认过的反馈，不直接接收模型猜测。"""

    task_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    # None 表示“用户本次没有反馈该字段”，不是把原状态重置为默认值。
    activity_status: ActivityStatus | None = None
    understanding_status: UnderstandingStatus | None = None


def apply_user_feedback(
    state: LearningState,
    feedback: UserFeedback,
) -> LearningState:
    """返回更新后的新状态；任务不存在则拒绝，不修改传入对象。

    调用方负责确认反馈确实来自用户。这个纯函数不调用模型，
    也不能判断用户是否真的理解；它只执行明确的数据更新。
    """
    if not any(task.task_id == feedback.task_id for task in state.tasks):
        raise ValueError(f"任务不存在：{feedback.task_id}")

    # exclude_none=True 保留“未提及”和“明确更新”之间的区别。
    changes = feedback.model_dump(
        exclude={"task_id", "text"},
        exclude_none=True,
    )
    changes.update(
        latest_feedback=feedback.text,
        feedback_source="user_report",
    )

    updated_tasks = []
    for task in state.tasks:
        task_data = task.model_dump()
        if task.task_id == feedback.task_id:
            task_data.update(changes)
        updated_tasks.append(task_data)

    # model_dump() 将对象转为普通数据；model_validate() 重新检查全部数据。
    # 不直接修改旧对象，也不使用跳过更新校验的快捷复制方式。
    return LearningState.model_validate({
        **state.model_dump(),
        "tasks": updated_tasks,
    })

def update_learning_context(
    state: LearningState,
    *,
    learning_goal: str | None = None,
    plan_markdown: str | None = None,
) -> LearningState:
    """更新当前目标/计划，保留初始版本和所有任务记录。

    None 表示不更新；空白字符串属于非法更新。
    动态评估：无实际变化不增版本；任务进度不能随改计划而被重置。
    """
    changes = {}
    if learning_goal is not None:
        changes["learning_goal"] = learning_goal
    if plan_markdown is not None:
        changes["plan_markdown"] = plan_markdown

    # 先完整校验，确保非法目标/计划不会产生半更新状态。
    candidate = LearningState.model_validate({
        **state.model_dump(),
        **changes,
    })

    changed = (
        candidate.learning_goal != state.learning_goal
        or candidate.plan_markdown != state.plan_markdown
    )
    if not changed:
        return state

    # 保守标记：目标或计划变更后，不假定旧任务仍准确对应新版本。
    # 已有待核对标记不会被后续反馈或再次修改自动清除。
    return LearningState.model_validate({
        **candidate.model_dump(),
        "context_revision": state.context_revision + 1,
        "tasks_need_review": state.tasks_need_review or bool(state.tasks),
    })
