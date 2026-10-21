"""交互式学习状态命令。

处理明确的任务登记、反馈更新、状态查询与快照保存。
不调用 LLM、不推断掌握程度、不调整计划，也不自动恢复会话。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from pathlib import Path

from ai_agent.config import resolve_project_path
from ai_agent.persistence.state_store import save_learning_state

from ai_agent.learning.learning_state import (
    LearningState,
    LearningTask,
    UserFeedback,
    apply_user_feedback,
    register_learning_task,
)

def new_current_state_path() -> Path:
    """为一次运行分配独立当前存档路径，不立即创建文件。

    每次恢复另开一个会话分支，不能把恢复来源当成覆盖目标。
    不默认跨进程共享路径，避免两个终端相互覆盖。
    """
    saved_at = datetime.now(
        timezone(timedelta(hours=8))
    ).strftime("%y%m%d_%H%M%S")

    return resolve_project_path(
        f".agents_data/learning_states/current/"
        f"{saved_at}_{uuid4().hex}.json"
    )

def save_state_snapshot(state: LearningState) -> Path:
    """按统一命名规则保存新快照；失败交给调用方处理。"""
    saved_at = datetime.now(
        timezone(timedelta(hours=8))
    ).strftime("%y%m%d_%H%M%S")

    # 时间便于查找，UUID 避免同一秒连续保存发生重名。
    filename = f"{saved_at}_{uuid4().hex}.json"
    path = resolve_project_path(
        f".agents_data/learning_states/{filename}"
    )

    # 复用上一轮的可靠写入，不重新实现磁盘操作。
    return save_learning_state(state, path)

def handle_learning_command(
    state: LearningState,
    user_input: str,
) -> tuple[LearningState, str | None]:
    """返回（状态，提示文本）。

    提示文本为 None：不是状态命令，交回原有模型问答流程。
    提示文本非 None：命令已处理，调用方应打印后跳过模型调用。

    命令失败时返回原状态，不留下只更新了一部分的数据。
    """
    parts = user_input.strip().split(maxsplit=1)
    if not parts or not parts[0].startswith("/"):
        return state, None

    command = parts[0]
    payload = parts[1] if len(parts) == 2 else ""

    if command not in {"/task", "/feedback", "/state", "/save-state"}:
        return state, (
            "未知状态命令，请使用 /task、/feedback、/state 或 /save-state。"
        )

    try:
        if command == "/save-state":
            if payload:
                raise ValueError("/save-state 不需要参数")

            saved_path = save_state_snapshot(state)

            return state, (
                f"学习状态快照已保存：{saved_path}\n"
                "已保存当前目标、计划与登记状态；"
                "含资料关联时可用--resume-state继续问答；"
                "包含当前已保存的对话记忆（若有），不包含旧工具执行证据。"
            )

        if command == "/state":
            if payload:
                raise ValueError("/state 不需要参数")

            review_notice = (
                "注意：目标或计划变更后，已有任务与当前版本的对应关系待核对。\n"
                if state.tasks_need_review
                else ""
            )
            return state, (
                f"当前学习状态（目标/计划修订号={state.context_revision}）：\n"
                + review_notice
                + state.model_dump_json(indent=2)
            )

        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("命令参数必须是 JSON 对象，即 {...}")

        if command == "/task":
            # 登记入口只接收身份信息，不能顺便把任务标成已完成。
            # 活动进度与理解情况必须通过明确的反馈入口更新。
            if set(data) - {"task_id", "title"}:
                raise ValueError("/task 只允许 task_id 和 title 字段")

            task = LearningTask.model_validate(data)

            # 保留整个原状态，只替换任务表。
            # 不逐项重建，否则容易重置初始记录、修订号和待核对标记。
            updated = register_learning_task(state, task)
            return updated, f"已登记任务：{task.task_id} / {task.title}"

        feedback = UserFeedback.model_validate(data)
        updated = apply_user_feedback(state, feedback)

        # next() 找到指定任务后立即返回；更新函数已保证该任务存在。
        task = next(
            task for task in updated.tasks
            if task.task_id == feedback.task_id
        )
        return updated, (
            f"已记录用户反馈：{task.task_id}\n"
            f"  活动进度：{task.activity_status}\n"
            f"  理解情况：{task.understanding_status}\n"
            f"  来源：{task.feedback_source}"
        )

    except (ValueError, OSError) as exc:
        # ValueError：输入或数据校验失败。
        # OSError：目录权限、磁盘写入或文件已存在等文件系统问题。
        # 动态评估：失败不得改变内存状态，也不得提示“保存成功”。
        return state, f"状态命令未执行：{exc}"
