"""将当前学习状态组织为普通问答的临时上下文。

只读取状态，不调用模型、不修改任务、不读写快照。
状态消息仅用于本轮请求，不写入聊天历史，避免旧状态不断累积。
用户自报和反馈原文属于参考数据，不是经过测评的掌握结论。
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from ai_agent.learning.learning_state import LearningState, learning_today


def build_learning_state_message(state: LearningState) -> HumanMessage:
    """生成当前任务状态的数据消息，不重复携带整份学习计划。"""
    data = {
        "context_type": "current_learning_state",
        "context_revision": state.context_revision,
        "tasks_need_review": state.tasks_need_review,

        # 保守范围：当前安排反馈只在记录当天、同一计划版本内主动使用。
        # 过期记录仍在/state和快照中，不删除，也不当成永久用户偏好。
        "current_context_feedback": [
            record.model_dump(
                mode="json",
                # 新记录quote=text，省去重复正文；旧版不同的原话片段继续保留。
                exclude={"quote"} if record.quote == record.text else set(),
            )
            for record in state.feedback_records
            if record.scope == "current_context"
            and record.recorded_on == learning_today()
            and record.context_revision == state.context_revision
        ],
        "tasks": [
            {
                "task_id": task.task_id,
                "title": task.title,
                "activity_status": task.activity_status,
                "understanding_status": task.understanding_status,
                "latest_feedback": task.latest_feedback,
                "feedback_source": task.feedback_source,
            }
            for task in state.tasks
        ],
    }

    # 状态是用户侧参考数据，不提升为系统指令。
    # JSON 保留字段边界，但不是防提示注入的绝对安全保证。
    # 动态评估：检查回答是否利用最新困难、是否把自报误当成客观掌握。
    return HumanMessage(
        content=json.dumps(data, ensure_ascii=False)
    )
