"""将当前学习状态组织为普通问答的临时上下文。

只读取状态，不调用模型、不修改任务、不读写快照。
状态消息仅用于本轮请求，不写入聊天历史，避免旧状态不断累积。
用户自报和反馈原文属于参考数据，不是经过测评的掌握结论。
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from ai_agent.learning.learning_state import LearningState, learning_today

def select_context_feedback(state: LearningState) -> dict[str, list[dict]]:
    """读取当天安排反馈，按来源版本分组，不判断有效期、不更新状态。"""
    groups = {
        "current_context_feedback": [],
        "previous_context_feedback": [],
    }
    today = learning_today()

    # 先解析更正关系，再做日期/版本筛选，避免旧记录在后续读取时重新出现。
    replaced = {
        r.supersedes_record_id
        for r in state.feedback_records
        if r.supersedes_record_id is not None
    }

    groups["feedback_replacements"] = [
        {
            "old_record_id": r.supersedes_record_id,
            "new_record_id": r.record_id,
        }
        for r in state.feedback_records
        if r.scope == "current_context"
        and r.recorded_on == today
        and r.context_revision <= state.context_revision
        and r.supersedes_record_id is not None
    ]

    for record in state.feedback_records:
        if record.record_id in replaced:
            continue
        if record.scope != "current_context" or record.recorded_on != today:
            continue
        if record.context_revision > state.context_revision:
            continue

        key = (
            "current_context_feedback"
            if record.context_revision == state.context_revision
            else "previous_context_feedback"
        )
        groups[key].append(record.model_dump(
            mode="json",
            exclude={"quote"} if record.quote == record.text else set(),
        ))

    # 当天只是本轮的读取窗口，不是“其他日期的事实均已失效”。
    # 动态评估：改计划后记录应换组而非消失；不能改写来源或自动采用旧约束。
    return groups

def build_learning_state_message(state: LearningState) -> HumanMessage:
    """从真实状态生成本轮唯一的当前正式状态视图。"""
    data = {
        "context_type": "current_learning_state",
        "context_revision": state.context_revision,
        "current_date": learning_today().isoformat(),
        # 当前值由状态对象提供，不从计划标题或历史助手回答推断。
        "learning_goal": state.learning_goal,
        "plan_markdown": state.plan_markdown,
        "plan_status": "active",
        "tasks_need_review": state.tasks_need_review,

        **select_context_feedback(state),
        "feedback_usage_note": (
            "text是确认采用的反馈正文，quote是来源用户原话；两者不同时不能把text当作直接引语。"
            "feedback_replacements表示已确认的更正关系，被替代内容仅作历史。"
            "分组只记录来源版本；当前适用性需结合用户请求核对，不按版本号判定有效期。"
            "本消息只展示当天未被替代的记录，不是完整历史；冲突或适用性不明时先核对。"
        ),
        # 登记任务用于说明历史进度，不推断它与计划活动的一一对应关系。
        "tasks_role": (
            "from_plan=true 标识已确认的计划任务。未标记的任务独立登记；"
            "取消的任务只保留历史。具体任务的估时和状态以任务字段为准，计划正文提供整体安排。"
            if any(task.from_plan for task in state.tasks) else
            "已登记任务及历史进度，不等于当前计划的活动或今日安排；"
            "未取消也不代表今天要执行。实际安排读取当前计划正文。"
        ),
        "tasks": [task.model_dump(mode="json") for task in state.tasks],
    }

    # 状态是用户侧参考数据，不提升为系统指令。
    # JSON 保留字段边界，但不是防提示注入的绝对安全保证。
    # 动态评估：检查回答是否利用最新困难、是否把自报误当成客观掌握。
    return HumanMessage(
        content=json.dumps(data, ensure_ascii=False)
    )
