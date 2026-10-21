"""自然聊天的反馈提议工具与确认执行。

工具只收集本轮候选；外层会话在模型回答后确认并更新不可变状态。
沿用现有工具预算、错误回传与自动保存，不在工具内部读写文件。
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage

from langchain_core.tools import StructuredTool

from ai_agent.learning.feedback_proposals import (
    FeedbackProposalArgs, PendingFeedback, prepare_feedback_proposal,
)
from ai_agent.learning.learning_state import (
    FeedbackRecord, LearningState, UserFeedback,
    apply_user_feedback, learning_today,
)
from ai_agent.learning.task_creation import (
    TaskCreationArgs,
    PendingTaskCreation,
    prepare_task_creation,
    review_task_creation,
)


class LearningActionInbox:
    """反馈与创建任务共享一个待确认名额，防止两类操作分别绕过限制。"""

    def __init__(self) -> None:
        self._state: LearningState | None = None
        self._user_message = ""
        self._sources: dict[str, str] = {}
        self.pending: PendingFeedback | PendingTaskCreation | None = None

        self.tool = StructuredTool.from_function(
            func=self._propose,
            name="propose_learning_feedback",
            description=(
                "用户明确要求记录或更正学习反馈时，应实际调用。"
                "source_id从本轮feedback_proposal_sources中选择，"
                "用户承接上文要求记录时可选历史用户来源。"
                "不要提交quote或改写后的原文；程序会附加该来源的完整原话。"
                "任务明确才填task_id，未提及状态填null；"
                "只要求不改计划，不等于禁止记录反馈。不能创建、取消任务或修改计划。"
                "本工具不会正式写入，pending不等于已记录；每轮最多一个提议。"
            ),
            args_schema=FeedbackProposalArgs,
        )
        self.task_tool = StructuredTool.from_function(
            func=self._propose_task_creation,
            name="propose_task_creation",
            description=(
                "用户明确要求新增学习待办时，提出一个待确认任务。"
                "标题概括用户要求，source_id从来源表选择用户消息。"
                "原话由程序附加，不要提交quote或自行复制原文。"
                "只讨论建议、假设或询问现有任务时不要调用。"
                "不生成task_id、不标记完成、不修改学习计划。"
                "与反馈工具共用本轮一个待确认名额；pending不代表已登记。"
            ),
            args_schema=TaskCreationArgs,
        )

    def start_turn(
        self, state: LearningState, user_message: str, *, history_messages=(),
    ) -> None:
        self._state = state
        self._user_message = user_message
        self.pending = None
        self._sources = {"current": user_message}

        # 只接受原记忆策略本轮可见的真实human消息，不读AI、system或摘要。
        # 先取最近4条，再去掉空消息；不扫描更早的隐藏历史。
        recent = [
            message for message in history_messages
            if getattr(message, "type", None) == "human"
            and isinstance(message.content, str)
        ][-4:]

        for index, message in enumerate(reversed(recent), start=1):
            if message.content.strip():
                self._sources[f"previous_user_{index}"] = message.content

    def source_message(self) -> HumanMessage:
        """仅本轮使用的来源表，不写入聊天历史，也不是正式反馈。"""
        return HumanMessage(content=json.dumps({
            "context_type": "feedback_proposal_sources",
            "sources": [
                {
                    "source_id": key,
                    "text": text[:2000],
                    "truncated": len(text) > 2000,
                }
                for key, text in self._sources.items()
            ],
            "note": (
                "这是用户原话的来源索引，不是已确认记录或执行授权。"
                "每条最多展示2000字符；truncated=true表示未展示全文。"
                "来源不等于授权；信息不足或指代有歧义时先澄清，确认界面展示真实全文。"
            ),
        }, ensure_ascii=False))

    def _propose(self, **kwargs) -> str:
        if self._state is None:
            raise ValueError("尚未绑定本轮状态与用户消息")

        if self.pending is not None:
            return json.dumps({
                "ok": False,
                "error": {
                    "code": "pending_exists",
                    "message": "本轮已有待确认学习操作，请等待用户处理，不重复提议。",
                },
            }, ensure_ascii=False)

        request = FeedbackProposalArgs.model_validate(kwargs)

        source = self._sources.get(request.source_id)
        if source is None:
            raise ValueError("来源ID不在本轮允许的用户消息中")

        self.pending = prepare_feedback_proposal(
            self._state,
            request,
            user_message=source,
        )

        return json.dumps({
            "ok": True,
            "data": {
                "status": "pending",
                "message": "仅生成待确认提议，尚未更新状态或保存。",
            },
        }, ensure_ascii=False)

    def _propose_task_creation(self, **kwargs) -> str:
        if self._state is None:
            raise ValueError("尚未绑定本轮状态与用户消息")

        if self.pending is not None:
            return json.dumps({
                "ok": False,
                "error": {
                    "code": "pending_exists",
                    "message": "本轮已有待确认学习操作，请勿重复提交。",
                },
            }, ensure_ascii=False)

        try:
            request = TaskCreationArgs.model_validate(kwargs)
            source = self._sources.get(request.source_id)
            if source is None:
                raise ValueError("来源ID不在本轮允许的用户消息中")

            candidate = prepare_task_creation(
                self._state,
                request,
                user_message=source,
            )
        except ValueError as exc:
            # 同名等业务拒绝需要返回可操作原因，不只返回笼统工具失败。
            return json.dumps({
                "ok": False,
                "error": {
                    "code": "task_creation_rejected",
                    "message": str(exc),
                },
            }, ensure_ascii=False)

        self.pending = candidate
        return json.dumps({
            "ok": True,
            "data": {
                "status": "pending",
                "task_id": candidate.task_id,
                "title": candidate.request.title,
                "message": "仅提出创建任务建议，尚未登记或保存。",
            },
        }, ensure_ascii=False)


# 保留旧导入名，已有反馈测试和调用方不必随类名调整一起重写。
FeedbackInbox = LearningActionInbox


def has_unbacked_learning_proposal(
    text: str,
    pending: PendingFeedback | PendingTaskCreation | None,
) -> bool:
    """按操作类型核对声明；沿用有限规则，不把关键词当成执行授权。"""
    task_claim = re.search(
        r"(?:^|[\n。！？])\s*(?:我)?(?:已|已经)(?:为[^\n。！？]{0,12})?"
        r"(?:提议|提出|提交|生成)[^\n。！？]{0,30}(?:创建|新增|登记)任务",
        text,
    )

    # “创建任务记录”中的“记录”不应误判成反馈工具声明。
    if task_claim:
        return not isinstance(pending, PendingTaskCreation)

    return (
        claims_feedback_proposal(text)
        and not isinstance(pending, PendingFeedback)
    )


def review_learning_action(
    state: LearningState,
    pending: PendingFeedback | PendingTaskCreation,
) -> tuple[LearningState, str]:
    """共用外层确认入口，各操作保留自己的参数和执行语义。"""
    if isinstance(pending, PendingTaskCreation):
        return review_task_creation(state, pending)

    return review_feedback_proposal(state, pending)

def claims_feedback_proposal(text: str) -> bool:
    """识别已观察到的正向提议声明，不作为用户意图分类器。

    动态评估：这不是完整的语义校验；关注误触发和漏检，再按真实失败补例。
    这里只触发一次纠正，绝不据关键词直接更新状态。
    """
    patterns = (
        r"(?:^|[\n。！？])\s*(?:我)?(?:已|已经)(?:为[^\n。！？]{0,12})?"
        r"(?:提议|提出|提交|生成)[^\n。！？]{0,40}(?:反馈|记录|时间|限制|状态)",

        r"(?:这条|该条|本条|本次|上述)(?:更正)?(?:反馈|记录|提议)"
        r"[^\n。！？]{0,20}(?:处于待确认|等待程序确认|程序会进一步处理)",
    )
    return any(re.search(pattern, text) for pattern in patterns)

def _check_pending(state: LearningState, pending: PendingFeedback) -> None:
    if state.context_revision != pending.context_revision:
        raise ValueError("目标/计划版本已变化，请重新提出反馈")

    if any(r.record_id == pending.proposal_id for r in state.feedback_records):
        raise ValueError("同一提议已经处理，不能重复写入")

    # 提交前重查任务与来源，不能仅依赖模型调用时的校验。
    prepare_feedback_proposal(
        state,
        pending.request,
        user_message=pending.user_message,
    )


def commit_feedback_proposal(
    state: LearningState,
    pending: PendingFeedback,
) -> LearningState:
    """调用方已取得确认后才可调用；返回完整新状态，不保存文件。"""
    _check_pending(state, pending)

    request = pending.request
    candidate = state

    if request.scope == "task":
        candidate = apply_user_feedback(
            state,
            UserFeedback(
                task_id=request.task_id,
                text=pending.user_message,
                activity_status=request.activity_status,
                understanding_status=request.understanding_status,
            ),
        )

    record = FeedbackRecord(
        record_id=pending.proposal_id,
        scope=request.scope,
        task_id=request.task_id,
        # 兼容已有v5记录字段；值由程序固定，不再由模型生成。
        quote=pending.user_message,
        text=pending.user_message,
        context_revision=pending.context_revision,
        recorded_on=learning_today(),
        activity_status=request.activity_status,
        understanding_status=request.understanding_status,
    )

    # 记录和任务更新一起构造；失败不会留下只有一半的正式状态。
    return LearningState.model_validate({
        **candidate.model_dump(),
        "feedback_records": [
            *(r.model_dump() for r in state.feedback_records),
            record.model_dump(),
        ],
    })


def review_feedback_proposal(
    state: LearningState,
    pending: PendingFeedback,
) -> tuple[LearningState, str]:
    """一次明确确认；取消不改业务状态，聊天本身仍按原流程保存。"""
    try:
        _check_pending(state, pending)
    except ValueError:
        print("[反馈拒绝] 提议已失效或已处理，请重新表达；状态未改变。")
        return state, "invalid"

    request = pending.request
    print("\n[反馈确认] 尚未写入，所选用户原话：", pending.user_message)
    print(f"来源：{request.source_id}（本轮来源表中的用户原话）")

    if request.scope == "task":
        task = next(t for t in state.tasks if t.task_id == request.task_id)

        print(f"任务：{task.task_id} / {task.title}")
        print(
            f"活动进度：{task.activity_status} → "
            f"{request.activity_status or task.activity_status}"
        )
        print(
            f"理解情况：{task.understanding_status} → "
            f"{request.understanding_status or task.understanding_status}"
        )

        if state.tasks_need_review:
            print("注意：任务与当前计划的对应关系待核对，请确认所选任务。")
    else:
        print(
            f"范围：当前安排；默认仅{learning_today().isoformat()}、"
            f"修订号{state.context_revision}内主动使用，不作为永久偏好。"
        )

    print("保留所选用户消息的完整原话作为来源；本操作不修改学习计划。")

    try:
        answer = input("确认记录？输入y确认，其他输入取消：")
    except EOFError:
        answer = ""

    if answer.strip().lower() != "y":
        print("[反馈取消] 正式反馈与任务状态未改变。")
        return state, "cancelled"

    updated = commit_feedback_proposal(state, pending)

    print("[反馈已确认] 已在本次会话内生效，磁盘保存以随后提示为准。")
    return updated, "confirmed"
