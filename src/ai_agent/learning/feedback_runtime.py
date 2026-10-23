"""自然聊天的反馈提议工具与确认执行。

工具只收集本轮候选；外层会话在模型回答后确认并更新不可变状态。
沿用现有工具预算、错误回传与自动保存，不在工具内部读写文件。
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage

from langchain_core.tools import StructuredTool

from ai_agent.learning.confirmation import read_confirmation, confirmation_details

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
from ai_agent.learning.task_cancellation import (
    TaskCancellationArgs,
    PendingTaskCancellation,
    prepare_task_cancellation,
    review_task_cancellation,
)
from ai_agent.learning.confirmation_history import (
    ConfirmationHistoryArgs, read_confirmation_history,
)
from ai_agent.learning.feedback_history import (
    FeedbackHistoryArgs,
    read_feedback_history,
)

PendingLearningAction = (
    PendingFeedback | PendingTaskCreation | PendingTaskCancellation
)

def sanitize_learning_action_reply(response_text: str) -> str:
    """模型正文不得包含程序回执标记；只过滤文本，不执行或取消学习操作。"""
    reserved_markers = (
        "【程序学习操作处理结果：",
        "【程序学习操作结果数据】",
        "【程序确认结果数据】",
    )
    if not any(marker in response_text for marker in reserved_markers):
        return response_text

    # 不解析或采用模型生成的decision、task_id、修订号，也不打印其正文。
    # 整条回复降级为提示，真实pending仍由外层确认。
    logging.getLogger(__name__).warning(
        "模型回复混入程序保留的学习操作结果标记，已隐藏模型正文。"
    )
    return (
        "本轮模型回复包含只能由程序生成的操作结果标记，已隐藏该条模型回复。"
        "若下方出现确认界面，请核对后选择；是否已生效以程序处理结果和/state为准。"
    )

def build_learning_action_history(
    response_text: str,
    state: LearningState,
    pending: PendingLearningAction,
    decision: str,
    *,
    confirmation_trace: dict[str, str] | None = None,
) -> str:
    """将实际处理结论附到聊天记忆，不执行更新，也不证明磁盘保存成功。"""
    if decision not in {"confirmed", "cancelled", "invalid"}:
        raise ValueError("未知学习操作处理结果")

    # 调用方提供的是模型正文；先去除冒充程序回执的内容，
    # 再由本函数根据真实状态追加唯一的程序结果。
    response_text = sanitize_learning_action_reply(response_text)

    if isinstance(pending, PendingTaskCreation):
        action = "task_creation"
        task_id = pending.task_id
    elif isinstance(pending, PendingTaskCancellation):
        action = "task_cancellation"
        task_id = pending.request.task_id
    else:
        action = (
            "task_feedback" if pending.request.scope == "task"
            else "current_context_feedback"
        )
        task_id = pending.request.task_id

    # 必须读取处理后的真实状态，不能把候选字段当成已经采用的结果。
    # 取消创建时查不到新ID；取消反馈时应显示原任务，不能显示候选进度。
    task = next((item for item in state.tasks if item.task_id == task_id), None)
    result = {
        "context_type": "learning_action_result",
        **confirmation_details(decision, confirmation_trace),
        "action": action,
        "decision": decision,
        "task_id": task_id,
        "task_after": task.model_dump(mode="json") if task is not None else None,
        "context_revision": state.context_revision,
        "proposal_pending": False,
        "persistence": "not_checked_here",
    }

    if isinstance(pending, PendingFeedback):
        proposed_text = pending.request.replacement_text or pending.user_message
        # 仅用于关联“哪条候选未采用”；不能把它混入正式反馈读取视图。
        result["feedback_proposal"] = {
            "record_id": pending.proposal_id,
            "supersedes_record_id": pending.request.supersedes_record_id,
            "text": proposed_text[:1000],
            "text_truncated": len(proposed_text) > 1000,
        }

    # 原回复保留为讨论背景，末尾追加程序事实；不清空历史或修改摘要策略。
    # 动态评估：检查下一轮是否仍将已登记任务说成待确认，而非只看JSON存在。
    return (
        response_text
        + f"\n【程序学习操作处理结果：{decision}；"
        "本次确认流程已结束，不能继续当成待确认提议；"
        "confirmed仅表示会话内生效，磁盘保存以自动保存结果为准】"
        + "\n【程序学习操作结果数据】\n"
        + json.dumps(result, ensure_ascii=False)
    )

class LearningActionInbox:
    """反馈与创建任务共享一个待确认名额，防止两类操作分别绕过限制。"""

    def __init__(self) -> None:
        self._state: LearningState | None = None
        self._user_message = ""
        self._sources: dict[str, str] = {}
        self.pending: PendingLearningAction | None = None

        self.tool = StructuredTool.from_function(
            func=self._propose,
            name="propose_learning_feedback",
            description=(
                '提出待确认反馈，不直接写入。用户要求记录或更正时调用，不负责创建/取消任务或修改计划。'
                'source_id选择包含事实的用户来源，原话由程序附加；任务不明先澄清，未表达状态填null。'
                '普通补充不填更正字段；更正安排反馈须指定旧记录ID和完整replacement_text，保留未要求改变的条件。'
                '更正正文是候选，不是用户原话；有歧义先澄清。每轮最多一个待确认操作。'
            ),
            args_schema=FeedbackProposalArgs,
        )
        self.confirmation_history_tool = StructuredTool.from_function(
            func=self._read_confirmation_history,
            name="read_confirmation_history",
            description=(
                '只读查询当前存档保留的程序确认历史，核对反馈、计划或任务候选的采用/否决结果。'
                '回执不在当前上下文时使用；候选摘要不是正式状态，缺失记录或原因时说明无法核实。'
            ),
            args_schema=ConfirmationHistoryArgs,
        )
        self.history_tool = StructuredTool.from_function(
            func=self._read_feedback_history,
            name="read_feedback_history",
            description=(
                '只读查询当前存档的正式安排反馈及更正关系，可按记录ID、记录日期筛选并分页。'
                '不查询候选处理结果或其他存档；截断、空结果和历史适用性须如实说明。'
            ),
            args_schema=FeedbackHistoryArgs,
        )
        self.task_tool = StructuredTool.from_function(
            func=self._propose_task_creation,
            name="propose_task_creation",
            description=(
                '用户要求新增待办时提出一个待确认任务，提交标题和用户source_id。'
                'ID及初始状态由程序生成；不修改计划，不标记开始/完成。仅讨论不调用，与其他提议共用一个待确认名额。'
            ),
            args_schema=TaskCreationArgs,
        )
        self.cancellation_tool = StructuredTool.from_function(
            func=self._propose_task_cancellation,
            name="propose_task_cancellation",
            description=(
                '用户要求取消登记任务时提出待确认建议，提交已有task_id和用户source_id；对象不明先澄清。'
                '不删除、不标记完成、不修改计划或恢复任务。讨论或否决某次提议不等于取消任务，共用一个待确认名额。'
            ),
            args_schema=TaskCancellationArgs,
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

    def _read_confirmation_history(self, **kwargs) -> str:
        if self._state is None:
            raise ValueError("尚未绑定本轮学习状态")
        return read_confirmation_history(
            self._state, ConfirmationHistoryArgs.model_validate(kwargs),
        )

    def _read_feedback_history(self, **kwargs) -> str:
        if self._state is None:
            raise ValueError("尚未绑定本轮学习状态")

        return read_feedback_history(
            self._state,
            FeedbackHistoryArgs.model_validate(kwargs),
        )

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

    def _propose_task_cancellation(self, **kwargs) -> str:
        if self._state is None:
            raise ValueError("尚未绑定本轮状态与用户消息")

        if self.pending is not None:
            return json.dumps({
                "ok": False,
                "error": {
                    "code": "pending_exists",
                    "message": "本轮已有待确认学习操作，请先处理。",
                },
            }, ensure_ascii=False)

        try:
            request = TaskCancellationArgs.model_validate(kwargs)
            source = self._sources.get(request.source_id)
            if source is None:
                raise ValueError("来源ID不在本轮允许的用户消息中")

            candidate = prepare_task_cancellation(
                self._state,
                request,
                user_message=source,
            )
        except ValueError as exc:
            return json.dumps({
                "ok": False,
                "error": {
                    "code": "task_cancellation_rejected",
                    "message": str(exc),
                },
            }, ensure_ascii=False)

        self.pending = candidate
        return json.dumps({
            "ok": True,
            "data": {
                "status": "pending",
                "task_id": request.task_id,
                "message": "仅提出取消任务建议，尚未取消或保存。",
            },
        }, ensure_ascii=False)


# 保留旧导入名，已有反馈测试和调用方不必随类名调整一起重写。
FeedbackInbox = LearningActionInbox


def has_unbacked_learning_proposal(
    text: str,
    pending: PendingLearningAction | None,
) -> bool:
    """按操作类型核对声明；沿用有限规则，不把关键词当成执行授权。"""
    cancellation_claim = re.search(
        r"(?:^|[\n。！？])\s*(?:我)?(?:已|已经)(?:为[^\n。！？]{0,12})?"
        r"(?:提议|提出|提交|生成)[^\n。！？]{0,30}取消[^\n。！？]{0,20}任务",
        text,
    )
    if cancellation_claim:
        return not isinstance(pending, PendingTaskCancellation)
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
    pending: PendingLearningAction,
    *,
    confirmation_trace: dict[str, str] | None = None,
) -> tuple[LearningState, str]:
    """共用外层确认入口，各操作保留自己的参数和执行语义。"""
    if isinstance(pending, PendingTaskCancellation):
        return review_task_cancellation(state, pending, confirmation_trace=confirmation_trace)

    if isinstance(pending, PendingTaskCreation):
        return review_task_creation(state, pending, confirmation_trace=confirmation_trace)

    return review_feedback_proposal(state, pending, confirmation_trace=confirmation_trace)

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

    if pending.request.scope == "task":
        # 预检真正的更新规则，但不采用返回值；
        # 避免确认后才发现取消任务不能改进度。
        apply_user_feedback(
            state,
            UserFeedback(
                task_id=pending.request.task_id,
                text=pending.user_message,
                activity_status=pending.request.activity_status,
                understanding_status=pending.request.understanding_status,
            ),
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
        # quote始终是程序绑定的用户原话；text是用户确认采用的反馈正文。
        # 普通反馈两者相同；更正正文可结合被指向的旧记录，但不能冒充直接引语。
        quote=pending.user_message,
        text=(
            request.replacement_text
            if request.supersedes_record_id is not None
            else pending.user_message
        ),
        context_revision=pending.context_revision,
        recorded_on=learning_today(),
        supersedes_record_id=request.supersedes_record_id,
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
    *,
    confirmation_trace: dict[str, str] | None = None,
) -> tuple[LearningState, str]:
    """一次明确确认；取消不改业务状态，聊天本身仍按原流程保存。"""
    if confirmation_trace is not None:
        confirmation_trace.clear()
    try:
        _check_pending(state, pending)
    except ValueError:
        print("[反馈拒绝] 提议已失效或已处理，请重新表达；状态未改变。")
        return state, "invalid"

    request = pending.request
    if request.supersedes_record_id is not None:
        target = next(
            r for r in state.feedback_records
            if r.record_id == request.supersedes_record_id
        )
        print(f"\n[反馈更正确认] 被更正记录：{target.record_id}")
        print(f"原记录正文：{target.text}")
        print(f"更正后完整正文：{request.replacement_text}")
        print("请核对未要求改变的条件是否保留、是否新增了你没说过的内容。")
        print("本操作替代整条反馈；正文中遗漏的其余条件也将从当前读取视图退出，发现问题请取消。")
        print("确认后原记录保留为历史，不再作为未被更正的安排反馈提供；本操作不修改计划。")

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
            f"范围：当前安排；记录日期{learning_today().isoformat()}、"
            f"记录时修订号{state.context_revision}；读取按日期窗口与来源版本分组，不作为永久偏好。"
        )

    print("保留所选用户消息的完整原话作为来源；更正时另存以上确认正文，本操作不修改学习计划。")

    if not read_confirmation(
        "确认记录？输入y确认，其他输入取消：",
        trace=confirmation_trace,
    ):
        print("[反馈取消] 正式反馈与任务状态未改变。")
        return state, "cancelled"

    updated = commit_feedback_proposal(state, pending)

    print("[反馈已确认] 已在本次会话内生效，磁盘保存以随后提示为准。")
    return updated, "confirmed"
