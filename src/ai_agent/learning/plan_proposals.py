"""计划候选提交工具：只保存本轮pending，不确认、不保存文件。"""
from __future__ import annotations

import json
from langchain_core.tools import StructuredTool
from pydantic import Field

from ai_agent.learning.learning_state import (
    LearningState,
    StateModel,
    update_learning_context,
)


class PlanProposalArgs(StateModel):
    plan_markdown: str = Field(min_length=1, strict=True)


class PlanProposalInbox:
    def __init__(
        self,
        plan_start: str,
        plan_end: str,
        goal_markers: tuple[str, str],
    ):
        # 这些标记仅用于识别与旧入口的冲突，不再用于生成候选展示。
        self._markers = (plan_start, plan_end, *goal_markers)
        self._state: LearningState | None = None
        self.pending: PlanProposalArgs | None = None
        self.confirmation_plan: str | None = None
        self.conflicted = False

        self.tool = StructuredTool.from_function(
            func=self._propose,
            name="propose_plan_change",
            args_schema=PlanProposalArgs,
            description=(
                "用户要求调整计划时提交完整待确认正文，保留未要求改变的部分。"
                "不修改目标、任务或反馈。仅登记本轮候选，程序负责展示、确认和保存；"
                "正文不包含更新标记。"
            ),
        )

    def start_turn(self, state: LearningState) -> None:
        self._state = state
        self.pending = None
        self.confirmation_plan = None
        self.conflicted = False

    @staticmethod
    def _error(code: str, message: str) -> str:
        return json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            ensure_ascii=False,
        )

    def _propose(self, plan_markdown: str) -> str:
        if self._state is None:
            return self._error(
                "turn_not_started",
                "本轮尚未绑定学习状态。",
            )
        if self.pending is not None:
            return self._error(
                "pending_exists",
                "本轮已有计划候选，未覆盖原候选。",
            )

        request = PlanProposalArgs(plan_markdown=plan_markdown)
        if any(marker in request.plan_markdown for marker in self._markers):
            return self._error(
                "reserved_marker",
                "仅提交计划正文，不能嵌套目标或计划更新标记。",
            )

        # 沿用纯状态校验；产生临时候选不代表采用，更不写磁盘。
        if update_learning_context(
            self._state,
            plan_markdown=request.plan_markdown,
        ) is self._state:
            return self._error(
                "unchanged_plan",
                "计划与当前正文相同，无需提交变更。",
            )

        self.pending = request
        return json.dumps(
            {
                "ok": True,
                "data": {
                    "status": "pending",
                    "action": "plan_change",
                },
            },
            ensure_ascii=False,
        )

    def prepare_delivery(
        self,
        model_text: str,
        *,
        has_learning_action: bool = False,
    ) -> PlanProposalArgs | None:
        """选择可交付候选；结果通过字段传递，不生成或解析更新标记。"""
        self.confirmation_plan = None
        self.conflicted = False
        if self.pending is None:
            return None

        self.conflicted = has_learning_action or any(
            marker in model_text for marker in self._markers
        )
        return None if self.conflicted else self.pending