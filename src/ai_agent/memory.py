"""短期记忆策略：全量保留 / 滑动窗口 / 摘要压缩。"""

from __future__ import annotations

# 注解详见： https://claude.ai/chat/112e64d0-6c97-4c67-8c70-dcc090dd8ecb

from dataclasses import dataclass, field
from typing import Literal
import logging

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from .config import create_llm, get_settings

logger = logging.getLogger("ai_agent.memory")

Strategy = Literal["full", "window", "summary"]

def estimate_tokens(messages: list[BaseMessage]) -> int:
    """粗略估算：先用字符数近似，够用即可，别在这一步过度设计。
    更准确的做法是换成 tiktoken，但百炼模型分词规则不完全一致，
    与其追求精确，不如先把"触发压缩"这个机制跑通。"""
    return sum(len(str(m.content)) for m in messages) // 2

@dataclass
class ConversationMemory:
    system_prompt: str
    strategy: Strategy = "window"
    max_chars: int = field(
        default_factory=lambda: get_settings().memory_summary_trigger_chars
    )
    window_size: int = field(
        default_factory=lambda: get_settings().memory_window_size
    )
    messages: list[BaseMessage] = field(default_factory=list)
    summary: str = ""
    # system_prompt 的历史快照：plan 或 learning_goal 任一项被真正修改前，
    # 把"即将被覆盖的旧 system_prompt 整段"存一笔。
    # 这两个字段都属于"决策类"信息（模型产出/用户表达的意图），更新方式相同
    # （整体重写 system_prompt），所以共用同一份历史列表，不需要分开记录。
    # 这份历史独立于 full/window/summary 三种压缩策略之外，不会因为窗口滑走
    # 或摘要压缩而丢失，用于回答"最初的计划/目标是什么、改过几次"这类问题。
    prompt_history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages = [SystemMessage(content=self.system_prompt)]

    def add_user(self, text: str) -> None:
        """先记录用户问题，不在半轮对话中间压缩。"""
        self.messages.append(HumanMessage(content=text))

    def add_ai(self, text: str) -> None:
        """AI 回答完成后，再按完整问答轮次进行压缩。"""
        self.messages.append(AIMessage(content=text))
        if self.strategy == "summary":
            self._maybe_compress()

    def update_system_message(
            self, new_content: str, *, plan_changed: bool = False, goal_changed: bool = False
    ) -> None:
        """替换第 0 条 system 消息的内容。

        用于 plan / learning_goal 等决策类信息在对话过程中被更新之后，把新内容
        同步进后续对话的 system_prompt，而不需要重建整个 ConversationMemory 对象——
        历史消息（messages[1:]）和已经生成的摘要（self.summary）都会被原样保留，
        三种策略（full/window/summary）都会从下一次 get_messages_for_llm 开始
        自动使用新的 system 内容。

        注意：analysis、material_context 不通过这个方法更新——它们是资料分析阶段
        一次性产出的"证据类"内容，不应该由对话模型整体重写，需要补充时应通过
        工具调用（见 main.py 中绑定给交互模式的只读工具）在当轮对话里临时读取，
        而不是替换这里的 system_prompt。

        plan_changed / goal_changed：分别标记这次更新是否伴随计划、学习目标的
        实质性变化。只要有一项为 True：
        1) 把即将被覆盖的旧 system_prompt 整段存进 prompt_history；
        2) 如果 summary 策略下已经生成过摘要，给摘要打一个"可能已过时"的标注——
           避免摘要里残留的旧计划/旧目标细节，和马上生效的新内容互相矛盾。
        """
        if plan_changed or goal_changed:
            self.prompt_history.append(self.system_prompt)
            if self.summary:
                changed_what = "、".join(
                    name for name, flag in (("学习计划", plan_changed), ("学习目标", goal_changed)) if flag
                )
                self.summary += (
                    f"\n【提示：以上摘要生成与{changed_what}更新之前"
                    "其中提到的相关细节可能已过时，请始终以当前 system 消息中的内容为准。】"
                )
        self.system_prompt = new_content
        self.messages[0] = SystemMessage(content=new_content)

    def get_messages_for_llm(self) -> list[BaseMessage]:
        system = self.messages[0]
        rest = self.messages[1:]

        if self.strategy == "full":
            return [system] + rest
        if self.strategy == "window":
            return [system] + rest[-self.window_size:]
        if self.strategy == "summary":
            if not self.summary:
                return [system] + rest
            summary_msg = SystemMessage(
                content=f"【更早的对话摘要，供背景参考】\n{self.summary}"
            )
            return [system, summary_msg] + rest

        raise ValueError(f"未知策略：{self.strategy}")

    def _maybe_compress(self) -> None:
        rest = self.messages[1:]
        # 这里只对固定的 system_prompt 以外的内容作更新
        total_chars = sum(len(str(m.content)) for m in rest)
        if total_chars < self.max_chars:
            return
        to_compress, keep = rest[:-2], rest[-2:]  # 最近一问一答不压缩
        if len(to_compress) < 2:
            return

        transcript_chars = sum(
            len(str(message.content))
            for message in to_compress
        )

        logger.info(
            "触发对话摘要：total_chars=%d "
            "compressed_messages=%d input_chars=%d "
            "retained_messages=%d",
            total_chars,
            len(to_compress),
            transcript_chars,
            len(keep),
        )
        llm = create_llm()
        transcript = "\n".join(f"[{m.type}] {m.content}" for m in to_compress)
        result = llm.invoke([
            SystemMessage(
                content=(
                    "把下面的多轮对话压缩成一段简洁的背景摘要，供后续对话参考。"
                    # 当前学习目标和最新计划的完整内容已经单独存在于 system 消息里，
                    # 不需要在摘要里复述目标或计划的具体天数/任务/日期——
                    # 复述既是重复劳动，一旦计划/目标后续再被更新，还会和新版本产生冲突。
                    "当前学习目标和最新计划的完整内容已经在 system 消息中单独提供，"
                    "你完全不需要在摘要里复述目标或计划的具体内容，只需要总结："
                    "用户提出过的修改意图和最终决定、双方讨论过的取舍原因、"
                    "尚未解决或用户还在犹豫的问题、约定好的后续行动。"
                    "按时间顺序组织成要点，直接输出摘要正文，"
                    "不要任何解释性开头；去掉寒暄、重复内容与无信息量的问候。"
                )
            ),
            HumanMessage(content=transcript),
        ])
        self.summary = (self.summary + "\n" + result.content).strip() if self.summary else result.content
        self.messages = [self.messages[0]] + keep
        logger.info(
            "对话摘要完成：summary_chars=%d",
            len(self.summary),
        )

        settings = get_settings()
        if settings.log_payloads:
            logger.debug(
                "对话摘要预览：%r",
                self.summary[:settings.log_preview_chars],
            )
