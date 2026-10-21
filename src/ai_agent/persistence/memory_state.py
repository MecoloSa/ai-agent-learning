"""可持久化的对话记忆数据，不包含模型客户端或可执行对象。

记录现有策略及其实际状态；消息仅允许human/ai文本。
旧系统提示词历史仅归档，不能作为当前system指令直接重放。
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class MemoryData(BaseModel):
    # 不自动去除文本空白，避免改变原来的对话内容。
    model_config = ConfigDict(extra="forbid", frozen=True)


class StoredMessage(MemoryData):
    role: Literal["human", "ai"]
    content: str


class MemorySnapshot(MemoryData):
    strategy: Literal["full", "window", "summary"]
    max_chars: int = Field(gt=0, strict=True)
    window_size: int = Field(gt=0, strict=True)

    messages: tuple[StoredMessage, ...] = ()
    summary: str = ""
    prompt_history: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_complete_turns(self) -> MemorySnapshot:
        if len(self.messages) % 2:
            raise ValueError("只能保存已经结束的完整问答")

        for index, message in enumerate(self.messages):
            expected = "human" if index % 2 == 0 else "ai"
            if message.role != expected:
                raise ValueError("问答消息必须按human/ai交替排列")

        return self
