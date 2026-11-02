"""评测脚本的分阶段输入；不生成回答，不决定是否提交状态。"""
from __future__ import annotations

from typing import Literal


class ReplayMismatch(RuntimeError):
    """实际业务请求输入的阶段，与脚本预期不一致。"""


class ReplayInput:
    def __init__(self, inputs: list[str]) -> None:
        # 使用副本，避免消费脚本时修改已加载案例，影响下一次对照。
        self._inputs = tuple(inputs)
        self._index = 0
        self.events: list[dict[str, object]] = []

    def _take(self, stage: Literal["chat", "plan_confirmation"]) -> str:
        if self._index >= len(self._inputs):
            raise ReplayMismatch(f"输入耗尽：index={self._index} stage={stage}")
        value = self._inputs[self._index]
        token = value.strip().lower()
        is_confirmation = token in {"v", "y", "n"}
        if (stage == "plan_confirmation") != is_confirmation:
            # 不消费错误输入，也不自动补n或EOF，让运行器明确记录脚本不匹配。
            # 只记录位置与阶段，不把用户正文写进异常或事件。
            raise ReplayMismatch(f"输入阶段不匹配：index={self._index} stage={stage}")
        self.events.append({"index": self._index, "stage": stage})
        self._index += 1
        return value

    def read_chat(self) -> str:
        return self._take("chat")

    def read_plan_confirmation(self) -> str:
        return self._take("plan_confirmation")

    def assert_finished(self) -> None:
        # 会话提前返回也不能误报成功；剩余输入说明仍有预期路径未覆盖。
        if self._index != len(self._inputs):
            raise ReplayMismatch(f"脚本尚未完成：remaining={len(self._inputs) - self._index}")