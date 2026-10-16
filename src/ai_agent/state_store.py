"""学习状态快照的保存与读取。

快照包含格式版本和 LearningState，不包含聊天历史或模型客户端。
先完整写入临时文件，再发布为正式快照；不覆盖已有快照。

本实现避免普通写入失败留下半截正式文件，但不承诺突然断电后的
绝对持久性。进程被强制终止时可能残留临时文件，读取时仍需校验。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from .learning_state import LearningState, StateModel

logger = logging.getLogger("ai_agent.state_store")


class StateSnapshot(StateModel):
    # 新保存默认使用 v2；允许读取此前的 v1。
    # schema_version 是文件格式版本，context_revision 是目标/计划修订号。
    schema_version: Literal[
        "learning-state-v1",
        "learning-state-v2",
    ] = "learning-state-v2"
    state: LearningState

def save_learning_state(state: LearningState, path: Path) -> Path:
    """完整写入后发布新快照；失败抛出异常，不覆盖已有文件。"""
    path = Path(path)

    snapshot = StateSnapshot.model_validate({
        "state": state.model_dump(),
    })
    payload = snapshot.model_dump_json(indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        # 临时文件放在目标目录，确保发布时位于同一文件系统。
        # delete=False：先关闭文件，再发布，兼容 Windows 的文件占用限制。
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".learning-state-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)

            # flush 把 Python 缓冲区交给操作系统；
            # fsync 请求系统将文件数据同步到存储设备。
            # 任一步失败，都不能继续发布正式快照。
            stream.flush()
            os.fsync(stream.fileno())

        # 硬链接：为同一份完整文件数据增加正式文件名，不再次复制内容。
        # 目标已存在时失败，因此也能防止并发保存覆盖旧快照。
        # 不使用 os.replace，因为它会覆盖已有目标。
        os.link(temporary_path, path)

    finally:
        if temporary_path is not None:
            try:
                # 发布成功后只删除临时名字，正式快照仍然存在。
                temporary_path.unlink(missing_ok=True)
            except OSError:
                # 清理失败不能掩盖原始保存异常，
                # 也不能把已成功发布的快照误报为保存失败。
                logger.warning(
                    "临时快照清理失败：%s",
                    temporary_path,
                    exc_info=True,
                )

    # 动态评估：正式快照必须能恢复；失败时不能出现半截正式文件。
    return path


def load_learning_state(path: Path) -> LearningState:
    """读取并校验快照；不存在、损坏或格式不兼容时拒绝加载。"""
    payload = Path(path).read_text(encoding="utf-8")

    # 不只检查 JSON 能否解析，还检查版本、任务字段、状态值和 ID 唯一性。
    # 校验失败直接抛出异常，不静默返回空状态，避免掩盖数据损坏。
    snapshot = StateSnapshot.model_validate_json(payload)

    if snapshot.schema_version == "learning-state-v1":
        logger.warning(
            "读取旧版快照：以文件中已有目标/计划作为恢复基线，"
            "初始字段已补齐；无法找回旧版未保存的后续聊天修改。"
        )

    return snapshot.state