"""学习状态快照的保存与读取。

快照包含学习状态及可选对话记忆数据，不包含模型客户端或可执行对象。
先完整写入临时文件，再发布：
历史快照禁止覆盖；当前存档允许用完整的新文件替换旧文件。

本实现避免普通写入失败留下半截正式文件，但不承诺突然断电后的
绝对持久性。进程被强制终止时可能残留临时文件，读取时仍需校验。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from ai_agent.learning.learning_state import LearningState, StateModel

logger = logging.getLogger("ai_agent.state_store")


class StateSnapshot(StateModel):
    # v7增加明确的反馈更正关系；旧记录缺失时默认没有更正关系。
    # 兼容读取v1至v6；新写入使用v7，旧版程序不保证能读取v7。
    schema_version: Literal[
        "learning-state-v1",
        "learning-state-v2",
        "learning-state-v3",
        "learning-state-v4",
        "learning-state-v5",
        "learning-state-v6",
        "learning-state-v7",
    ] = "learning-state-v7"
    state: LearningState

def save_learning_state(state: LearningState, path: Path) -> Path:
    """保存不可覆盖的历史快照；已有路径会导致保存失败。"""
    return _write_learning_state(
        state, path, replace_existing=False,
    )


def save_current_learning_state(
    state: LearningState,
    path: Path,
) -> Path:
    """保存当前存档，允许用完整的新状态替换已有文件。

    调用方必须提供本次会话专用路径，不能传入用户选中的历史快照路径。
    本函数只负责可靠写入，不负责选择存档路径。
    """
    return _write_learning_state(
        state, path, replace_existing=True,
    )


def _write_learning_state(
    state: LearningState,
    path: Path,
    *,
    replace_existing: bool,
) -> Path:
    """共用序列化、临时写入和清理流程，只区分最终发布方式。"""
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

        if replace_existing:
            # 当前存档：新内容已完整写入并关闭，再替换正式文件。
            # 不先清空旧文件；替换失败时，旧存档仍保留。
            # 临时文件与正式文件同目录，避免跨文件系统移动。
            os.replace(temporary_path, path)
        else:
            # 历史快照：目标已存在时必须失败，不能覆盖旧版本。
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

    # 动态评估：成功后应能完整恢复；发布失败不能破坏原存档。
    # 当前存档要求调用方保证单写入者，不提供多进程冲突合并。
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
