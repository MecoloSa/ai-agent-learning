"""检查点的资料版本保护。

只比较受支持文件的相对路径和原始字节，不解析PDF、不调用模型。
指纹不是资料副本，也不能提供并发写入期间的原子快照。
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

from ai_agent.resources.materials import iter_material_files, resolve_material_root

logger = logging.getLogger("ai_agent.checkpoint_guard")

# 这是“检查点兼容契约”的版本，不是项目发布版本。
# 状态结构、分析产物含义或节点衔接发生不兼容变化时才升级。
WORKFLOW_REVISION = "learning-graph-v1"


def require_workflow_revision(expected: object) -> None:
    """拒绝没有兼容版本或版本不匹配的检查点，不自动迁移旧状态。"""
    if not isinstance(expected, str) or not expected:
        raise ValueError(
            "检查点缺少工作流兼容版本；请新建任务"
        )

    if expected != WORKFLOW_REVISION:
        # 动态评估：版本不兼容必须在业务节点执行前拒绝，
        # 不能把旧版本字段直接改成当前版本后继续运行。
        raise ValueError(
            "工作流兼容版本与检查点不一致；请新建任务"
        )


def material_fingerprint(material_dir: str) -> str:
    root = resolve_material_root(material_dir)
    paths = list(iter_material_files(root))
    rows = []
    started = time.perf_counter()

    for path in paths:
        # 不为资料目录之外的链接目标计算指纹。
        if not path.resolve().is_relative_to(root):
            raise ValueError("资料文件越出指定目录，拒绝计算指纹")

        before = path.stat()
        with path.open("rb") as stream:
            # 分块读取原始字节，不将整本书一次性加载到内存。
            digest = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
        after = path.stat()

        keys = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_ino")
        if any(
            getattr(before, key) != getattr(after, key)
            for key in keys
        ):
            raise ValueError(
                "计算指纹期间资料变化，请停止编辑后重试"
            )

        rows.append((
            path.relative_to(root).as_posix(),
            digest,
        ))

    if paths != list(iter_material_files(root)):
        raise ValueError("计算指纹期间文件清单变化，请重试")

    payload = json.dumps(
        {"version": "materials-v1", "files": rows},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    result = "materials-v1:" + hashlib.sha256(payload).hexdigest()

    logger.debug(
        "检查点资料指纹完成：files=%d elapsed_ms=%d",
        len(paths),
        round((time.perf_counter() - started) * 1000),
    )
    return result


def require_material_fingerprint(
    material_dir: str,
    expected: object,
) -> None:
    if not isinstance(expected, str) or not expected:
        raise ValueError(
            "检查点缺少资料指纹，不能恢复；请新建任务"
        )

    if material_fingerprint(material_dir) != expected:
        # 动态评估：资料变化必须拒绝混用旧分析，不能静默重新绑定指纹。
        raise ValueError(
            "资料版本与检查点不一致；请新建任务"
        )
