"""工作流执行尝试的可读摘要存档。

检查点负责恢复，本模块负责解释每次执行发生了什么。
只保存允许的摘要字段，不保存资料正文、模型消息或异常全文。
使用独立文件保留各次尝试，不覆盖旧记录。
"""
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID

from ai_agent.config import resolve_project_path

logger = logging.getLogger("ai_agent.workflow_audit")

DEFAULT_AUDIT_DIR = ".agents_data/workflow_traces"


def _pick(data: dict, names: tuple[str, ...]) -> dict:
    """只选择已声明的字段，避免未来新增正文时被整体写入。"""
    return {
        name: data[name]
        for name in names
        if name in data
    }


def save_workflow_attempt(
    trace: dict,
    db_path: str | Path,
    *,
    directory: str | Path | None = None,
) -> Path:
    """保存一次尝试；失败向调用方抛异常，不决定业务是否成功。"""
    # 文件名只使用合法UUID，不能让其他轨迹字段变成路径。
    run_id = UUID(str(trace["run_id"])).hex

    attempt = _pick(trace, (
        "trace_version",
        "run_id",
        "thread_id",
        "execution_kind",
        "status",
        "failed_node",
        "error_type",
    ))

    summary_fields = (
        "analysis_chars",
        "material_context_chars",
        "plan_chars",
        "materials_valid",
    )
    events = []

    for event in trace.get("events", []):
        item = _pick(event, (
            "node",
            "status",
            "elapsed_ms",
            "error_type",
        ))
        for direction in ("input", "output"):
            summary = event.get(direction)
            if isinstance(summary, dict):
                item[direction] = _pick(
                    summary, summary_fields
                )
        events.append(item)

    attempt["events"] = events
    payload = {
        "schema_version": "workflow-attempt-v1",
        "checkpoint_db": str(resolve_project_path(db_path)),
        "attempt": attempt,
    }

    root = resolve_project_path(
        DEFAULT_AUDIT_DIR if directory is None else directory
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{run_id}.json"
    temporary_path = None

    try:
        # 先写完整临时文件，避免普通写入失败留下半截正式记录。
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root,
            prefix=".workflow-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(
                payload, stream,
                ensure_ascii=False, indent=2,
            )
            stream.flush()
            os.fsync(stream.fileno())

        # 与已有学习状态快照采用相同发布方式：
        # 目标已存在就失败，不能覆盖之前的运行证据。
        os.link(temporary_path, target)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("工作流轨迹临时文件清理失败")

    return target
