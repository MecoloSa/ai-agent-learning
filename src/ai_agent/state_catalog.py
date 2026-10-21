"""只读查看本项目的学习存档，不恢复会话、不调用模型、不修改文件。

只扫描learning_states根目录和current目录；不遍历资料或临时目录。
展示的是磁盘中已成功保存的状态，不是仍在运行的会话内存。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import resolve_project_path
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.state_store import load_learning_state


@dataclass(frozen=True)
class StateEntry:
    """列表中的一个文件记录；不是新的学习状态存储格式。"""

    path: Path
    kind: str
    modified_at: datetime
    state: LearningState | None
    error_type: str | None = None


def list_state_entries(
    root: Path,
    *,
    limit: int = 10,
) -> list[StateEntry]:
    """按修改时间列出最近文件；坏文件可见，但不能冒充可恢复状态。"""
    if limit < 1:
        raise ValueError("limit必须至少为1")

    root = Path(root).resolve()
    if not root.exists():
        return []

    candidates = []

    for directory, kind in (
        (root, "历史/旧版"),
        (root / "current", "当前存档"),
    ):
        if not directory.exists():
            continue

        # iterdir只检查这一层。
        # 目录读取失败应上报，不能假装目录为空。
        for path in directory.iterdir():
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix.lower() != ".json"
            ):
                continue

            candidates.append((
                path.stat().st_mtime,
                path,
                kind,
            ))

    # 按最后修改时间降序；时间相同时按路径排序，保持展示顺序稳定。
    candidates.sort(key=lambda item: (-item[0], str(item[1])))

    entries = []

    # 先限制数量，再读取完整JSON，避免展示十条却解析全部聊天存档。
    # 动态评估：最新文件损坏时应明确展示，不能悄悄跳到旧存档。
    for modified, path, kind in candidates[:limit]:
        try:
            state = load_learning_state(path)
            error_type = None
        except (OSError, ValueError) as exc:
            state = None
            error_type = type(exc).__name__

        entries.append(StateEntry(
            path=path,
            kind=kind,
            # 先明确时间戳属于UTC，再转换成本机时区，避免含糊的时区解释。
            modified_at=datetime.fromtimestamp(
                modified,
                tz=timezone.utc,
            ).astimezone(),
            state=state,
            error_type=error_type,
        ))

    return entries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="只读列出最近的学习存档",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="显示数量，默认10",
    )
    args = parser.parse_args(argv)

    if args.limit < 1:
        parser.error("--limit必须至少为1")

    root = resolve_project_path(".agents_data/learning_states")

    try:
        entries = list_state_entries(root, limit=args.limit)
    except (OSError, ValueError) as exc:
        # 不打印完整异常，避免意外带出存档中的用户文本。
        parser.error(
            f"存档目录读取失败：{type(exc).__name__}"
        )

    if not entries:
        print(f"暂无存档：{root}")
        return

    for index, entry in enumerate(entries, start=1):
        stamp = entry.modified_at.strftime(
            "%Y-%m-%d %H:%M:%S %z"
        )
        print(
            f"[{index}] {entry.kind} | 最后修改：{stamp}"
        )

        if entry.state is None:
            print(
                f"  无法读取：{entry.error_type}；"
                "请勿据此自动恢复。"
            )
        else:
            state = entry.state

            # 只显示识别存档所需的信息，不输出计划、聊天和反馈正文。
            goal = " ".join(state.learning_goal.split())
            goal = goal[:80] + (
                "…" if len(goal) > 80 else ""
            )

            strategy = (
                state.conversation_memory.strategy
                if state.conversation_memory is not None
                else "未保存"
            )

            print(f"  目标：{goal}")
            print(
                f"  修订号={state.context_revision} "
                f"任务数={len(state.tasks)} "
                f"记忆策略={strategy}"
            )

        print(f"  路径：{entry.path}")

    print(
        "核对目标后，将对应完整路径传给--resume-state；"
        "编号仅用于本次展示。"
    )


if __name__ == "__main__":
    main()
