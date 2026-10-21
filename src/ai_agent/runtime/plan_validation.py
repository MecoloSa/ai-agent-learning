"""学习计划的轻量结构检查，不评价事实正确性或引用支持度。

支持常见“任务1：...”段落和独立字段行；表格等布局暂仅提醒人工核对。
检查结果不修改模型原文，不触发新的模型调用。
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("ai_agent.plan_validation")

_TASK = re.compile(r"^任务\s*\d+(?:\s*[:：、.．]|\s|$)")
_DAY = re.compile(r"^(?:Day\s*\d+\b|第\s*\d+\s*天)", re.I)

_FIELD = re.compile(
    r"^(学习内容|学习主题|具体行动|学习行动|完成标志|验收标准|"
    r"证据位置|参考资料)\s*[:：]\s*(.*)$"
)

_ALIASES = {
    "学习内容": ("学习内容", "学习主题"),
    "具体行动": ("具体行动", "学习行动"),
    "完成标志": ("完成标志", "验收标准"),
}


def _visible_lines(text: str) -> list[str]:
    """提取用于检查的展示行；保留原始文本不变。"""
    lines = []
    fence_char = ""
    fence_size = 0

    for raw in text.splitlines():
        fence = re.match(r"^\s*(`{3,}|~{3,})", raw)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_size = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_size:
                fence_char = ""
            continue

        if fence_char:
            continue

        # 仅清理常见展示符号；代码块中的示例不算真实任务。
        line = re.sub(r"^\s*(?:#{1,6}\s+|[-*+]\s+)", "", raw)
        lines.append(
            line.replace("**", "").replace("__", "").strip()
        )

    return lines


def inspect_plan_format(text: str) -> dict:
    """返回可序列化诊断；未识别不等于模型一定没有提供内容。"""
    lines = _visible_lines(text)
    blocks: list[list[str]] = []
    current: list[str] | None = None

    for line in lines:
        if _TASK.match(line):
            current = []
            blocks.append(current)
        elif _DAY.match(line):
            # 防止下一天没有任务标题时，内容被算进前一天的任务。
            current = None
        elif current is not None:
            current.append(line)

    if not blocks:
        return {
            "status": "unrecognized",
            "task_count": 0,
            "warnings": [
                "未识别到任务段落；可能使用了表格或其他布局，请人工核对。"
            ],
        }

    warnings = []

    for number, block in enumerate(blocks, 1):
        markers = [
            (index, match.group(1), match.group(2))
            for index, line in enumerate(block)
            if (match := _FIELD.match(line))
        ]

        populated = set()

        for pos, (index, name, tail) in enumerate(markers):
            end = (
                markers[pos + 1][0]
                if pos + 1 < len(markers)
                else len(block)
            )

            # 字段内容可写在冒号后，也可写在后续多行。
            # 下一字段开始时停止，不能借用另一个字段的内容。
            content = "\n".join(
                [tail, *block[index + 1:end]]
            ).strip(" \t\n-*_:：")

            # 单独的引用不是完成标志或行动内容，避免空字段被引用“填满”。
            content = re.sub(
                r"\[source[^\]]*\]", "", content
            ).strip(" \t\n-*_:：")

            if content:
                populated.add(name)

        for field, aliases in _ALIASES.items():
            if not populated.intersection(aliases):
                warnings.append(
                    f"第{number}个任务未识别到非空的“{field}”。"
                )

        # 这里只识别引用标记存在，不核验来源、页码或正文支持度。
        if not re.search(
            r"\[source\s*=\s*[^\]\s][^\]]*\]",
            "\n".join(block),
        ):
            warnings.append(
                f"第{number}个任务未识别到资料引用标记。"
            )

    # 动态评估：先观察误报，暂不因Markdown变体拦截或自动重试。
    # 无警告仅表示这些结构信号存在，不代表内容真实、完整或可执行。
    return {
        "status": (
            "warnings"
            if warnings
            else "no_structural_warnings"
        ),
        "task_count": len(blocks),
        "warnings": warnings,
    }


class PlanExecutionError(RuntimeError):
    """计划结果不可用；与分析失败分开，供CLI和评测识别。"""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(f"{reason}: {message}")


def finalize_plan_output(
    value: object,
    *,
    trace: dict | None = None,
) -> str:
    """所有计划返回路径共用；空文本失败，格式问题仅提示。"""
    if not isinstance(value, str) or not value.strip():
        if trace is not None:
            trace["plan_output_check"] = {
                "status": "invalid_text",
                "warnings": [],
            }

        raise PlanExecutionError(
            "invalid_plan_output",
            "模型未返回非空学习计划；停止后续交互与导出。",
        )

    report = inspect_plan_format(value)

    if trace is not None:
        trace["plan_output_check"] = report

    if report["warnings"]:
        # 终端最多展示前三条；完整诊断保存在审计对象中。
        # 软提醒不增加模型往返，也不改写原计划。
        logger.warning(
            "计划结构需核对：status=%s tasks=%d issues=%s",
            report["status"],
            report["task_count"],
            report["warnings"][:3],
        )
    else:
        logger.debug(
            "计划结构检查完成：tasks=%d；未核验事实或引用支持度",
            report["task_count"],
        )

    return value
