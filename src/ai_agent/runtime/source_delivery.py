"""从真实结构化读取结果展示来源，不判断回答是否被资料支持。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class TurnSourceLedger:
    """仅收集本轮成功读取的位置，不把旧来源当成本轮重新核验。"""

    sources: dict[str, list[str]] = field(default_factory=dict)

    def _add(self, citation: object, notes: list[str]) -> None:
        if not isinstance(citation, str) or not citation.strip():
            return
        stored = self.sources.setdefault(citation, [])
        for note in notes:
            if note not in stored:
                stored.append(note)

    def observe(self, tool_name: str, result: str, *, executed: bool) -> None:
        if not executed or tool_name not in {
            "read_material_lines", "read_learning_file", "read_pdf_pages",
            "retrieve_learning_context",
            "inspect_document_structure",
            "delegate_learning_task",
        }:
            return
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            # 未迁移的文本协议不猜测来源，继续沿用原引用路径。
            return
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            return
        data = payload.get("data")
        if not isinstance(data, dict):
            return

        if tool_name == "delegate_learning_task":
            for source in data.get("evidence_catalog", {}).values():
                notes = ["该位置仅返回部分文字"] if source.get("partial") is True else []
                self._add(source.get("citation"), notes)
            return

        if tool_name == "inspect_document_structure":
            notes = {
                "whole_document": [],
                "verified_outline": ["已核对书签落点的标题"],
            }.get(data.get("mode"), ["结构识别线索，尚未完整读取章节正文"])
            for source in data.get("sources", []):
                self._add(source.get("citation"), notes)
            return

        if tool_name == "retrieve_learning_context":
            sources = data.get("sources")
            if not isinstance(sources, list):
                return
            notes = []
            if data.get("omitted_blocks", 0) > 0:
                notes.append("本次检索有未返回的证据块")
            for source in sources:
                if isinstance(source, dict):
                    self._add(source.get("citation"), notes)
            return

        notes = ["本次调用有截断"] if data.get("truncated") is True else []
        if tool_name == "read_pdf_pages":
            pages = data.get("pages")
            if not isinstance(pages, list):
                return
            for page in pages:
                if not isinstance(page, dict):
                    continue
                page_notes = list(notes)
                if page.get("partial") is True:
                    page_notes.append("该页仅返回部分文字")
                if page.get("text_status") == "no_text":
                    page_notes.append("未提取到非空文字，不代表页面空白")
                # citation由读取程序生成，不从PDF正文提取类似标记。
                self._add(page.get("citation"), page_notes)
            return

        lines = data.get("lines", [])
        if isinstance(lines, list) and any(
            isinstance(line, dict) and line.get("partial") is True
            for line in lines
        ):
            notes.append("存在未完整返回的行")
        self._add(data.get("citation"), notes)

    def render(self) -> str:
        if not self.sources:
            return ""
        rows = []
        for citation, notes in self.sources.items():
            suffix = "；" + "；".join(notes) if notes else ""
            rows.append(f"- {citation}{suffix}")
        return (
            "【程序整理的本轮读取来源】\n"
            "以下列出本轮工具实际返回的资料位置，不包含预载资料，"
            "也不表示每项来源都支持回答的全部论断。\n"
            + "\n".join(rows)
        )
