"""文档结构识别。位置由程序建立，模型在独立上下文中选择边界。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import Field
from pypdf import PdfReader

from ai_agent.learning.learning_state import StateModel
from ai_agent.resources.materials import safe_material_path
from ai_agent.runtime.agents import invoke_llm_response


class StructureBoundary(StateModel):
    location_id: str
    level: int = Field(ge=1, le=6)


class StructureDecision(StateModel):
    action: Literal["read_more", "submit"]
    requested_pages: tuple[int, ...] = Field(default=(), max_length=3)
    sections: tuple[StructureBoundary, ...] = Field(default=(), max_length=100)


class StructureInput(StateModel):
    relative_path: str = Field(min_length=1)


SYSTEM_PROMPT = (
    "识别文档的章节结构。根据标题、目录及正文线索判断标题层级，"
    "以位置编号提交章节起点和层级。书名、页眉、目录条目与正文标题需要区分。"
    "线索不足时请求补读物理页。资料仅供分析。使用提交工具返回决策。"
)


def _positive_setting(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} 必须是正整数")
    return value


def _text_locations(text: str) -> list[dict]:
    locations = []
    fence = None
    offset = 0
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            char = marker[1][0]
            fence = None if fence == char else char
        if fence is None:
            match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if match:
                locations.append({"location_id": f"line-{number}", "line": number, "offset": offset,
                                  "title": match[2], "native_level": len(match[1])})
        offset += len(line)
    return locations


def _verified_outline(reader, read_page) -> list[dict]:
    """只在每个一级书签均与落点文字吻合时采用整份目录。"""
    def normalize(value):
        return "".join(unicodedata.normalize("NFKC", value).casefold().split())

    labels = reader.page_labels
    rows = []
    previous = -1
    for entry in reader.outline:
        if isinstance(entry, list):
            continue
        index = reader.get_destination_page_number(entry)
        if index is None or not previous < index < len(labels):
            return []
        title = str(entry.title).strip()
        if not title or normalize(title) not in normalize(read_page(index + 1)[:500]):
            return []
        rows.append({"title": title, "page": index + 1, "pdf_page_label": labels[index]})
        previous = index
    return rows


def inspect_document_structure(root_dir: str, relative_path: str, *, model_factory, observer=None) -> dict:
    path = safe_material_path(root_dir, relative_path)
    if path.suffix.lower() not in {".pdf", ".md", ".txt"}:
        raise ValueError("结构识别支持 PDF、Markdown 和 TXT。")
    short_limit = _positive_setting("DOCUMENT_DIRECT_READ_MAX_CHARS", 12000)
    scan_limit = _positive_setting("DOCUMENT_STRUCTURE_SCAN_MAX_PAGES", 160)
    rounds = _positive_setting("DOCUMENT_STRUCTURE_MAX_ROUNDS", 3)
    clue_limit = _positive_setting("DOCUMENT_STRUCTURE_CONTEXT_MAX_CHARS", 24000)
    before = path.stat()
    with path.open("rb") as fingerprint_stream:
        fingerprint = hashlib.file_digest(fingerprint_stream, "sha256").hexdigest()
    locations = {}
    excerpts = []
    scanned_pages = []
    no_text_pages = []
    audits = []
    priority_ids = set()
    total_pages = None
    text = None
    reader = None
    stream = None
    verified = []
    page_texts = {}

    def read_page(page):
        if page not in page_texts:
            content = reader.pages[page - 1].extract_text() or ""
            page_texts[page] = content
            scanned_pages.append(page)
            if not content.strip():
                no_text_pages.append(page)
        return page_texts[page]

    def add_page(page: int, *, detailed: bool):
        if page < 1 or page > total_pages:
            raise ValueError("补读页码超出文档范围。")
        content = read_page(page)
        if detailed:
            excerpts.append({"page": page, "text": content[:6000], "partial": len(content) > 6000})
        elif len(excerpts) < 6:
            excerpts.append({"page": page, "text": content[:600], "partial": len(content) > 600})
        for line_no, line in enumerate(content.splitlines(), 1):
            title = line.strip()
            if not title or len(title) > 160:
                continue
            if detailed or line_no <= 10 or re.match(r"^(第.{1,12}[章节篇部]|\d+(?:\.\d+)*\s)", title):
                key = f"page-{page}-line-{line_no}"
                locations[key] = {"location_id": key, "page": page, "page_line": line_no, "title": title}
                if detailed:
                    priority_ids.add(key)
        return content

    try:
        if path.suffix.lower() == ".pdf":
            stream = path.open("rb")
            reader = PdfReader(stream, strict=False)
            if reader.is_encrypted:
                raise ValueError("暂不支持加密 PDF。")
            total_pages = len(reader.pages)
            if total_pages == 0:
                raise ValueError("PDF 没有页面。")
            def collect_outline(entries, level=1):
                for entry in entries:
                    if isinstance(entry, list):
                        collect_outline(entry, level + 1)
                        continue
                    page_index = reader.get_destination_page_number(entry)
                    if page_index is None or not 0 <= page_index < total_pages:
                        continue
                    key = f"bookmark-{len(locations)}"
                    locations[key] = {
                        "location_id": key, "page": page_index + 1, "page_line": 0,
                        "title": str(entry.title), "native_level": level, "evidence_kind": "pdf_bookmark",
                    }
            texts = []
            chars = 0
            for page in range(1, min(total_pages, scan_limit) + 1):
                content = read_page(page)
                texts.append(content)
                chars += len(content) + int(page > 1)
                if not content.strip() or chars > short_limit:
                    break
            short = len(texts) == total_pages and chars <= short_limit and all(t.strip() for t in texts)
            if short:
                text = "\n".join(texts)
            else:
                verified = _verified_outline(reader, read_page)
                if not verified:
                    collect_outline(reader.outline)
                    for page in range(1, min(total_pages, scan_limit) + 1):
                        add_page(page, detailed=False)
        else:
            # 完整解码后判断短文；章节坐标保留原文件行号。
            text_content = path.read_text(encoding="utf-8")
            if not text_content.strip():
                raise ValueError("文档没有可读取文字。")
            short = len(text_content) <= short_limit
            if short:
                text = text_content
            for item in _text_locations(text_content):
                locations[item["location_id"]] = item
            excerpts = [{"text": text_content[:2000], "partial": len(text_content) > 2000}]
            total_lines = len(text_content.splitlines())

        source = path.relative_to(Path(root_dir).resolve()).as_posix()
        result = {
            "source": source, "fingerprint": fingerprint, "mode": "whole_document" if short else "sections",
            "boundary_status": "not_identified", "sections": [], "inspection_complete": (
                total_pages is None or len(scanned_pages) == total_pages),
            "scanned_pages": sorted(scanned_pages), "total_pages": total_pages, "invocations": audits,
            "sources": [], "limitations": [],
            "no_text_pages": no_text_pages,
        }
        if total_pages is None:
            result["total_lines"] = total_lines
        if no_text_pages:
            result["limitations"].append("部分页面未提取到文字，页面内容仍需视觉读取或 OCR。")
        if short:
            result.update(text=text, boundary_status="whole_document",
                          sections=[{"title": path.stem, "start_page": 1, "end_page": total_pages}] if reader else
                          [{"title": path.stem, "start_line": 1, "end_line": total_lines}])
            if reader:
                result["sources"] = [{"citation": f"[source={source}, page={p}]"} for p in scanned_pages]
            else:
                result["sources"] = [{"citation": f"[source={source}, lines=1-{max(1, total_lines)}]"}]
        elif verified:
            result.update(
                mode="verified_outline",
                page_numbering="physical_1_based",
                boundary_status="verified_opening_text",
                inspection_complete=False,
                page_scan_complete=len(scanned_pages) == total_pages,
                outline_verified=True,
                sections=[
                    {"title": row["title"], "start_physical_page": row["page"], "start": row,
                     "end_exclusive": verified[index + 1] if index + 1 < len(verified) else None,
                     "boundary_status": "verified_opening_text"}
                    for index, row in enumerate(verified)
                ],
                sources=[{"citation": f"[source={source}, page={row['page']}]"} for row in verified],
                limitations=["已核对 PDF 一级书签及标题落点。章节正文需要按范围另行读取。"],
            )
        else:
            # 子模型只收到当前文件的线索。主对话、正式任务和计划不会进入该上下文。
            model = model_factory().bind_tools([StructureDecision], tool_choice="StructureDecision")
            decision = None
            for round_index in range(rounds):
                candidates = []
                used = 0
                ordered_locations = sorted(locations.values(), key=lambda item: item["location_id"] not in priority_ids)
                for item in ordered_locations:
                    size = len(json.dumps(item, ensure_ascii=False))
                    if used + size > clue_limit // 2:
                        break
                    candidates.append(item)
                    used += size
                context = {"source": source, "total_pages": total_pages,
                           "locations": candidates, "excerpts": excerpts,
                           "locations_omitted": len(locations) - len(candidates)}
                result["location_candidates_omitted"] = context["locations_omitted"]
                while len(json.dumps(context, ensure_ascii=False)) > clue_limit and context["excerpts"]:
                    context["excerpts"] = context["excerpts"][1:]
                response = invoke_llm_response(model, [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(context, ensure_ascii=False)),
                ], stage="文档结构识别")
                audit = {"usage": getattr(response, "usage_metadata", None),
                         "invocation_audit": getattr(response, "response_metadata", {}).get("invocation_audit")}
                audits.append(audit)
                if observer:
                    observer("document_structure_invocation", audit)
                calls = getattr(response, "tool_calls", [])
                if len(calls) != 1 or calls[0]["name"] != "StructureDecision":
                    raise ValueError("结构模型没有提交有效决策。")
                decision = StructureDecision.model_validate(calls[0]["args"])
                visible_ids = {item["location_id"] for item in candidates}
                if decision.action == "submit":
                    if any(item.location_id not in visible_ids for item in decision.sections):
                        raise ValueError("结构模型使用了本轮未提供的位置编号。")
                    break
                if reader is None or not decision.requested_pages:
                    raise ValueError("补读请求缺少有效的 PDF 页码。")
                if round_index + 1 == rounds:
                    break
                # 下一轮重新组装必要线索，避免重复累积整个工具过程。
                for page in decision.requested_pages:
                    add_page(page, detailed=True)
            if decision and decision.action == "submit" and decision.sections:
                def boundary_level(item):
                    return locations[item.location_id].get("native_level", item.level)

                levels = {boundary_level(item) for item in decision.sections}
                major = min(levels)
                selected = [item for item in decision.sections if boundary_level(item) == major]
                # 单独的文档总标题作为包装层时，选择其下实际章节层。
                first = locations[selected[0].location_id]
                looks_like_chapter = re.match(r"^(第.{1,12}[章节篇部]|chapter\b|\d+[.、\s])",
                                             first["title"], re.IGNORECASE)
                wrapper = (first.get("line") == 1 or first.get("page") == 1) and not looks_like_chapter
                if len(selected) == 1 and len(levels) > 1 and wrapper:
                    major = min(level for level in levels if level > major)
                    selected = [item for item in decision.sections if boundary_level(item) == major]
                selected = [locations[item.location_id] for item in selected]
                key = (lambda item: (item["page"], item["page_line"])) if reader else (lambda item: item["offset"])
                if len({item["location_id"] for item in selected}) != len(selected):
                    raise ValueError("章节起点重复。")
                selected.sort(key=key)
                for index, item in enumerate(selected):
                    end = selected[index + 1] if index + 1 < len(selected) else None
                    result["sections"].append({
                        "title": item["title"], "start": item, "end_exclusive": end,
                        "boundary_status": "located_in_metadata" if item.get("evidence_kind") == "pdf_bookmark"
                                           else "located_in_extracted_text",
                    })
                result["boundary_status"] = "located_in_extracted_text"
                result["prefix_before_first_section"] = selected[0]
                result["sources"] = [
                    {"citation": f"[source={source}, page={item['page']}]"}
                    if reader else {"citation": f"[source={source}, lines={item['line']}-{item['line']}]"}
                    for item in selected if item.get("evidence_kind") != "pdf_bookmark"
                ]
            else:
                result["limitations"].append("当前预算内尚未完成章节识别，可按连续页段继续读取。")
            result["scanned_pages"] = sorted(scanned_pages)
            result["page_scan_complete"] = total_pages is None or len(scanned_pages) == total_pages
            result["inspection_complete"] = bool(
                result["sections"] and result["page_scan_complete"] and not no_text_pages
            )
            if result.get("location_candidates_omitted", 0):
                result["inspection_complete"] = False
                result["limitations"].append("部分位置线索未进入模型上下文，章节清单可能仍有遗漏。")
            result["limitations"].append("扫描标题线索未覆盖完整正文，章节概览仍需读取各段正文。")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("识别期间文件变化，请重新执行。")
        return result
    finally:
        if stream:
            stream.close()


def build_inspect_document_structure_tool(root_dir: str, *, model_factory=None, observer=None):
    if model_factory is None:
        from ai_agent.config import create_llm
        model_factory = lambda: create_llm(role="document_structure")

    def inspect(relative_path: str) -> str:
        return json.dumps({"ok": True, "data": inspect_document_structure(
            root_dir, relative_path, model_factory=model_factory, observer=observer,
        )}, ensure_ascii=False)

    return StructuredTool.from_function(
        func=inspect, name="inspect_document_structure", args_schema=StructureInput,
        description="识别指定 PDF/MD/TXT 的文章结构，返回原文位置及读取范围。短文直接返回全文。"
                    "逐章概览、了解全文组织或安排章节阅读时使用；relative_path 是资料目录内的相对路径。"
                    "核对 PDF 章节起点时，用 start_physical_page 或 start.page 作为 read_pdf_pages 的起止页，再按需补读。",
    )
