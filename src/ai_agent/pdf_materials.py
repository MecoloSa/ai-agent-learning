"""PDF指定页的有界文字读取。

复用资料目录边界检查，只提取指定物理页的文字，不执行资料内容。
不进行OCR，不保证图片、表格布局或公式被完整保留。
本模块不调用LLM，也不修改学习状态。

页码从1开始，指PDF阅读器显示的物理页序号，
不一定等于书中印刷页码。
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from .materials import safe_material_path


logger = logging.getLogger("ai_agent.pdf_materials")

# 先保守限制输出范围；它们不是解析耗时或内存的硬上限。
MAX_READ_PDF_PAGES = 3
MAX_READ_PDF_TEXT_CHARS = 8_000


def read_pdf_pages(
    root_dir: str,
    relative_path: str,
    start_page: int,
    end_page: int,
) -> dict[str, object]:
    """读取指定物理页的可提取文字，包含起止两端。"""
    # bool也是int的子类，因此用type检查拒绝True等非页码输入。
    if type(start_page) is not int or type(end_page) is not int:
        raise ValueError("起止页必须是整数")

    if start_page < 1 or end_page < start_page:
        raise ValueError("页码必须满足1 <= start_page <= end_page")

    if end_page - start_page + 1 > MAX_READ_PDF_PAGES:
        raise ValueError(
            f"单次最多读取{MAX_READ_PDF_PAGES}页"
        )

    if Path(relative_path).is_absolute():
        raise ValueError("请使用资料目录内的相对路径")

    path = safe_material_path(root_dir, relative_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError("按页读取仅支持PDF")

    if not path.is_file():
        raise FileNotFoundError("指定PDF不存在")

    source = path.relative_to(
        Path(root_dir).expanduser().resolve()
    ).as_posix()

    rows: list[dict[str, object]] = []
    used_chars = 0

    # 文件流在整个页面提取期间保持打开，结束后自动关闭。
    with path.open("rb") as stream:
        try:
            reader = PdfReader(stream, strict=False)
        except Exception as exc:
            raise ValueError("无法解析PDF文件") from exc

        # 本轮先明确拒绝加密PDF，不让模型处理或猜测密码。
        if reader.is_encrypted:
            raise ValueError("暂不支持加密PDF，请先解除密码保护")

        total_pages = len(reader.pages)
        if start_page > total_pages:
            raise ValueError("起始页超出PDF范围")

        effective_end = min(end_page, total_pages)

        for page_no in range(start_page, effective_end + 1):
            remaining = MAX_READ_PDF_TEXT_CHARS - used_chars
            if remaining <= 0:
                break

            try:
                # 用户页码从1开始，pypdf页面索引从0开始。
                text = reader.pages[page_no - 1].extract_text() or ""
            except Exception as exc:
                # 不静默跳过失败页，否则可能误称整段范围已读取。
                raise ValueError(
                    f"PDF第{page_no}页文字提取失败"
                ) from exc

            partial = len(text) > remaining
            returned_text = text[:remaining]

            rows.append({
                "page": page_no,
                "text": returned_text,
                "text_status": (
                    "extracted" if text.strip() else "no_text"
                ),
                "partial": partial,
                "citation": f"[source={source}, page={page_no}]",
            })
            used_chars += len(returned_text)

            if partial:
                break

    # 预算为正且起始页合法时，至少会产生一条页面记录。
    returned_end = rows[-1]["page"]

    # 超出文档末尾和输出预算截断是两种不同情况。
    # 例如请求2~4页、PDF只有3页，不属于字符预算截断。
    truncated = (
        bool(rows[-1]["partial"])
        or returned_end < effective_end
    )

    logger.debug(
        "PDF按页读取：requested=%d-%d returned=%d-%d "
        "chars=%d truncated=%s no_text_pages=%d",
        start_page,
        end_page,
        rows[0]["page"],
        returned_end,
        used_chars,
        truncated,
        sum(row["text_status"] == "no_text" for row in rows),
    )

    # 动态评估：观察截断比例、无文字页比例及真实页码核对结果。
    # 经常截断时先缩小范围；无文字不能直接判断为扫描版或空白页。
    return {
        "source": source,
        "page_numbering": "physical_1_based",
        "total_pages": total_pages,
        "requested_start_page": start_page,
        "requested_end_page": end_page,
        "returned_start_page": rows[0]["page"],
        "returned_end_page": returned_end,
        "end_clipped_to_document": end_page > total_pages,
        "truncated": truncated,
        "pages": rows,
    }
