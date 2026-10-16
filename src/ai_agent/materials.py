"""学习资料路径、类型和目录遍历的公共功能。"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator

SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".py", ".pdf"})


def resolve_material_root(root_dir: str) -> Path:
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"资料目录不存在：{root}")
    return root


def iter_material_files(root: Path) -> Iterator[Path]:
    """递归、稳定地返回全部受支持资料文件。"""
    yield from sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def safe_material_path(root_dir: str, relative_path: str) -> Path:
    """阻止 ../ 越界，并限制可读取的文件类型。"""
    root = resolve_material_root(root_dir)
    candidate = (root / relative_path).resolve()

    if candidate != root and root not in candidate.parents:
        raise ValueError("不允许读取资料目录以外的文件。")

    if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的资料类型：{candidate.suffix}")

    return candidate

def read_material_excerpt(
    root_dir: str,
    relative_path: str,
    max_chars: int,
) -> dict:
    """读取文本开头，返回真实文件引用及范围信息，不制造向量 chunk 编号。"""
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars 必须是正整数")
    if Path(relative_path).is_absolute():
        raise ValueError("请使用资料目录内的相对路径")

    path = safe_material_path(root_dir, relative_path)
    if path.suffix.lower() not in {".txt", ".md", ".py"}:
        raise ValueError("直接摘录只支持 TXT/MD/PY")

    # 多读一个字符即可判断截断，无需为了少量摘录读取整个大文件。
    # 严格解码：编码损坏时报告错误，不把替换后的文字冒充准确原文。
    with path.open("r", encoding="utf-8-sig") as stream:
        raw = stream.read(max_chars + 1)

    text = raw[:max_chars]
    source = path.relative_to(
        Path(root_dir).expanduser().resolve()
    ).as_posix()

    # 文件级引用表示来源文件，不表示已读取全文。
    # 动态评估：截断时只能解释实际返回内容；空内容不生成正文引用。
    return {
        "source": source,
        "citation": f"[source={source}]" if text.strip() else None,
        "scope": "prefix",
        "returned_chars": len(text),
        "truncated": len(raw) > max_chars,
        "text": text,
    }

# 先使用明确、容易验证的输出边界；不是模型上下文窗口的总预算。
MAX_READ_LINES = 200
MAX_READ_TEXT_CHARS = 8_000


def read_material_lines(
    root_dir: str,
    relative_path: str,
    start_line: int,
    end_line: int,
) -> dict[str, object]:
    """读取 TXT/MD/PY 的指定行范围，行号从 1 开始且包含两端。

    不执行文件，不修改内容，不调用模型。
    返回实际行号与截断信息，避免把部分内容误当成完整证据。

    本函数限制返回正文的规模，不保证超大文件的扫描耗时。
    """
    # bool 在 Python 中是 int 的子类，必须显式拒绝。
    # 不把 1.5、"10" 等输入静默转换为行号。
    if type(start_line) is not int or type(end_line) is not int:
        raise ValueError("起止行必须是整数")

    if start_line < 1 or end_line < start_line:
        raise ValueError("行范围必须满足 1 <= start_line <= end_line")

    if end_line - start_line + 1 > MAX_READ_LINES:
        raise ValueError(f"单次最多读取 {MAX_READ_LINES} 行")

    if Path(relative_path).is_absolute():
        raise ValueError("请使用资料目录内的相对路径")

    # 复用已有路径校验，包括 ../ 和解析后的目录边界检查。
    path = safe_material_path(root_dir, relative_path)

    if path.suffix.lower() not in {".txt", ".md", ".py"}:
        raise ValueError("按行读取仅支持 TXT/MD/PY；PDF 需要专门的按页读取")

    if not path.is_file():
        raise FileNotFoundError("指定资料文件不存在")

    rows: list[dict[str, object]] = []
    used_chars = 0
    truncated = False

    # utf-8-sig 兼容普通 UTF-8 和带 BOM 的文件。
    # 不使用 errors="replace"，避免把损坏编码伪装成准确原文。
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line_number < start_line:
                continue
            if line_number > end_line:
                break

            remaining = MAX_READ_TEXT_CHARS - used_chars
            if remaining <= 0:
                truncated = True
                break

            text = line.rstrip("\r\n")
            partial = len(text) > remaining

            rows.append({
                "line": line_number,
                "text": text[:remaining],
                "partial": partial,
            })
            used_chars += len(text[:remaining])

            if partial:
                truncated = True
                break

    if not rows:
        raise ValueError("起始行超出文件范围，没有返回正文")

    source = path.relative_to(
        Path(root_dir).expanduser().resolve()
    ).as_posix()

    # requested 与 returned 分开：
    # 请求到第100行，但文件只有70行时，不能声称读取到了第100行。
    # 动态评估：检查定位准确性、截断比例和越界拒绝；
    # 若经常截断，应缩小读取范围，而不是直接无限扩大预算。
    return {
        "source": source,
        "requested_start_line": start_line,
        "requested_end_line": end_line,
        "returned_start_line": rows[0]["line"],
        "returned_end_line": rows[-1]["line"],
        "truncated": truncated,
        "lines": rows,
    }