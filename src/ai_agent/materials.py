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