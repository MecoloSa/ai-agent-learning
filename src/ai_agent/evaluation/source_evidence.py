"""
从原始资料提取评审证据，不依赖生成者的摘要或向量相似度。
    1.页级邻页策略：cited 模式下取引用页 ±1 页，避免 chunk 粒度带来的索引漂移问题
    2.宁可报错也不截断：超预算抛异常而非静默截断，防止"证据完整"的虚假宣称
    3.去重引用：list(dict.fromkeys(...)) 保证同一引用只处理一次
    4.安全边界：软链接过滤防止读取越权文件
    5.SHA-256 哈希：记录每个源文件的哈希，支持后续校验资料是否被篡改
    6.不做 OCR：明确声明能力边界，图像内容不处理
    7.citation_checks 记录每条引用的定位状态；定位成功不代表语义支持
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


CITATION = re.compile(
    r"\[source=([^,\]\n]+)(?:,\s*page=(\d+))?(?:,\s*chunk=(\d+))?\]"
)
SUPPORTED = {".pdf", ".md", ".txt", ".py"}


def load_source_evidence(
    materials: Path, plan: str, *, mode: str = "cited",
    max_chars: int = 150_000,
) -> dict:
    """cited读取引用页及邻页；full读取指定文件/目录全部支持的文本。

    页码从PDF第一页计为1。这里只做页级核对，不重建历史chunk索引。
    超预算直接报错，避免截断后错误宣称证据完整；图像内容不做OCR。
    """
    materials = materials.resolve(strict=True)
    if mode not in {"cited", "full"} or max_chars <= 0:
        raise ValueError("证据模式或字符预算不合法")
    root = materials if materials.is_dir() else materials.parent
    files = sorted(
        (p for p in materials.rglob("*") if p.is_file()
         and p.suffix.lower() in SUPPORTED),
        key=lambda p: p.as_posix(),
    ) if materials.is_dir() else [materials]
    # 不允许软链接等将读取范围带到用户指定目录之外。
    files = [p for p in files if p.resolve().is_relative_to(root)]
    refs = list(dict.fromkeys(CITATION.findall(plan)))
    selected: dict[Path, set[int | None]] = {}
    issues: list[str] = []
    citation_checks: list[dict] = []
    # 内部保留引用与实际文件的对应关系，输出只包含相对路径，便于复核。
    targets: list[tuple[dict, Path]] = []
    for marker in dict.fromkeys(re.findall(r"\[source=[^\]\n]+\]", plan)):
        match = CITATION.fullmatch(marker)
        check = {"citation": marker, "status": "incomplete", "reason": "尚未读取原文"}
        citation_checks.append(check)
        if match is None:
            issues.append(f"引用格式无法定位：{marker}")
            check.update(status="invalid", reason="引用格式无法定位")
            continue
        source, page, _chunk = match.groups()
        matches = [p for p in files if source.strip() in {
            p.name, p.relative_to(root).as_posix(),
        }]
        if len(matches) != 1:
            issues.append(f"引用文件缺失或名称有歧义：{source}")
            # 歧义尚不能证明文件不存在；零匹配则可确认本次资料范围内没有该文件。
            check.update(status="invalid" if not matches else "incomplete",
                         reason="引用文件缺失" if not matches else "引用文件名有歧义")
            continue
        path = matches[0]
        check.update(source=path.relative_to(root).as_posix(),
                     page=int(page) if page else None)
        if path.suffix.lower() == ".pdf" and not page:
            issues.append(f"PDF引用缺少可定位的页码：{source}")
            check.update(status="invalid", reason="PDF引用缺少页码")
        else:
            targets.append((check, path))
        selected.setdefault(path, set()).add(int(page) if page else None)

    if not refs:
        issues.append("没有可解析的资料引用，不能核验引用支持度")
    if mode == "full":
        for path in files:
            selected.setdefault(path, set())

    pack = {
        "scope": mode,
        "requested_citations": [
            {"source": s, "page": int(p) if p else None,
             "chunk": int(c) if c else None} for s, p, c in refs
        ],
        "chunk_validation": "not_verified",
        "citation_checks": citation_checks,
        "files": [], "documents": [], "issues": issues,
    }

    def add_text(source: str, page: int | None, text: str, role: str) -> None:
        if not text.strip():
            issues.append(f"无法提取文本（可能需要OCR）：{source}, page={page}")
            return
        pack["documents"].append({"source": source, "page": page, "text": text,
                                  "evidence_role": role})
        # 动态评估：观察实际证据量；不在日志打印资料全文。
        if len(json.dumps(pack, ensure_ascii=False)) > max_chars:
            raise ValueError("原文证据超过字符预算，请拆分评审；未静默截断")

    for path, pages in selected.items():
        source = path.relative_to(root).as_posix()
        try:
            raw = path.read_bytes()
            pack["files"].append({
                "source": source, "sha256": hashlib.sha256(raw).hexdigest(),
            })
            if path.suffix.lower() == ".pdf":
                reader = PdfReader(path)
                valid = {p for p in pages if p is not None and 1 <= p <= len(reader.pages)}
                for p in pages:
                    if p is not None and p not in valid:
                        issues.append(f"PDF页码越界：{source}, page={p}")
                        for check, target in targets:
                            if target == path and check["page"] == p:
                                check.update(status="invalid", reason="PDF页码越界")
                wanted = set(range(1, len(reader.pages) + 1)) if mode == "full" else {
                    n for p in valid for n in (p - 1, p, p + 1)
                    if 1 <= n <= len(reader.pages)
                }
                for number in sorted(wanted):
                    role = "cited" if number in valid else (
                        "full" if mode == "full" else "neighbor"
                    )
                    add_text(source, number, reader.pages[number - 1].extract_text() or "", role)
            elif path.suffix.lower() in SUPPORTED:
                add_text(source, None, raw.decode("utf-8-sig"), "cited" if pages else "full")
            else:
                issues.append(f"不支持的资料格式：{source}")
        except ValueError:
            raise
        except Exception as exc:
            issues.append(f"资料读取失败：{source} ({type(exc).__name__})")

    loaded = {(d["source"], d["page"]) for d in pack["documents"]}
    for check, _ in targets:
        if check["status"] != "invalid":
            found = (check["source"], check["page"]) in loaded
            check.update(status="located" if found else "incomplete",
                        reason="引用原文已读取，语义尚待评审" if found else "引用原文未成功提取")
    # 动态评估：invalid 是确定定位错误；incomplete 是未判定，不能当作语义错误。
    pack["coverage_complete"] = bool(refs and pack["documents"]) and not issues
    if len(json.dumps(pack, ensure_ascii=False)) > max_chars:
        raise ValueError("原文证据超过字符预算，请拆分评审；未静默截断")
    return pack
