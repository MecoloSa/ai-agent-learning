"""学习资料 RAG：规模判断、加载、切分、增量索引和检索"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import create_embeddings, get_settings
from .materials import iter_material_files

logger = logging.getLogger("ai_agent.rag")

class RAGError(RuntimeError):
    """PDF解析、索引构建或检索失败。"""

def needs_rag(root_dir: str) -> bool:
    """路由判断：单文件、总语料规模或 PDF 任意一條件滿足即走 RAG"""
    settings = get_settings()
    root = Path(root_dir).resolve()

    all_files = (
        list(iter_material_files(root))
        if root.is_dir()
        else []
    )

    # Python始终完整读取，不参与RAG路由判断。
    # 即使Python文件很大，也不能单独触发向量流水线。
    rag_files = [
        path
        for path in all_files
        if path.suffix.lower() != ".py"
    ]

    total = sum(
        path.stat().st_size
        for path in rag_files
    )

    return (
            any(
                path.stat().st_size
                >= settings.rag_large_file_bytes
                for path in rag_files
            )
            or total >= settings.rag_large_corpus_bytes
            or any(
        path.suffix.lower() == ".pdf"
        for path in rag_files
    )
    )

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()       # 创建一个 SHA-256 计算器
    with path.open("rb") as file:   # 用二进制读取，因为哈希针对原始字节计算
        # iter 将可迭代对象转换为迭代器，从而可以逐个访问其中的元素
        # 每次调用 lambda 函数，就会从文件中读取 1MB 的数据块
        # 如果读到文件末尾（EOF/b""），则返回空字符串 b""
        for block in iter(lambda: file.read(1024*1024), b""):
            digest.update(block)
    # 返回 16 进制摘要
    return digest.hexdigest()

def _load_file(root: Path, path: Path) -> list[Document]:
    """保留 source/page 元数据，使最終回答能夠給出证据位置"""
    # as_posix() 在任何操作系统上用 POSIX 风格的字符串表示路径对象
    # 即用 ‘/’ 作为路径分隔符
    source = path.relative_to(root).as_posix()
    # pdf 文件读取
    if path.suffix.lower() == ".pdf":
        try:
            # strict=False 对部分格式不完全规范的 PDF 更宽容。
            reader = PdfReader(str(path), strict=False)
        except Exception as exc:
            raise RAGError(f"无法打开 PDF：{source}；原因：{exc}") from exc
        # pdf 加密情况
        if reader.is_encrypted:
            try:
                decrypt_result = reader.decrypt("")
            except Exception as exc:
                raise RAGError(f"PDF 已加密且无法解密：{source}") from exc

            if not decrypt_result:
                raise RAGError(
                    f"PDF 需要密码：{source}。请先解除密码保护后再导入。"
                )

        documents: list[Document] = []
        failed_pages: list[int] = []

        # enumerate 遍历可迭代对象时同时获取元素的索引和值，这里定义从 1 开始
        for page_no, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning(
                    "PDF 页面解析失败：source=%s page=%d error=%s",
                    source,
                    page_no,
                    str(exc)[:200],
                )
                logger.debug(
                    "PDF 页面解析异常详情",
                    exc_info=True,
                )
                failed_pages.append(page_no)
                continue

            if text.strip():
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": source,
                            "page": page_no,
                            "file_type": "pdf",
                        },
                    )
                )

        if not documents:
            raise RAGError(
                f"PDF 未提取到任何文本：{source}。"
                "它可能是扫描版/图片版 PDF，需要先执行 OCR；"
                "也可能是加密、损坏或字体编码异常。"
            )

        if failed_pages:
            logger.warning(
                "PDF %s 有部分页面解析失败：%s",
                source,
                failed_pages,
            )

        return documents

    # 非 pdf 文件读取，遇到非法字符时候，用替代字符代替（不报错）
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": source})]

def _allocate_overview_budgets(
    readable_sizes: dict[Path, int],
    total_budget: int,
    *,
    preferred_base_chars: int = 600,
) -> dict[Path, int]:
    """为 overview 按“基础额度 + 平方根权重”分配字符预算。

    设计目标：
    1. 小文件至少获得基础展示机会；
    2. 大文件获得更多额度；
    3. 大文件额度不与文件大小线性增长；
    4. 单个文件不能长期垄断整个 overview。
    """
    paths = list(readable_sizes)

    if not paths or total_budget <= 0:
        return {path: 0 for path in paths}

    file_count = len(paths)

    # 最多使用约一半总预算作为基础额度，
    # 另一半以上留给按文件文本量加权分配。
    #
    # 文件很多时，base 会自动下降。
    # 这是固定总预算下不可避免的退化，而不是程序错误。
    base_chars = min(
        preferred_base_chars,
        total_budget // max(1, file_count * 2),
    )

    budgets = {
        path: base_chars
        for path in paths
    }

    remaining = max(
        0,
        total_budget - base_chars * file_count,
    )

    # 当文件数>=3时，单文件最多约占总预算1/3；
    # 一个或两个文件时允许它们充分使用总预算。
    per_file_cap = max(
        base_chars,
        total_budget // min(file_count, 3),
    )

    # 使用可提取文本量的平方根，而不是磁盘字节数。
    # PDF图片、字体等会占磁盘空间，但不一定产生可读文本。
    weights = {
        path: math.sqrt(
            max(1, readable_sizes[path])
        )
        for path in paths
    }

    # 若部分文件达到上限，将剩余预算继续分配给其他文件。
    while remaining > 0:
        eligible = [
            path
            for path in paths
            if budgets[path] < per_file_cap
        ]

        if not eligible:
            break

        weight_sum = sum(weights[path] for path in eligible)
        remaining_at_start = remaining
        allocated_this_round = 0

        for path in eligible:
            proportional_share = max(
                1,
                int(
                    remaining_at_start
                    * weights[path]
                    / weight_sum
                ),
            )

            capacity = per_file_cap - budgets[path]
            addition = min(
                proportional_share,
                capacity,
                remaining,
            )

            budgets[path] += addition
            remaining -= addition
            allocated_this_round += addition

            if remaining <= 0:
                break

        if allocated_this_round == 0:
            break

    return budgets

class LearningRAG:
    def __init__(self, root_dir: str, data_dir: str | None = None) -> None:
        settings = get_settings()
        self.root = Path(root_dir).resolve()
        # 不把索引写进用户资料目录；不同资料根目录使用不同集合.
        # 目录编码哈希值，到第 12 个字符 [start:end:step]
        # 同一批文件移动到另一目录，也会有一个新的 corpus_id
        corpus_id = hashlib.sha256(str(self.root).encode()).hexdigest()[:12]
        # 在启动目录下的 rag 文件夹里以文件路径哈希为名新建文件夹
        configured_data_dir = data_dir or settings.rag_data_dir
        self.persist_dir = Path(configured_data_dir).resolve() / corpus_id
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # 再在下面新建 manifest 文件（保存 相对文件路径 和 当前文件内容哈希）
        self.manifest_path = self.persist_dir / "manifest.json"
        # 向量库持久化到磁盘 sha/chroma
        # 只有文档和问题使用相同的向量模型，它们才能在同一向量空间中比较相似度
        self.store = Chroma(
            collection_name=f"learning_{corpus_id}",            # 类似数据库表名
            embedding_function=create_embeddings(),             # 返回 Embedding 模型对象
            persist_directory=str(self.persist_dir / "chroma"),
            # 明确使用余弦距离，便于解释相关性分数。
            collection_metadata={
                "hnsw:space": "cosine",
            },
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            # 中文标点放在空字符串之前，降低词语被切开的概率
            # 优先级：从前到后依次切分 —— 从大到小
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""],
            # 向元数据中加入 chunk 在原文中的起始字符位置
            add_start_index=True,
        )

    def _read_manifest(self) -> dict[str, str]:
        if not self.manifest_path.exists():
            return {}

        try:
            data = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            logger.warning("RAG manifest 损坏，将重新建立索引。")
            return {}

        return data if isinstance(data, dict) else {}

    def build_overview_text(
        self,
        *,
        max_pdf_page: int = 10,
        max_chars: int = 14_000,
    ) -> str:
        """确定性读取资料开头，用于识别目录、前言和阅读指引。

        这一步不做向量相似度检索，避免在尚不了解资料结构时，
        用一个宽泛问题从整本书中任意挑选若干 chunk。
        """
        # blocks: 存放每个文件（或 PDF 页面）的文本片段
        blocks: list[str] = []
        total_chars = 0

        # Python源码会在 analyse_materials() 中完整读取，
        # overview 只负责PDF、Markdown和TXT。
        paths = [
            path
            for path in iter_material_files(self.root)
            if path.suffix.lower() != ".py"
        ]

        if not paths:
            return "没有需要生成overview的非Python资料"

        # 先读取各文件，并记录真正可以交给模型的文本量。
        #
        # 不能直接使用 path.stat().st_size：
        # 例如一个3MB扫描PDF可能几乎没有可提取文本，
        # 而一个100KB源码文件可能全部都是有效文本。
        loaded_files: dict[Path, list[Document]] = {}
        readable_sizes: dict[Path, int] = {}

        for path in paths:
            documents = _load_file(self.root, path)

            selected_documents = (
                documents[:max_pdf_page]
                if path.suffix.lower() == ".pdf"
                else documents
            )

            if not selected_documents:
                continue

            loaded_files[path] = selected_documents
            readable_sizes[path] = sum(
                len(document.page_content)
                for document in selected_documents
            )

        file_budgets = _allocate_overview_budgets(
            readable_sizes=readable_sizes,
            total_budget=max_chars,
        )

        logger.info(
            "Overview预算分配：total=%d files=%d budgets=%s",
            max_chars,
            len(file_budgets),
            {
                path.relative_to(self.root).as_posix(): budget
                for path, budget in file_budgets.items()
            },
        )

        for path, selected_documents in loaded_files.items():
            file_budget = file_budgets.get(path, 0)

            if file_budget <= 0:
                continue

            # 文件内部按Document平均。
            # PDF的Document对应页面，普通文本文件通常只有一个Document。
            per_document_budget = max(
                1,
                file_budget // len(selected_documents),
            )

            for document in selected_documents:
                page = document.metadata.get("page")
                source = document.metadata["source"]

                header = (
                    f"[source={source}, page={page}]"
                    if page is not None
                    else f"[source={source}]"
                )

                body = document.page_content.strip()
                body_budget = max(
                    0,
                    per_document_budget - len(header) - 1,
                )

                if body_budget <= 0:
                    continue

                if len(body) > body_budget:
                    truncation_mark = "\n[该文件的overview内容已截断]"
                    usable_chars = max(
                        0,
                        body_budget - len(truncation_mark),
                    )
                    body = (
                            body[:usable_chars]
                            + truncation_mark
                    )

                block = f"{header}\n{body}"

                if total_chars + len(block) > max_chars:
                    break

                blocks.append(block)
                total_chars += len(block)

        return (
            "\n\n---\n\n".join(blocks)
            if blocks
            else "没有可用于生成资料概览的文本"
        )

    def sync(self) -> dict[str, int] | None:
        """仅重建新增/变化/删除文件，避免每次运行重复 embedding。"""
        old = self._read_manifest()
        # Python源码始终完整读取，不做embedding，也不写入Chroma。
        current = {
            path.relative_to(self.root).as_posix(): _sha256(path)
            for path in iter_material_files(self.root)
            if path.suffix.lower() != ".py"
        }
        changed = {name for name, digest in current.items() if old.get(name) != digest}
        deleted = set(old) - set(current)

        # | 表示并集
        for source in sorted(changed | deleted):
            # 先删除这个 source 的旧 chunk；文件删除时也能清理残留索引。
            self.store.delete(where={"source": source})

        # 加载并切分变化文件
        added_chunks = 0
        for source in sorted(changed):
            docs = _load_file(self.root, self.root / source)
            chunks = self.splitter.split_documents(docs)
            file_hash = current[source]
            ids: list[str] = []
            for index, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "file_hash": file_hash,
                    "chunk_index": index,
                })
                ids.append(hashlib.sha256(
                    f"{source}:{file_hash}:{index}".encode()
                ).hexdigest())
            if chunks:
                # add_document 完成：
                # Document.page_content -> embedding_function
                # -> 浮点向量 -> 正文 + 向量 + metadata + id 写入 Chroma
                self.store.add_documents(chunks, ids=ids)
                added_chunks += len(chunks)

        # 先成功更新向量库，在提交 manifest，避免失败后错误地认为已同步
        # 同时用新建 tmp 的方式，这样程序中途退出时，不容易留下只写了一半的 JSON。
        manifest_tmp = self.manifest_path.with_suffix(".json.tmp")
        manifest_tmp.write_text(
            json.dumps(current, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_tmp.replace(self.manifest_path)
        return {"changed_files": len(changed), "deleted_files": len(deleted),
            "added_chunks": added_chunks,}

    def retrieve(
            self,
            query: str,
            k: int = 6,
            *,
            min_relevance: float = 0.35,
            relative_margin: float = 0.12,
            min_hits: int = 3,
            exclude_suffixes: set[str] | None = None,
    ) -> list[Document]:
        """"执行带阈值的相似度检索。

        流程：
        1. 先调用 sync() 确保向量库与文件系统同步（增量更新）。
        2. 用多倍于 k 的 fetch_k 获取候选，然后通过双重阈值过滤：
           - 绝对阈值 min_relevance：分数低于此值的直接丢弃。
           - 相对阈值 relative_margin：与最佳结果的差距超过此值的丢弃。
        3. 对同一文件的同一页去重，避免返回高度重复的 chunk。
        4. 最多返回 k 个结果。

        参数:
            query:           用户提问文本，会被同一个 Embedding 模型编码为向量。
            k:               最终期望返回的命中数量。
            min_relevance:   绝对相关性下限（余弦相似度，范围通常 0~1）。
                             低于该分数的候选结果直接排除。
            relative_margin: 相对边距。动态阈值 = max(min_relevance, 最佳分数 - relative_margin)。
                             例如最佳结果为 0.72，margin=0.12，则低于 0.60 的被排除。
                             这样当最佳结果本身都不高时（比如 0.40），不会因相对阈值
                             把勉强可用的结果也全部丢掉。

        返回:
            过滤后的 Document 列表，每个 Document 的 metadata 会新增：
            - relevance_score: float 类型的相似度分数（余弦距离，越高越相关）。
        """
        self.sync()
        logger.info(
            "RAG 查询：query=%r k=%d "
            "min_relevance=%.4f relative_margin=%.4f",
            " ".join(query.split())[:300],
            k,
            min_relevance,
            relative_margin,
        )

        normalized_excluded = {
            suffix.lower()
            for suffix in (exclude_suffixes or set())
        }

        # 如果后面要排除.py，需要先多取一些候选。
        # 否则原始前32名大量为.py时，过滤后可能剩不下文档证据。
        fetch_k = (
            max(k * 8, 60)
            if normalized_excluded
            else max(k * 4, 20)
        )

        # MMR（最大边际相关性） 在相关性之外一直高度重复的 chunk。
        # 简化理解：MMR分数=λ×查询相关性−(1−λ)×结果间重复度
        # fetch_k：从向量库中拉取的候选数量，取值 max(k * 4, 20)。
        # 先多拉一些候选，再通过阈值过滤，比直接拉 k 个更有可能保留高质量结果。
        # 注释中提到 MMR 已不适用——旧版 Chroma 用 max_marginal_relevance_search，
        # 新版改用 similarity_search_with_relevance_scores，直接返回余弦相似度分数。
        # similarity_search_with_relevance_scores：
        # 1. 将 query 文本通过 embedding_function 编码为向量
        # 2. 在向量空间中用余弦距离找到最相似的 fetch_k 个 chunk
        # 3. 返回列表，每个元素为 (Document, score)，score 为余弦相似度
        raw_candidates = self.store.similarity_search_with_relevance_scores(
            query=query,
            k=fetch_k,
        )

        candidates = [
            (document, score)
            for document, score in raw_candidates
            if Path(
                str(document.metadata.get("source", ""))
            ).suffix.lower() not in normalized_excluded
        ]

        logger.info(
            "候选文件类型过滤：raw=%d remaining=%d exclude_suffixes=%s",
            len(raw_candidates),
            len(candidates),
            sorted(normalized_excluded),
        )

        # 阈值计算必须发生在过滤之后，这样 .py 的高分结果不会影响 PDF/MD/TXT 的动态阈值。
        if not candidates:
            logger.info("检索没有返回候选结果。")
            return []

        best_score = max(score for _, score in candidates)
        relative_threshold = best_score - relative_margin
        # 动态阈值：这里使用的是固定阈值差
        # 先按最终的“位置去重规则”，找出通过绝对阈值的候选。
        # 避免同一 PDF 页的多个 chunk 虚假满足 min_hits。
        qualified_unique_scores: list[float] = []
        floor_seen_locations: set[tuple[str, str, int]] = set()

        for document, score in candidates:
            if score < min_relevance:
                continue

            source = str(document.metadata.get("source"))
            page = document.metadata.get("page")
            chunk_index = int(document.metadata.get("chunk_index") or 0)

            location = (
                (source, "page", int(page))
                if page is not None
                else (source, "chunk", chunk_index)
            )

            if location in floor_seen_locations:
                continue

            floor_seen_locations.add(location)
            qualified_unique_scores.append(float(score))

        # coverage_floor 是第 min_hits 个合格候选的分数。
        # 相对阈值不得高于它，否则一次查询可能只留下一个离群高分结果。
        if qualified_unique_scores:
            floor_index = min(
                min_hits,
                len(qualified_unique_scores),
            ) - 1
            coverage_floor = qualified_unique_scores[floor_index]

            # 相对阈值不能高于第 min_hits 个合格候选的分数。
            # 例如第一名0.545、第三名0.415时，
            # 最终相对阈值会从0.425放宽到0.415。
            softened_relative_threshold = min(
                relative_threshold,
                coverage_floor,
            )
        else:
            coverage_floor = min_relevance
            softened_relative_threshold = relative_threshold

        dynamic_threshold = max(
            min_relevance,
            softened_relative_threshold,
        )
        logger.info(
            "检索阈值计算：best_score=%.4f "
            "raw_relative=%.4f coverage_floor=%.4f "
            "final_threshold=%.4f min_hits=%d active_rule=%s",
            best_score,
            relative_threshold,
            coverage_floor,
            dynamic_threshold,
            min_hits,
            (
                "absolute"
                if min_relevance >= softened_relative_threshold
                else "softened"
            ),
        )

        selected: list[Document] = []

        # 只保留得分最高的少量阈值淘汰项，用于判断阈值是否过严。
        # 不记录重复位置和 k 数量限制造成的淘汰。
        borderline_rejected: list[tuple] = []

        # 保存已经选中过的位置，避免同一 PDF 页面返回多个高度重复的 chunk。
        #
        # key 的格式为：
        #   PDF：     (source, "page", page)
        #   普通文件： (source, "chunk", chunk_index)
        #
        # 不能继续直接使用 (source, page)，因为 txt/md 文件没有 page，
        # 它们的 page 都是 None，会导致同一文件只能保留第一个 chunk。
        seen_locations: set[tuple[str, str, int]] = set()

        # 统计每种处理结果的数量，便于最后快速观察阈值效果。
        decision_counts = {
            "selected": 0,
            "reject_absolute": 0,
            "reject_relative": 0,
            "reject_duplicate": 0,
            "reject_k_limit": 0,
        }

        for rank, (document, score) in enumerate(
                candidates,
                start=1,
        ):
            source = str(document.metadata.get("source"))
            page = document.metadata.get("page")
            chunk_index = document.metadata.get("chunk_index")

            # 分别判断是否通过绝对阈值与相对阈值。
            # 虽然最终只需要比较 dynamic_threshold，
            # 但分开判断才能在日志中看出具体被哪条规则淘汰。
            passes_absolute = score >= min_relevance
            passes_relative = score >= dynamic_threshold

            # PDF 按页面去重；普通文本没有 page，改为按 chunk 去重。
            if page is not None:
                location_key = (
                    source,
                    "page",
                    int(page),
                )
            else:
                location_key = (
                    source,
                    "chunk",
                    int(chunk_index),
                )

            is_duplicate = location_key in seen_locations

            # 按顺序确定候选结果的最终处理原因。
            if not passes_absolute:
                # 分数低于固定下限，例如 score < 0.35。
                decision = "reject_absolute"

            elif not passes_relative:
                # 虽然超过固定下限，但与最佳结果差距过大。
                # 例如 best_score=0.72、relative_margin=0.12，
                # 则低于 0.60 的结果在这里被淘汰。
                decision = "reject_relative"

            elif is_duplicate:
                # 已经选择过同一 PDF 页面，避免重复内容占用名额。
                decision = "reject_duplicate"

            elif len(selected) >= k:
                # 已经选满 k 个结果。
                decision = "reject_k_limit"

            else:
                decision = "selected"

                # 把分数和原始排名写入 metadata，
                # 后续输出上下文或排查问题时可以继续使用。
                document.metadata["relevance_score"] = float(score)
                document.metadata["retrieval_rank"] = rank

                selected.append(document)
                seen_locations.add(location_key)

            decision_counts[decision] += 1
            # selected 候选正常打印；因重复页面被淘汰的不打印；
            # 因 k 已满被淘汰的不打印；因阈值被淘汰的候选先暂存，稍后只展示最高的 3 条。
            if decision == "selected":
                preview = " ".join(
                    document.page_content[:120].split()
                )

                logger.info(
                    "RAG 命中：rank=%d score=%.4f "
                    "source=%s page=%s chunk=%s preview=%r",
                    rank,
                    score,
                    source,
                    page,
                    chunk_index,
                    preview,
                )

            elif decision in {
                "reject_absolute",
                "reject_relative",
            }:
                borderline_rejected.append(
                    (
                        rank,
                        float(score),
                        decision,
                        source,
                        page,
                        chunk_index,
                    )
                )

        for (
                rejected_rank,
                rejected_score,
                rejected_reason,
                rejected_source,
                rejected_page,
                rejected_chunk,
        ) in borderline_rejected[:3]:
            logger.info(
                "RAG 临界淘汰：rank=%d score=%.4f "
                "reason=%s source=%s page=%s chunk=%s",
                rejected_rank,
                rejected_score,
                rejected_reason,
                rejected_source,
                rejected_page,
                rejected_chunk,
            )
        logger.info(
            "检索过滤完成：候选=%d, 保留=%d, "
            "best_score=%.4f, relative_threshold=%.4f, "
            "dynamic_threshold=%.4f, "
            "绝对阈值淘汰=%d, 相对阈值淘汰=%d, "
            "重复位置淘汰=%d, k限制淘汰=%d",
            len(candidates),
            len(selected),
            best_score,
            relative_threshold,
            dynamic_threshold,
            decision_counts["reject_absolute"],
            decision_counts["reject_relative"],
            decision_counts["reject_duplicate"],
            decision_counts["reject_k_limit"],
        )

        return selected

    def retrieve_many(
            self,
            queries: tuple[tuple[str, str], ...],
            *,
            k_per_query: int,
            max_hits: int,
            min_relevance: float,
            relative_margin: float,
            min_hits: int,
            exclude_suffixes: set[str] | None = None,
    ) -> list[Document]:
        """分别召回四类 hit，再以轮询方式合并和去重。

        轮询不是按全局分数直接排序，而是依次从每种 query 中取一个结果。
        这样可以避免 organization 的多个高分结果占满全部名额，
        让 implementation 和 usage_validation 至少获得基本覆盖。

        动态评估：
        - merged_hits 明显小于各查询 hit 总和，说明跨查询重复较多；
        - 某类 query 长期贡献 0 个，需要检查 query 表达或阈值；
        - 如果低相关类别挤占过多名额，再考虑加入 reranker，而不是先扩大 k。
        """
        grouped_hits: list[tuple[str, list[Document]]] = []

        for query_id, query in queries:
            hits = self.retrieve(
                query=query,
                k=k_per_query,
                min_relevance=min_relevance,
                relative_margin=relative_margin,
                min_hits=min_hits,
                exclude_suffixes=exclude_suffixes,
            )

            # matched_queries 只用于本次内存中的上下文解释，不写回 Chroma。
            for hit in hits:
                hit.metadata["matched_queries"] = query_id

            grouped_hits.append((query_id, hits))

        selected: list[Document] = []
        selected_by_key: dict[tuple[str, int], Document] = {}

        max_rank_count = max(
            (len(hits) for _, hits in grouped_hits),
            default=0,
        )

        # 先取各 query 的第 1 名，再取各 query 的第 2 名……
        # 这是一种简单的覆盖公平策略。
        for rank_index in range(max_rank_count):
            for query_id, hits in grouped_hits:
                if rank_index >= len(hits):
                    continue

                candidate = hits[rank_index]
                key = (
                    str(candidate.metadata.get("source", "")),
                    int(candidate.metadata.get("chunk_index", -1)),
                )

                existing = selected_by_key.get(key)
                if existing is not None:
                    existing_queries = {
                        value
                        for value in str(
                            existing.metadata.get("matched_queries", "")
                        ).split(",")
                        if value
                    }
                    existing_queries.add(query_id)
                    existing.metadata["matched_queries"] = ",".join(
                        sorted(existing_queries)
                    )
                    continue

                selected.append(candidate)
                selected_by_key[key] = candidate

                if len(selected) >= max_hits:
                    break

            if len(selected) >= max_hits:
                break

        logger.info(
            "多查询命中合并：queries=%d raw_hits=%d merged_hits=%d max_hits=%d",
            len(grouped_hits),
            sum(len(hits) for _, hits in grouped_hits),
            len(selected),
            max_hits,
        )

        return selected

    def format_documents(
            self,
            documents: list[Document],
            *,
            max_context_chars: int,
    ) -> str:
        """按照传入顺序格式化文档，并在完整 chunk 边界处执行预算截断。

        该方法不会重新排序 documents：
        - 如果上游传入的是 hit、hit-1、hit+1，那么输出仍保持这一局部顺序；
        - 不执行“所有命中优先、邻居后补”，避免破坏原文上下文关系。

        动态评估：
        - 观察日志中的候选字符数、实际输入字符数和截断位置；
        - 如果补充检索经常无法进入上下文，说明需要调整召回数量或预算分配；
        - 不建议仅因为模型窗口较大就无限增加内容，还应观察规划质量和 token 成本。
        """
        sections: list[str] = []
        used_chars = 0

        for document in documents:
            metadata = document.metadata

            # retrieve_many() 当前把 matched_queries 保存为逗号分隔字符串，
            # 因此这里不能再次使用 ",".join()，否则字符串会被逐字符拆开。
            matched_query_text = str(
                metadata.get("matched_queries", "")
            ).strip()

            header_parts = [
                f"source={metadata.get('source', 'unknown')}",
            ]

            # Markdown、TXT通常没有page元数据。
            # 缺少page时不输出page=unknown，保证模型引用格式与现有提示词一致。
            page = metadata.get("page")
            if page is not None:
                header_parts.append(f"page={page}")

            header_parts.extend([
                f"chunk={metadata.get('chunk_index', 'unknown')}",
                f"role={metadata.get('retrieval_role', 'hit')}",
            ])

            score = metadata.get("relevance_score")
            if isinstance(score, (int, float)):
                header_parts.append(f"score={float(score):.4f}")

            if matched_query_text:
                header_parts.append(f"queries={matched_query_text}")

            header = f"[{', '.join(header_parts)}]"

            section = f"{header}\n{document.page_content.strip()}"

            # 在完整 chunk 边界处截断，避免把一句话或代码块从中间切断。
            if sections and used_chars + len(section) > max_context_chars:
                break

            sections.append(section)
            used_chars += len(section)

        logger.info(
            "检索上下文格式化完成 documents=%s used_chars=%s max_chars=%s",
            len(sections),
            used_chars,
            max_context_chars,
        )

        # 保留项目现有证据块分隔符。
        # _limit_evidence_context() 正是通过该分隔符按完整证据块截断，
        # 如果这里只使用两个换行，整个检索结果可能被误认为一个超大证据块。
        return "\n\n---\n\n".join(sections)

    def expand_neighbors(
        self,
        hits: list[Document],
        *,
        window: int = 1,
    ) -> list[Document]:
        """为每个命中结果补充前后相邻 chunk。

        参数:
            hits: retrieve() 返回的命中结果列表。
            window: 每个命中结果向前/向后各取几个相邻 chunk。
                    例如 window=2，则每个命中会尝试拉取前 2 个和后 2 个 chunk。
        返回:
            扩展后的 Document 列表，每个 Document 的 metadata 中会包含:
            - retrieval_role: "hit"（真正命中）或 "neighbor"（上下文补充）
            - relevance_score: 仅 "hit" 继承原始的相似度分数
        """
        if window < 0:
            raise ValueError("window 不能小于 0。")

        # 存放扩展后的 Document（hit + neighbor）
        expanded: list[Document] = []
        # 去重，避免多个结果命中同一邻居
        seen: set[tuple[str, int]] = set()

        # 按 source 缓存该文件的全部 chunk，避免重复查询 Chroma。
        # 结构: { source字符串 -> { chunk_index整数 -> Document对象 } }
        source_cache: dict[str, dict[int, Document]] = {}

        for hit in hits:
            source = str(hit.metadata["source"])
            hit_index = int(hit.metadata["chunk_index"])

            # 如果该文件尚未缓存，则一次性从 Chroma 拉取该 source 的所有 chunk
            if source not in source_cache:
                stored = self.store.get(
                    where={"source": source},               # 元数据(metadata)筛选
                    include=["documents", "metadatas"],     # 指定返回内容，包括embeddings, documents, metadatas
                )

                # 构建 {chunk_index -> Document} 的映射字典
                chunk_map: dict[int, Document] = {}

                # zip 将 documents 列表和 metadatas 列表一一配对遍历
                for content, metadata in zip(
                    stored["documents"],
                    stored["metadatas"],
                ):
                    if content is None or metadata is None:
                        continue

                    # 提取 chunk_index，没有则跳过
                    index = metadata.get("chunk_index")
                    if index is None:
                        continue

                    # 重建为 Document 对象并存入映射表
                    # metadata 用 dict() 包装，避免共享引用
                    chunk_map[int(index)] = Document(
                        page_content=content,
                        metadata=dict(metadata),
                    )
                # 缓存该文件的所有 chunk 映射
                source_cache[source] = chunk_map
            # 从缓存中取出当前文件的分块映射
            chunk_map = source_cache[source]
            # 遍历命中位置 ± window 范围内的所有索引
            # range(start, end) 这里 end = hit_index + window + 1，确保包含右边界
            for index in range(
                    hit_index - window,
                    hit_index + window + 1,
            ):
                neighbor = chunk_map.get(index)
                if neighbor is None:
                    continue
                # 构建唯一标识，用于去重（这里的去重时基于 chunk 的，所以不用担心数据丢失问题
                identity = (source, index)
                if identity in seen:
                    continue

                neighbor.metadata["retrieval_role"] = (
                    "hit" if index == hit_index else "neighbor"
                )

                # 只有真正命中的 chunk 继承相关性分数和命中它的query。
                #
                # expand_neighbors() 会从Chroma重新构造Document，所以如果不显式复制，
                # retrieve_many() 写入hit.metadata的matched_queries会在这里丢失。
                if index == hit_index:
                    neighbor.metadata["relevance_score"] = (
                        hit.metadata.get("relevance_score")
                    )

                    matched_queries = hit.metadata.get("matched_queries")
                    if matched_queries:
                        neighbor.metadata["matched_queries"] = matched_queries

                expanded.append(neighbor)
                seen.add(identity)

        return expanded

    def retrieve_text(
            self,
            query: str,
            k: int = 6,
            *,
            min_relevance: float = 0.35,
            relative_margin: float = 0.12,
            neighbor_window: int = 1,
            max_context_chars: int = 16_000,
            exclude_suffixes: set[str] | None = None,
    ) -> str:
        """把检索结果转换成 LLM 上下文，检索、过滤并扩展相邻 chunk。

        role=hit 是真正向量命中；
        role=neighbor 是为了补上下文而加入；
        neighbor 不应该伪装成高相关结果；
        max_context_chars 防止多个命中扩展后无限膨胀。
        """
        hits = self.retrieve(
            query=query,
            k=k,
            min_relevance=min_relevance,
            relative_margin=relative_margin,
            min_hits=get_settings().rag_min_hits_per_query,
            exclude_suffixes=exclude_suffixes,
        )

        if not hits:
            return "没有达到相关性阈值的资料。"

        docs = self.expand_neighbors(
            hits,
            window=neighbor_window,
        )

        blocks: list[str] = []
        total_chars = 0

        if not docs:
            return "没有检索到相关资料。"
        blocks = []
        for doc in docs:
            page = (
                f", page={doc.metadata['page']}"
                if "page" in doc.metadata
                else ""
            )

            score = doc.metadata.get("relevance_score")
            score_text = (
                f", score={score:.4f}"
                if isinstance(score, float)
                else ""
            )

            role = doc.metadata.get(
                "retrieval_role",
                "hit",
            )

            block = (
                f"[source={doc.metadata['source']}{page}, "
                f"chunk={doc.metadata.get('chunk_index')}, "
                f"role={role}{score_text}]\n"
                f"{doc.page_content}"
            )

            if total_chars + len(block) > max_context_chars:
                break

            blocks.append(block)
            total_chars += len(block)

        logger.info(
            "最终上下文：命中 %d 个，扩展后 %d 个，输出 %d 字符。",
            len(hits),
            len(docs),
            total_chars,
        )

        return "\n\n---\n\n".join(blocks)
        """可能结果:

        [source=memory.md, chunk=2]
        短期记忆通常保存当前会话中的消息……

        ---

        [source=agent_notes.pdf, page=13, chunk=21]
        长期记忆可以跨会话保存用户目标、偏好与约束……
        """
        return "\n\n---\n\n".join(blocks)



