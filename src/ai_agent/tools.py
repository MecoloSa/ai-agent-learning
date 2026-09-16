"""供智能体使用的本地资料工具。

这些工具只允许访问用户明确指定的资料目录，并限制单个文件读取长度，
这样既能保护文件边界，也能控制发送给模型的 token 数量。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import logging

from langchain_core.tools import tool, BaseTool
from .rag import LearningRAG, needs_rag
from .materials import (
    iter_material_files,
    safe_material_path,
)
from .config import get_settings
settings = get_settings()

logger = logging.getLogger("ai_agent.tools")
MAX_CHARS_PER_FILE = settings.tool_max_chars_per_file

@tool
def list_learning_files(root_dir: str) -> str:
    """列出资料目录中的文本、Markdown、Python 和 PDF 文件及其大小。"""
    logger.debug("调用 list_learning_files(root_dir=%r)", root_dir)
    root = Path(root_dir).resolve()
    if not root.is_dir():
        logger.warning("目录不存在：%s", root)
        return f"目录不存在：{root}"
    files = list(iter_material_files(root))
    if not files:
        logger.info("目录 %s 中没有找到可分析的资料文件。", root)
        return "没有找到可分析的资料文件。"
    result = "\n".join(
        f"{p.relative_to(root)} ({p.stat().st_size} bytes)"
        for p in files[:settings.tool_file_list_limit]
    )
    logger.debug(
        "资料扫描完成：root=%s files=%d returned=%d",
        root,
        len(files),
        min(len(files), settings.tool_file_list_limit),
    )
    return result


@tool
def read_learning_file(root_dir: str, relative_path: str, max_chars: int = MAX_CHARS_PER_FILE) -> str:
    """读取一个 TXT、MD 或 PY 资料的开头内容；最多读取 MAX_CHARS_PER_FILE 个字符。"""
    logger.debug(
        "调用 read_learning_file(root_dir=%r, relative_path=%r, max_chars=%d)",
        root_dir, relative_path, max_chars,
    )
    path = safe_material_path(root_dir, relative_path)
    if path.suffix.lower() == ".pdf":
        logger.info("跳过 PDF 文件：%s", relative_path)
        return "PDF 暂不直接读取；请转换为 TXT 或 Markdown 后再分析。"
    if not path.is_file():
        logger.warning("文件不存在：%s", relative_path)
        return f"文件不存在：{relative_path}"
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-8-sig", errors="replace")
    truncated = content[: min(max_chars, MAX_CHARS_PER_FILE)]
    logger.debug(
        "读取资料文件：file=%s original_chars=%d returned_chars=%d truncated=%s",
        relative_path,
        len(content),
        len(truncated),
        len(truncated) < len(content),
    )
    return truncated


@tool
def search_learning_files(root_dir: str, keyword: str) -> str:
    """在 TXT、MD、PY 资料中检索关键词，返回最多 12 条包含文件名的上下文。"""
    root = Path(root_dir).resolve()
    results: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".py"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if keyword.lower() in line.lower():
                results.append(f"{path.relative_to(root)}:{line_number}: {line.strip()[:160]}")
                if len(results) >= settings.tool_search_result_limit:
                    return "\n".join(results)
    return "\n".join(results) if results else f"未找到关键词：{keyword}"


@tool
def summarize_text_statistics(text: str) -> str:
    """统计一段文本的字数、行数和高频词，帮助判断资料规模和重点。"""
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text.lower())
    common = Counter(words).most_common(8)
    frequency = "、".join(f"{word}:{count}" for word, count in common) or "无"
    return f"字符数：{len(text)}；行数：{len(text.splitlines())}；高频词：{frequency}"

BASE_TOOLS: list[BaseTool] = [
    list_learning_files,
    read_learning_file,
    search_learning_files,
    summarize_text_statistics,
]


def build_retrieve_learning_context_tool(
    root_dir: str,
) -> BaseTool:
    """创建一个绑定资料目录的多query语义检索工具。

    root_dir由应用层固定，不暴露给模型生成。
    模型只能决定：
    1. 还缺少哪些信息，即queries；
    2. 是否需要读取命中块相邻的一个chunk。

    这比让模型传入任意root_dir更安全，也避免路径参数生成错误。
    """
    settings = get_settings()
    resolved_root = str(Path(root_dir).resolve())

    # 延迟创建RAG实例：
    # 如果规划模型认为初始证据已经充分并且不调用工具，
    # 就不会产生额外的索引同步和Embedding初始化开销。
    rag_instance: LearningRAG | None = None

    @tool("retrieve_learning_context")
    def retrieve_learning_context(
        queries: list[str],
        neighbor_window: int = 1,
    ) -> str:
        """一次检索多个明确的信息缺口，返回可引用的本地资料证据。

        Args:
            queries:
                需要补充检索的具体问题列表。应一次提交当前已经发现的
                所有重要缺口，而不是每次只提交一条近义query。
            neighbor_window:
                0表示只返回直接命中chunk；
                1表示返回命中chunk及前后各一个chunk。
        """
        nonlocal rag_instance

        normalized_queries: list[str] = []
        seen_queries: set[str] = set()

        for query in queries:
            normalized = " ".join(str(query).split()).strip()
            key = normalized.casefold()

            if not normalized or key in seen_queries:
                continue

            # 避免模型误把整段规划内容作为query传入。
            normalized_queries.append(normalized[:300])
            seen_queries.add(key)

        if not normalized_queries:
            return "检索未执行：queries中没有有效查询。"

        query_limit = settings.plan_max_queries_per_tool_call
        original_query_count = len(normalized_queries)
        limit_notice = ""

        if original_query_count > query_limit:
            # 提示词已经要求模型按重要程度排列query，
            # 因此超限时保留前N条，而不是让整次工具调用失败。
            #
            # 这对pipeline模式尤其重要：
            # pipeline只有一次额外检索机会，如果整体拒绝，
            # 就会因为一次参数超限而完全失去补充证据。
            normalized_queries = normalized_queries[:query_limit]

            # 该提示会随工具结果返回给模型，使模型知道并非所有query都已执行，
            # 避免它错误地认为被截断的内容已经检索过。
            limit_notice = (
                f"注意：模型提交了{original_query_count}条query，"
                f"超过单次上限{query_limit}条；"
                f"本次只执行按重要程度排列在前面的{query_limit}条。"
            )

            logger.warning(
                "检索query数量超过上限，已按顺序截取："
                "original=%d retained=%d",
                original_query_count,
                len(normalized_queries),
            )

        if neighbor_window not in {0, 1}:
            return (
                "检索未执行：neighbor_window只能为0或1。"
                "0表示只取命中chunk，1表示取命中chunk±1。"
            )

        if rag_instance is None:
            rag_instance = LearningRAG(
                resolved_root,
                data_dir=settings.rag_data_dir,
            )

        query_pairs = tuple(
            (f"tool_query_{index}", query)
            for index, query in enumerate(
                normalized_queries,
                start=1,
            )
        )

        hits = rag_instance.retrieve_many(
            query_pairs,
            k_per_query=settings.rag_retrieval_k,

            # 不在工具内部过早删除某条query的结果。
            # 跨query重复chunk仍会由retrieve_many统一去重。
            max_hits=(
                settings.rag_retrieval_k
                * len(query_pairs)
            ),
            min_relevance=settings.rag_min_relevance,
            relative_margin=settings.rag_relative_margin,
            min_hits=settings.rag_min_hits_per_query,
            exclude_suffixes={".py"},
        )

        if not hits:
            no_hit_message = (
                "没有达到相关性阈值的新增资料。"
                "这只表示本次实际执行的query没有命中，"
                "不能据此断言原始资料不存在相关内容。"
            )

            # 如果曾发生query截断，要让模型知道无命中结论
            # 只针对实际执行的前N条query。
            return "\n\n".join(
                part
                for part in (limit_notice, no_hit_message)
                if part
            )

        documents = rag_instance.expand_neighbors(
            hits,
            window=neighbor_window,
        )

        result = rag_instance.format_documents(
            documents,

            # 单次补充检索最多使用总RAG预算的一半，
            # 防止Agent连续调用时工具结果无限累积。
            max_context_chars=max(
                1,
                settings.rag_max_context_chars // 2,
            ),
        )

        logger.info(
            "规划检索工具完成：queries=%d hits=%d "
            "expanded_documents=%d result_chars=%d",
            len(query_pairs),
            len(hits),
            len(documents),
            len(result),
        )

        if not result:
            result = "没有检索到可返回的资料证据。"

        # 正常返回证据的同时，明确告诉模型是否有query被截断。
        # 如果没有发生截断，limit_notice为空，不会增加多余文本。
        return "\n\n".join(
            part
            for part in (limit_notice, result)
            if part
        )

    return retrieve_learning_context


def build_agent_tools(root_dir: str) -> list[BaseTool]:
    """根据当前语料创建Agent可用工具。

    小型MD/TXT或纯Python目录不需要向量检索工具；
    Python源码仍通过read_learning_file或确定性全量读取处理。
    """
    tools = list(BASE_TOOLS)

    if needs_rag(root_dir):
        tools.append(
            build_retrieve_learning_context_tool(root_dir)
        )

    return tools


# 暂时保留名称，避免其他尚未迁移的导入立即失败。
# 新的Agent流程应使用build_agent_tools(material_dir)。
TOOLS = BASE_TOOLS
