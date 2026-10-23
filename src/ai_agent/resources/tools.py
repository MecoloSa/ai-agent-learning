"""供智能体使用的本地资料工具。

这些工具只允许访问用户明确指定的资料目录，并限制单个文件读取长度，
这样既能保护文件边界，也能控制发送给模型的 token 数量。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import logging
import json

from pydantic import BaseModel, ConfigDict, Field, StrictInt
from langchain_core.tools import tool, BaseTool
from ai_agent.retrieval.rag import LearningRAG, needs_rag
from ai_agent.resources.materials import (
    iter_material_files,
    safe_material_path,
    read_material_lines,
    read_material_excerpt,
)
from ai_agent.config import get_settings
from ai_agent.resources.pdf_materials import read_pdf_pages

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
        logger.debug("目录 %s 中没有找到可分析的资料文件。", root)
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
        logger.debug("跳过 PDF 文件：%s", relative_path)
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
    *,
    max_queries_per_call: int | None = None,
    max_context_chars: int | None = None,
) -> BaseTool:
    """创建一个绑定资料目录的多query语义检索工具。

    root_dir由应用层固定，不暴露给模型生成。
    模型只能决定：
    1. 还缺少哪些信息，即queries；
    2. 是否需要读取命中块相邻的一个chunk。

    这比让模型传入任意root_dir更安全，也避免路径参数生成错误。
    交互模式可以传入独立配置，避免一次普通问答加载过多证据。
    """
    settings = get_settings()
    resolved_root = str(Path(root_dir).resolve())

    # 默认保持现有规划流程不变，交互模式创建工具时，会传入 INTERACTIEIVE_* 配置
    effective_query_limit = (
        max_queries_per_call
        if max_queries_per_call is not None
        else settings.plan_max_queries_per_tool_call
    )

    effective_context_limit = (
        max_context_chars
        if max_context_chars is not None
        else max(1, settings.rag_max_context_chars // 2)
    )
    # 延迟创建RAG实例：
    # 如果规划模型认为初始证据已经充分并且不调用工具，
    # 就不会产生额外的索引同步和Embedding初始化开销。
    rag_instance: LearningRAG | None = None

    @tool("retrieve_learning_context")
    def retrieve_learning_context(
        queries: list[str],
        neighbor_window: int = 1,
    ) -> str:
        """按需检索当前必要的信息，返回可引用的本地资料证据。

        Args:
            queries:
                聚焦问题默认只提交一个明确查询。
                多个已知且互补、无需等待彼此结果的信息缺口可以合并提交。
                近义改写应合并；必须先看结果才能确定的下一问留到后续调用。
                查询需说明证据用途，例如目录条目、原句或正文解释。
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

        query_limit = effective_query_limit
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

        logger.debug(
            "检索请求：query_count=%d neighbor_window=%d",
            len(query_pairs),
            neighbor_window,
        )

        # 仅显式开启内容日志时展示实际查询，避免默认记录用户问题。
        # 动态评估：检查是否出现近义重复、遗漏用途或无必要的查询拆分。
        if settings.log_payloads:
            logger.debug(
                "检索请求参数：queries=%s",
                normalized_queries,
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

            # 限制本次格式化证据正文，不是会话累计余额。
            # 调用方可显式指定；未指定时默认取RAG预算的一半。
            # 多次工具结果仍会在本轮上下文中累积，另受调用次数约束。
            max_context_chars=effective_context_limit,
        )

        logger.debug(
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
    """创建绑定资料目录的分析工具，模型不能选择或替换根目录。"""
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("资料目录不存在")

    current_settings = get_settings()

    class NoExtraArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class ReadArgs(NoExtraArgs):
        relative_path: str = Field(min_length=1)
        max_chars: StrictInt = Field(
            default=MAX_CHARS_PER_FILE,
            ge=1,
            le=MAX_CHARS_PER_FILE,
        )

    class SearchArgs(NoExtraArgs):
        keyword: str = Field(min_length=1)

    def safe_files():
        # 固定根目录还不够：目录中的文件链接也可能指向外部。
        # 复用解析后的边界检查，列举与搜索都不接受这类文件。
        for path in iter_material_files(root):
            relative_path = path.relative_to(root).as_posix()
            try:
                checked_path = safe_material_path(
                    str(root), relative_path
                )
            except (ValueError, OSError):
                continue
            yield path, checked_path

    @tool(
        "list_learning_files",
        args_schema=NoExtraArgs.model_json_schema(),
    )
    def bound_list_files(**arguments: object) -> str:
        """列出当前绑定目录中的 TXT、MD、PY、PDF 文件，无需目录参数。"""
        # 当前 LangChain 会跳过空 Pydantic schema 的参数校验。
        # 显式提供 JSON schema，让原始参数进入函数，再在扫描前校验。
        # 动态评估：{} 应成功，任何额外字段都必须在读取目录前被拒绝。
        NoExtraArgs.model_validate(arguments)

        files = list(safe_files())
        if not files:
            return "没有找到可分析的资料文件。"

        return "\n".join(
            f"{path.relative_to(root).as_posix()} "
            f"({checked_path.stat().st_size} bytes)"
            for path, checked_path in (
                files[:current_settings.tool_file_list_limit]
            )
        )

    @tool("read_learning_file", args_schema=ReadArgs)
    def bound_read_file(
        relative_path: str,
        max_chars: int = MAX_CHARS_PER_FILE,
    ) -> str:
        """读取绑定目录内 TXT、MD、PY 文件的开头，使用清单中的相对路径。"""
        if Path(relative_path).is_absolute():
            raise ValueError("请使用资料目录内的相对路径")

        data = read_material_excerpt(
            str(root),
            relative_path,
            max_chars,
        )
        # citation 由程序生成，随正文一起交给模型。
        return json.dumps(
            {"ok": True, "data": data},
            ensure_ascii=False,
        )

    @tool("search_learning_files", args_schema=SearchArgs)
    def bound_search_files(keyword: str) -> str:
        """在绑定目录的 TXT、MD、PY 文件中查找关键词，返回文件名和行号。"""
        results: list[str] = []
        for path, checked_path in safe_files():
            if checked_path.suffix.lower() == ".pdf":
                continue

            try:
                lines = checked_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                continue

            for line_number, line in enumerate(lines, start=1):
                if keyword.casefold() in line.casefold():
                    results.append(
                        f"{path.relative_to(root).as_posix()}:"
                        f"{line_number}: {line.strip()[:160]}"
                    )
                    if (
                        len(results)
                        >= current_settings.tool_search_result_limit
                    ):
                        return "\n".join(results)

        return "\n".join(results) or f"未找到关键词：{keyword}"

    tools: list[BaseTool] = [
        bound_list_files,
        bound_read_file,
        bound_search_files,
        summarize_text_statistics,
    ]
    if needs_rag(str(root)):
        tools.append(build_retrieve_learning_context_tool(str(root)))

    # 动态评估：工具 schema 中不应再有 root_dir；
    # 正常读取应成功，额外目录参数及越界路径应被拒绝。
    return tools

def build_read_material_lines_tool(root_dir: str) -> BaseTool:
    """创建绑定资料目录的只读工具；模型不能指定或切换根目录。"""
    resolved_root = str(Path(root_dir).expanduser().resolve())

    @tool("read_material_lines")
    def read_lines(
        relative_path: str,
        start_line: StrictInt,
        end_line: StrictInt,
    ) -> str:
        """读取已知 TXT/MD/PY 文件的指定行，不执行代码。

        relative_path 是资料目录内的相对路径。
        行号从1开始，包含两端，每次最多200行。
        需要已知文件的准确正文时使用；不支持PDF，不负责搜索未知文件。
        返回JSON，包含实际行号、正文及截断标记。
        truncated=true或partial=true时，不能把返回内容当作完整原文。
        """
        # StrictInt 在工具参数校验阶段拒绝字符串、浮点数和布尔值。
        # 底层函数继续校验目录边界、文件类型、范围和输出大小。
        data = read_material_lines(
            resolved_root,
            relative_path,
            start_line,
            end_line,
        )

        # 引用位置由程序按实际返回范围生成，不让模型猜测行号。
        data["citation"] = (
            f"[source={data['source']}, "
            f"lines={data['returned_start_line']}-{data['returned_end_line']}]"
        )
        return json.dumps(
            {"ok": True, "data": data},
            ensure_ascii=False,
        )

    return read_lines

def build_list_material_files_tool(root_dir: str) -> BaseTool:
    """创建绑定资料目录的分页清单工具，不读取文件正文。"""
    resolved_root = Path(root_dir).expanduser().resolve()

    @tool("list_material_files")
    def list_files(
        offset: StrictInt = 0,
        limit: StrictInt = 20,
    ) -> str:
        """列出当前资料目录中的 PDF/MD/TXT/PY 相对路径与类型。

        文件名未知时可先调用本工具，再选择读取或检索。
        offset从0开始；limit为1到20；根据next_offset获取下一页。
        返回JSON。清单不是正文证据，不能据此猜测文件内容。
        根目录由程序固定，模型不能切换目录。
        """
        if offset < 0:
            raise ValueError("offset必须大于等于0")
        if not 1 <= limit <= 20:
            raise ValueError("limit必须在1到20之间")
        if not resolved_root.is_dir():
            raise ValueError("资料目录不存在")

        relative_paths: list[str] = []

        for path in iter_material_files(resolved_root):
            try:
                relative_path = path.relative_to(resolved_root).as_posix()

                # 复用读取边界检查，避免列出无法安全访问的越界符号链接。
                # ValueError表示该路径不属于允许访问的资料范围。
                safe_material_path(str(resolved_root), relative_path)
            except ValueError:
                continue

            relative_paths.append(relative_path)

        # 统一按返回的相对路径排序；目录不变时分页顺序稳定。
        relative_paths.sort()
        total_files = len(relative_paths)

        if offset > total_files:
            raise ValueError("offset超过当前资料清单范围")

        page = relative_paths[offset:offset + limit]
        next_offset = offset + len(page)
        has_more = next_offset < total_files

        data = {
            "files": [
                {
                    "relative_path": relative_path,
                    "file_type": Path(relative_path).suffix.lower()[1:],
                }
                for relative_path in page
            ],
            "offset": offset,
            "limit": limit,
            "returned_count": len(page),
            "total_files": total_files,
            "has_more": has_more,
            "next_offset": next_offset if has_more else None,
        }

        # 只记录规模，不输出完整路径清单或资料正文。
        # 动态评估：观察清单调用次数及是否经常耗尽两次工具预算；
        # 不应仅为了遍历整个目录而持续调用。
        logger.debug(
            "交互资料清单：total=%d offset=%d returned=%d has_more=%s",
            total_files,
            offset,
            len(page),
            has_more,
        )

        return json.dumps(
            {"ok": True, "data": data},
            ensure_ascii=False,
        )

    return list_files

def build_read_pdf_pages_tool(root_dir: str) -> BaseTool:
    """创建绑定资料目录的PDF按页读取工具。"""
    resolved_root = str(Path(root_dir).expanduser().resolve())

    @tool("read_pdf_pages")
    def read_pages(
        relative_path: str,
        start_page: StrictInt,
        end_page: StrictInt,
    ) -> str:
        """读取已知PDF指定物理页的可提取文字。

        relative_path为资料目录内的相对路径，不能切换根目录。
        页码从1开始，包含两端，每次最多3页、返回正文最多8000字符。
        物理页序号可能不同于书中印刷页码。

        返回JSON，包含实际页码、逐页文字、截断情况与页级引用。
        text_status=no_text表示没有提取到非空文字，不等于页面空白。
        partial或truncated为true时，不得声称返回了完整文字。
        不支持OCR、加密PDF，不保证图片、表格和公式信息完整。
        """
        # 模型只决定相对路径与页码；目录、边界和输出限制由程序控制。
        data = read_pdf_pages(
            resolved_root,
            relative_path,
            start_page,
            end_page,
        )

        # 引用已由底层函数按实际页序号生成，不让模型补造chunk。
        # 即便无文字，读取尝试也可能成功；回答必须解释no_text限制。
        return json.dumps(
            {"ok": True, "data": data},
            ensure_ascii=False,
        )

    return read_pages

# 暂时保留名称，避免其他尚未迁移的导入立即失败。
# 新的Agent流程应使用build_agent_tools(material_dir)。
TOOLS = BASE_TOOLS
