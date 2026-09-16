"""两个职责清晰的智能体，以及它们之间的协作流程。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_exponential, before_log, before_sleep_log
)

from .config import create_llm, create_llm_with_tools, get_settings
from .tools import (
    read_learning_file,
    list_learning_files,
    summarize_text_statistics,
    build_agent_tools,
    build_retrieve_learning_context_tool,
)
from .rag import LearningRAG, needs_rag
# 用于识别当前语料实际包含哪些文件类型，这里只读取文件路径。
from .materials import iter_material_files

import hashlib
import logging
import json
import re

logger = logging.getLogger("ai_agent.agents")
def _content_to_text(content) -> str:
    """把 LangChain 的字符串或结构化 content 转成可记录文本。"""
    if isinstance(content, str):
        return content

    try:
        return json.dumps(
            content,
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError):
        return str(content)


def _preview(text: str, limit: int) -> str:
    """生成有长度上限的日志预览。"""
    normalized = text.replace("\x00", "")
    if len(normalized) <= limit:
        return normalized

    omitted = len(normalized) - limit
    return f"{normalized[:limit]}\n...<省略 {omitted} 字符>"


def _log_model_request(
    stage: str,
    messages,
) -> None:
    """记录模型请求结构；按配置决定是否显示截断预览。"""
    settings = get_settings()
    message_list = list(messages)

    message_info: list[dict[str, object]] = []
    full_text_parts: list[str] = []

    for message in message_list:
        message_type = getattr(
            message,
            "type",
            message.__class__.__name__,
        )
        text = _content_to_text(
            getattr(message, "content", "")
        )

        message_info.append({
            "type": message_type,
            "chars": len(text),
        })
        full_text_parts.append(
            f"[{message_type}]\n{text}"
        )

    request_text = "\n\n".join(full_text_parts)
    request_fingerprint = hashlib.sha256(
        request_text.encode("utf-8")
    ).hexdigest()[:12]

    logger.debug(
        "模型请求：stage=%s model=%s "
        "temperature=%.3f timeout=%.1f "
        "messages=%d total_chars=%d fingerprint=%s "
        "message_info=%s",
        stage,
        settings.chat_model,
        settings.llm_temperature,
        settings.request_timeout,
        len(message_list),
        len(request_text),
        request_fingerprint,
        message_info,
    )

    if not settings.log_payloads:
        return

    # Agentic 循环会不断累积消息。这里只预览最后 3 条，
    # 避免每一轮都重复打印全部历史。
    preview_start = max(
        0,
        len(message_list) - 3,
    )

    for index, message in enumerate(
        message_list[preview_start:],
        start=preview_start,
    ):
        message_type = getattr(
            message,
            "type",
            message.__class__.__name__,
        )
        text = _content_to_text(
            getattr(message, "content", "")
        )

        logger.debug(
            "模型输入预览：stage=%s index=%d "
            "type=%s chars=%d content=%r",
            stage,
            index,
            message_type,
            len(text),
            _preview(
                text,
                settings.log_preview_chars,
            ),
        )


def _log_model_response(
    stage: str,
    response,
) -> None:
    """记录模型响应元数据和可选的正文预览。"""
    settings = get_settings()
    content = _content_to_text(
        getattr(response, "content", "")
    )
    metadata = (
        getattr(response, "response_metadata", None)
        or {}
    )
    tool_calls = (
        getattr(response, "tool_calls", None)
        or []
    )

    tool_names = [
        call.get("name", "<unknown>")
        for call in tool_calls
        if isinstance(call, dict)
    ]

    logger.debug(
        "模型响应：stage=%s response_id=%s "
        "model=%s finish_reason=%s "
        "content_chars=%d tool_calls=%d tools=%s",
        stage,
        getattr(response, "id", None),
        metadata.get("model_name")
        or metadata.get("model")
        or settings.chat_model,
        metadata.get("finish_reason"),
        len(content),
        len(tool_calls),
        tool_names,
    )

    if settings.log_payloads:
        logger.debug(
            "模型输出预览：stage=%s chars=%d content=%r",
            stage,
            len(content),
            _preview(
                content,
                settings.log_preview_chars,
            ),
        )

settings = get_settings()

# 原来的 TOOL_MAP 为全局，现在的工具绑定具体的资料目录
MAX_TOOL_ROUNDS = settings.agent_max_tool_rounds
MAX_PLAN_EVIDENCE_CHARS = settings.plan_max_evidence_chars

RETRY_STOP = stop_after_attempt(settings.llm_retry_attempts)
RETRY_WAIT = wait_exponential(
    multiplier=1,
    min=settings.llm_retry_min_wait,
    max=settings.llm_retry_max_wait,
)

class LLMInvocationError(RuntimeError):
    """模型调用最终失败时抛出，携带面向用户的原因。"""

class RetryableLLMError(LLMInvocationError):
    """可重试错误（超时、限流、服务端错误等）；@retry 只重试本类。"""

def _is_retryable(exc: BaseException) -> bool:
    """断是否值得重试：超时、限流、服务端错误重试；鉴权/参数错误不重试。"""
    text = str(exc).lower()
    retryable_makers = ("timeout", "time out", "429", "rate limit", "502", "timed out",
                        "503", "504", "service not available", "connection")
    return any(m in text for m in retryable_makers)

@retry(
    retry=retry_if_exception_type(RetryableLLMError),
    stop=RETRY_STOP,
    wait=RETRY_WAIT,
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _invoke_llm_raw(llm, messages, *, stage: str):
    """带重试的模型调用包装：返回完整响应对象（含 tool_calls）。

    Agentic 循环需要检查 response.tool_calls，因此必须保留完整对象；
    纯文本调用（如 _invoke_llm）再从这里取 .content。
    """
    try:
        import time
        # 所有经过 _invoke_llm_raw() 的调用都会自动获得一致日志
        # 包括：
        #   pipeline 资料分析；
        #   Agentic 每一轮；
        #   学习计划生成；
        #   重试后的再次请求。
        _log_model_request(
            stage,
            messages,
        )

        start = time.perf_counter()
        response = llm.invoke(messages)
        elapsed = time.perf_counter() - start

        _log_model_response(
            stage,
            response,
        )
        usage = getattr(
            response,
            "usage_metadata",
            None,
        )
        logger.info(
            "【%s】模型调用完成：耗时=%.1f秒 tokens=%s",
            stage,
            elapsed,
            (
                dict(usage)
                if usage
                else "未知"
            ),
        )
        return response
    except Exception as exc:
        if _is_retryable(exc):
            # 交给 before_sleep_log 每次只记录一条重试日志。
            raise RetryableLLMError(
                f"{stage} 模型调用失败（可重试）：{exc}"
            ) from exc

        # 最终由 main() 在流程边界统一记录一次。
        raise LLMInvocationError(
            f"{stage} 模型调用失败：{exc}"
        ) from exc

def invoke_llm_response(
    llm,
    messages,
    *,
    stage: str,
):
    """公开的统一模型调用入口，保留响应对象和工具调用信息。"""
    return _invoke_llm_raw(
        llm,
        messages,
        stage=stage,
    )

def _invoke_llm(llm, messages, *, stage:str) -> str:
    """纯文本包装：取 .content，重试由统一入口 invoke_llm_response 提供。"""
    response = invoke_llm_response(llm, messages, stage=stage)
    return _content_to_text(response.content)

_CITATION_PATTERN = re.compile(
    r"\[source=[^\]\n]+\]"
)

def log_citation_summary(
    stage: str,
    content: str,
) -> None:
    """简要记录模型回答实际使用了哪些资料引用。"""
    # dict.fromkeys 保留原顺序的同时去重
    citations = list(dict.fromkeys(
        _CITATION_PATTERN.findall(content)
    ))

    if not citations:
        logger.warning(
            "回答未包含资料引用：stage=%s",
            stage,
        )
        return

    max_display = 12
    displayed = citations[:max_display]
    omitted = len(citations) - len(displayed)

    logger.info(
        "回答引用汇总：stage=%s count=%d "
        "citations=%s omitted=%d",
        stage,
        len(citations),
        displayed,
        omitted,
    )

def collect_material_context(material_dir: str) -> str:
    """先用确定性的工具收集少量上下文，再交给模型。

    这样不需要让模型反复决定要调用哪个工具，通常只需两次模型请求。
    """
    file_list = list_learning_files.invoke({"root_dir": material_dir})
    root = Path(material_dir)
    settings = get_settings()
    snippets: list[str] = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            # Python源码由 _collect_python_sources() 统一完整读取，
            # 这里仅收集MD/TXT摘要，避免同一代码重复进入上下文。
            if (
                    len(snippets) >= settings.material_snippet_file_limit
                    or path.suffix.lower() not in {".txt", ".md"}
            ):
                continue
            if path.is_file():
                relative_path = path.relative_to(root)
                snippets.append(
                    f"\n--- {relative_path} ---\n"
                    + read_learning_file.invoke({
                        "root_dir": material_dir,
                        "relative_path": str(relative_path),
                        "max_chars": settings.material_snippet_max_chars,
                    })
                )
    combined = "\n".join(snippets)
    stats = summarize_text_statistics.invoke({"text": combined}) if combined else "无可读文本。"
    return f"文件清单：\n{file_list}\n\n资料摘录：{combined}\n\n资料统计：{stats}"

def _collect_python_sources(
    material_dir: str,
) -> str:
    """确定性地完整读取资料目录中的所有Python源码。

    Python源码不使用向量相似度检索，也不设置字符阈值。
    每个文件保留source，供分析和学习计划引用。

    注意：
        本函数不负责限制模型上下文。
        所有.py文件都会完整进入material_context。
    """
    root = Path(material_dir).resolve()

    python_paths = sorted(
        (
            path
            for path in iter_material_files(root)
            if path.suffix.lower() == ".py"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )

    if not python_paths:
        logger.info("资料目录中没有Python源码。")
        return ""

    blocks: list[str] = []
    total_chars = 0

    for path in python_paths:
        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        source = path.relative_to(root).as_posix()

        blocks.append(
            f"[source={source}, mode=full_read]\n"
            f"{content}"
        )
        total_chars += len(content)

    logger.info(
        "Python源码已全部完整读取：files=%d chars=%d",
        len(python_paths),
        total_chars,
    )

    return "\n\n---\n\n".join(blocks)

def _build_broad_retrieval_queries(
    material_dir: str,
    learning_goal: str,
) -> tuple[tuple[str, str], ...]:
    """根据实际文件类型构造四个短而单一的检索意图。

    动态评估：
    - 观察四类 query 的命中预览是否各自集中在对应主题；
    - 如果不同 query 仍大量命中相同 chunk，说明需要进一步细化主题，
      而不是继续向 query 中堆叠关键词。
    """
    root = Path(material_dir).resolve()
    suffixes = {
        path.suffix.lower()
        for path in iter_material_files(root)
    }

    # 用户目标可能包含换行或多余空格，先压缩成适合向量化的一句话。
    goal = " ".join(learning_goal.split())

    organization_focus: list[str] = []
    implementation_focus: list[str] = []
    validation_focus: list[str] = []

    if ".pdf" in suffixes:
        organization_focus.append("章节关系、前置知识与学习顺序")
        implementation_focus.append("正文中的核心机制与技术原理")
        validation_focus.append("实验、示例、练习与验收方式")

    if ".md" in suffixes:
        organization_focus.append("文档标题结构与文件关系")
        implementation_focus.append("架构说明、配置与接口")
        validation_focus.append("安装、使用示例与故障排查")

    if ".txt" in suffixes:
        organization_focus.append("主题关系与要点顺序")
        implementation_focus.append("具体流程、规则与限制")
        validation_focus.append("实践步骤、结果与注意事项")

    # 如果只有 Python，下面四类文档查询不会真正用于 RAG；
    # 保留回退值可以避免函数返回含义不完整。
    organization_need = "；".join(organization_focus) or "资料结构与内容关系"
    implementation_need = "；".join(implementation_focus) or "核心实现与数据流"
    validation_need = "；".join(validation_focus) or "使用方法与验证方式"

    return (
        (
            "goal",
            f"{goal}；核心主题、关键定义与工作原理",
        ),
        (
            "organization",
            f"{goal}；{organization_need}",
        ),
        (
            "implementation",
            f"{goal}；{implementation_need}",
        ),
        (
            "usage_validation",
            f"{goal}；{validation_need}",
        ),
    )

def analyse_materials(material_dir: str, learning_goal: str) -> tuple[str, str]:
    """资料分析智能体：把原始资料压缩为学习重点。

    返回 (分析结果, 资料上下文)。资料上下文就是 collect_material_context
    收集到的文件清单/摘录/统计，供 interactive 模式直接复用，避免重复读文件。
    """
    logger.info(
        "【资料分析智能体】开始，material_dir=%s, goal=%s",
        material_dir,
        learning_goal,
    )

    # 不论其他资料是否进入RAG，Python源码始终完整读取。
    python_context = _collect_python_sources(
        material_dir=material_dir,
    )

    if needs_rag(material_dir):
        settings = get_settings()
        rag = LearningRAG(
            material_dir,
            data_dir=settings.rag_data_dir,
        )

        # 第一步：不使用相似度，从书籍开头提取目录、前言和阅读指引。
        overview = rag.build_overview_text(
            max_pdf_page=settings.rag_overview_max_pdf_pages,
            max_chars=settings.rag_overview_max_chars,
        )

        # 第二步：使用多查询检索机制，针对学习目标检索具体内容。
        # 仍然只执行四次向量检索：
        # 1. 目标相关事实；
        # 2. 资料/代码组织关系；
        # 3. 技术实现；
        # 4. 使用、实践与验证。
        #
        # 不依赖“章节结构”，也不增加额外 LLM 调用。
        retrieval_queries = _build_broad_retrieval_queries(
            material_dir=material_dir,
            learning_goal=learning_goal,
        )
        # retrieval_queries = (
        #     (
        #         "goal",
        #         f"查找与以下学习目标直接相关的核心原理、技术机制和正文论述："
        #         f"{learning_goal}",
        #     ),
        #     (
        #         "structure",
        #         "查找资料各主要章节的核心知识、章节依赖关系和推荐学习顺序。"
        #         "优先返回正文和章节小结，避免只返回目录、前言或阅读指引。",
        #     ),
        #     (
        #         "implementation",
        #         "查找具体技术实现、算法过程、架构设计、代码说明和工程约束。"
        #         "优先返回包含实现细节的正文。",
        #     ),
        #     (
        #         "practice",
        #         "查找资料中的实验、实践步骤、代码仓库说明、运行条件和验收标准。",
        #     ),
        # )

        for query_id, _ in retrieval_queries:
            logger.info(
                "RAG 检索任务：query_id=%s",
                query_id,
            )

        # 先执行四类检索并统一合并直接命中的chunk。
        #
        # max_hits使用“四类query各自最多k条”的理论上限，
        # 目的是不因为引入retrieve_many而主动缩小原有召回范围。
        # 真正的输入上限仍由rag_max_context_chars控制。
        merged_hits = rag.retrieve_many(
            retrieval_queries,
            k_per_query=settings.rag_retrieval_k,
            max_hits=(
                    settings.rag_retrieval_k
                    * len(retrieval_queries)
            ),
            min_relevance=settings.rag_min_relevance,
            relative_margin=settings.rag_relative_margin,
            min_hits=settings.rag_min_hits_per_query,

            # Python源码仍走确定性全量读取，不重复进入向量上下文。
            exclude_suffixes={".py"},
        )

        # 统一合并hit之后只扩展一次邻居。
        #
        # expand_neighbors()仍按照：
        # hit-window → ... → hit → ... → hit+window
        # 的局部顺序输出，不采用“所有hit优先”的重新排序。
        expanded_documents = rag.expand_neighbors(
            merged_hits,
            window=settings.rag_neighbor_window,
        )

        if expanded_documents:
            targeted_evidence = rag.format_documents(
                expanded_documents,
                max_context_chars=settings.rag_max_context_chars,
            )
        else:
            targeted_evidence = "没有达到相关性阈值的资料。"

        logger.info(
            "多查询上下文生成完成：direct_hits=%d expanded_documents=%d "
            "context_chars=%d",
            len(merged_hits),
            len(expanded_documents),
            len(targeted_evidence),
        )

        context_parts = [
            (
                "## 非Python资料结构概览"
                "（确定性读取，非向量检索）\n"
                f"{overview}"
            ),
            (
                "## PDF、Markdown和TXT检索证据"
                "（向量检索并扩展上下文）\n"
                f"{targeted_evidence}"
            ),
        ]

        if python_context:
            context_parts.append(
                "## Python源码（确定性全量读取，非向量检索）\n"
                f"{python_context}"
            )

        context = "\n\n".join(context_parts)
    else:
        # 这里只收集较小的MD/TXT资料。
        context = collect_material_context(material_dir)

        # 即使整个目录没有进入RAG，Python仍然必须完整进入上下文。
        if python_context:
            context += (
                "\n\n## Python源码"
                "（确定性全量读取，非向量检索）\n"
                f"{python_context}"
            )
    logger.debug("资料分析上下文统计：chars=%d", len(context),)
    llm = create_llm()
    messages = [
        SystemMessage(content=(
            "你是资料结构与学习路径分析智能体。"
            "仅依据提供的资料概览和证据工作，不得编造。\n\n"

            "请输出以下结构：\n"
            "1. 资料定位：主题、目标读者、覆盖范围；\n"
            "2. 知识结构：按基础层、核心层、进阶层组织；\n"
            "3. 前置依赖：说明为什么必须先学；\n"
            "4. 核心模块：每个模块说明概念、价值、学习难点；\n"
            "5. 实践项目：说明可执行步骤和验收标准；\n"
            "6. 推荐学习顺序：明确先后依赖；\n"
            "7. 证据覆盖与不足：先列出本次实际覆盖的章节或主题，"
            "再列出与学习目标直接相关、但当前证据无法支持的内容。\n\n"
            
            "必须严格区分以下三种情况：\n"
            "1. 当前检索结果未覆盖某部分正文：只能表述为"
            "“本轮检索证据未覆盖”，不得断言原始资料中不存在；\n"
            "2. 资料明确说明内容位于外部仓库、外部论文或其他资源："
            "应表述为“当前本地语料未包含该外部资源”；\n"
            "3. 资料明确声明不覆盖某项内容：只有这种情况下，"
            "才能表述为“资料本身未包含”。\n\n"
            
            "不得主动引入学习目标、目录和检索证据均未要求的领域，"
            "例如金融、医疗等垂直行业，并把它们列为“证据不足”。\n\n"
            
            "每一个核心模块、实现说明、练习和学习顺序判断都必须标注来源。\n"
            "引用格式必须根据文件类型和读取方式选择：\n"
            "- PDF向量证据使用 [source=..., page=..., chunk=...]；\n"
            "- MD、TXT向量证据使用 [source=..., chunk=...]；\n"
            "- Python源码为全量读取，只使用 [source=...]，"
            "不得编造 page 或 chunk；\n"
            "- 来自确定性资料概览的内容可以只标注 source，"
            "PDF概览还可以标注 page。\n"
            "不得用 PDF 目录、前言、README 概览或代码文件顶部摘要，"
            "替代正文、函数实现或具体技术细节的引用；\n"
            "不得合并为 page=6-7 这样的范围，PDF 必须引用实际获得的单页。"
        )),
        HumanMessage(content=f"学习目标：{learning_goal}\n\n{context}"),
    ]
    logger.debug(
        "资料分析请求统计：messages=%s",
        [
            {
                "type": message.type,
                "chars": len(str(message.content)),
            }
            for message in messages
        ],
    )
    content = _invoke_llm(llm, messages, stage="资料分析智能体")
    logger.info("资料分析完成：response_chars=%d", len(content))
    log_citation_summary("资料分析智能体", content,)
    return content, context

def analyse_materials_agentic(material_dir:str, learning_goal:str) -> tuple[str, str]:
    """资料分析智能体（Agentic 版本）：让模型自行决定调用哪些工具。

    返回 (分析结果, 资料上下文)：资料上下文是本轮所有工具调用及其结果按顺序拼接
    而成的文本，供后续多轮对话直接复用，避免模型在 interactive 模式下重复调用
    list_learning_files / read_learning_file 等工具。
    """
    logger.info(
        "【资料分析智能体（Agentic）】开始，material_dir=%s, goal=%s",
        material_dir,
        learning_goal,
    )

    # 工具在本轮运行时绑定material_dir，模型不再负责生成root_dir。
    agent_tools = build_agent_tools(material_dir)
    tool_map = {
        tool.name: tool
        for tool in agent_tools
    }

    llm = create_llm_with_tools(agent_tools)
    messages = [
        SystemMessage(content=(
            "你是资料分析智能体，可以使用工具查看资料目录。"
            f"资料目录固定为：{material_dir}，调用工具时请使用这个路径作为 root_dir。"
            "先调用 list_learning_files 理解语料范围。"
            "对于大文件、PDF或跨文件概念，调用retrieve_learning_context。"
            "该工具的queries参数可以一次接收多条查询；"
            "每轮调用前应先汇总当前所有资料缺口，合并近义query，"
            "不要为每一个关键词单独调用一次工具。"
            "neighbor_window=0表示只取命中块，"
            "neighbor_window=1表示读取命中块前后各一个chunk。"
            "只有需要查看某个小文件原文时才调用read_learning_file。"
            "最终结论的每个关键判断都要标注工具结果里的 source/page/chunk，"
            "没有证据就明确说资料不足，不要编造。"
            "确认信息足够后，直接输出最终分析结果，不要再调用工具。"
        )),
        HumanMessage(content=f"学习目标：{learning_goal}"),
    ]
    tool_trace: list[str] = []            # 累计工具调用的输入输出，作为可复用的资料上下文（已去重）
    seen_signatures: set[tuple] = set()   # 已记录过的 (工具名, 参数) 组合，用于给 tool_trace 去重
    call_signatures: list[tuple] = []     # 按时间顺序记录每次调用的签名，用于检测"连续3次相同调用"
    for round_idx in range(MAX_TOOL_ROUNDS):
        response = _invoke_llm_raw(llm, messages, stage=f"资料分析-第{round_idx+1}轮")
        messages.append(response)
        if not response.tool_calls:
            content = str(response.content)

            logger.info(
                "Agentic 资料分析完成：rounds=%d "
                "response_chars=%d",
                round_idx + 1,
                len(content),
            )

            log_citation_summary(
                "Agentic 资料分析智能体",
                content,
            )

            return content, "\n".join(tool_trace)
        stuck_in_loop = False
        for call in response.tool_calls:
            logger.info(
                "Agent 工具调用：round=%d tool=%s",
                round_idx + 1,
                call["name"],
            )

            logger.debug(
                "Agent 工具参数：round=%d tool=%s args=%s",
                round_idx + 1,
                call["name"],
                {
                    key: (
                        f"<{len(value)} chars>"
                        if isinstance(value, str) and len(value) > 100
                        else value
                    )
                    for key, value in call["args"].items()
                },
            )
            signature = (
                call["name"],
                json.dumps(call["args"], ensure_ascii=False, sort_keys=True, default=str),
            )
            call_signatures.append(signature)
            tool = tool_map.get(call["name"])
            if tool is None:
                messages.append(
                    ToolMessage(
                        content=(
                            f"没有名为 {call['name']} 的工具，"
                            f"可选工具：{', '.join(tool_map)}"
                        ),
                        tool_call_id=call["id"],
                    )
                )
            else:
                # 这里需要做异常保护，如遇到工具内部抛异常，IO错误等
                try:
                    result = tool.invoke(call["args"])
                except Exception as exc:
                    result = f"工具执行出错：{exc}"
                    logger.warning("工具 %s 执行异常：%s", call["name"], exc)
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
                if signature not in seen_signatures:
                    tool_trace.append(f"--- 调用 {call['name']}({call['args']}) ---\n{result}")
                    seen_signatures.add(signature)
            if len(call_signatures) >= 3 and call_signatures[-1] == call_signatures[-2] == call_signatures[-3]:
                logger.warning("检测到连续 3 次完全相同的工具调用 %s，提前终止以避免死循环", signature)
                stuck_in_loop = True
                break
        if stuck_in_loop:
            return (
                "检测到模型反复以相同参数调用同一工具，已提前终止分析；"
                "请检查资料目录内容，或简化学习目标后重试。",
                "\n".join(tool_trace),
            )
    logger.warning("达到最大工具调用轮数，强制结束")
    return "多次调用工具后仍未生成结论，请检查资料目录或简化学习目标", "\n".join(tool_trace)

def _limit_evidence_context(
    context: str,
    max_chars: int = MAX_PLAN_EVIDENCE_CHARS,
) -> str:
    """按完整证据快截断，避免从 chunk 中间直接切断"""
    separator = "\n\n---\n\n"
    blocks = context.split(separator)

    selected: list[str] = []
    current_length = 0
    for block in blocks:
        next_length = current_length + len(block) + len(separator)
        if next_length > max_chars:
            break
        selected.append(block)
        current_length = next_length
    return separator.join(selected)


def create_study_plan(
    analysis: str,
    material_context: str,
    learning_goal: str,
    days: int,
    *,
    material_dir: str,
    mode: str,
) -> str:
    """计划智能体：根据第一位智能体的结论制作可执行计划。

    根据运行模式限制规划智能体的补充检索次数。

    pipeline模式：
    - 已经通过确定性流程获得初始证据；
    - 规划模型最多额外调用一次检索工具；
    - 一次调用可以提交多条query；
    - 如果初始证据充分，直接输出计划，不调用工具。

    agent模式：
    - 规划模型可根据每轮工具结果继续判断证据缺口；
    - 最多额外调用6次，实际值由配置控制；
    - 达到上限后切换为无工具模型，强制生成最终计划。

    动态评估：
    - 记录工具调用次数、每次query数量和新增证据字符数；
    - pipeline经常使用满1次，说明初始四类query覆盖可能不足；
    - agent经常接近6次，说明query过碎或停止条件不清楚；
    - 补充检索增加很多token但计划不变时，应降低调用上限。
    """
    settings = get_settings()
    plan_evidence = _limit_evidence_context(material_context)

    retrieval_available = needs_rag(material_dir)

    # pipeline只允许一次额外检索；
    # agent允许多轮观察工具结果后继续决定，但初始上限为6。
    max_retrieval_calls = (
        settings.plan_agent_max_retrieval_calls
        if mode == "agent"
        else 1
    )

    retrieval_tool = (
        build_retrieve_learning_context_tool(material_dir)
        if retrieval_available
        else None
    )

    retrieval_rules = (
        "你可以使用retrieve_learning_context补充检索资料。\n"
        "调用规则：\n"
        f"1. 本次最多允许调用{max_retrieval_calls}次；\n"
        f"2. 工具的queries一次最多提交"
        f"{settings.plan_max_queries_per_tool_call}条query；"
        "必须按对学习计划的影响程度从高到低排列；\n"
        "3. 每次调用前，必须把当前已发现的重要证据缺口集中到"
        "同一个queries列表，合并近义问题；"
        "如果缺口数量超过上限，只保留最可能改变学习顺序、"
        "任务内容或验收标准的查询；\n"
        "4. query必须描述需要从资料中找到的具体信息，"
        "不能写成“如何制定计划”或“寻找更多内容”；\n"
        "5. neighbor_window只能为0或1："
        "概念定义可优先使用0，需要理解连续论述时使用1；\n"
        "6. 不得重复已经调用过的query；\n"
        "7. 如果当前证据已经能够支持完整计划，"
        "不要调用工具，直接输出最终学习计划；\n"
        "8. 工具没有命中只代表本次query没有召回，"
        "不能断言原始资料中不存在该内容。\n"
        if retrieval_tool is not None
        else
        "当前语料不启用向量检索工具。"
        "请依据现有分析和证据制定计划；证据不足时明确说明，不得编造。\n"
    )

    messages = [
        SystemMessage(content=(
            "你是学习路径设计智能体，需要把资料中的知识结构"
            "转化为适合的学习计划。\n"
            "你将同时获得资料分析结果和带来源的原始证据。\n\n"

            f"{retrieval_rules}\n"

            "最终计划必须遵守：\n"
            "1. 学习顺序必须体现前置依赖，由基础到进阶；\n"
            "2. 每天最多两个任务，每个任务必须包含学习内容、"
            "具体步骤和可验证产出；\n"
            "3. 每个任务必须引用至少一个真实证据位置："
            "PDF向量证据使用[source=..., page=..., chunk=...]；"
            "MD、TXT向量证据使用[source=..., chunk=...]；"
            "Python全量源码只使用[source=...]，"
            "不得编造页码或chunk编号；\n"
            "4. 不得把资料没有明确说明的字段、规范或参数"
            "描述成官方要求；\n"
            "5. 如果某项内容是根据资料做出的教学设计，"
            "必须标记为“计划设计”，不能伪装成资料原文；\n"
            "6. 内容不足时明确说明，不得编造；\n"
            "7. 最终回答只输出完整学习计划，"
            "不要输出内部判断过程或证据缺口分析。"
        )),
        HumanMessage(content=(
            f"学习目标：{learning_goal}\n"
            f"计划天数：{days}\n\n"
            f"资料分析结果：\n{analysis}\n\n"
            f"当前可引用的资料证据：\n{plan_evidence}"
        )),
    ]

    # 没有RAG语料时，不绑定无效的语义检索工具。
    if retrieval_tool is None:
        content = _invoke_llm(
            create_llm(),
            messages,
            stage="学习计划智能体",
        )
        logger.info(
            "学习计划生成完成：retrieval_available=false "
            "response_chars=%d",
            len(content),
        )
        log_citation_summary("学习计划智能体", content)
        return content

    tool_llm = create_llm_with_tools([retrieval_tool])

    used_retrieval_calls = 0
    seen_retrieval_signatures: set[str] = set()

    while True:
        response = _invoke_llm_raw(
            tool_llm,
            messages,
            stage=(
                "学习计划智能体"
                f"-检索判断{used_retrieval_calls + 1}"
            ),
        )
        messages.append(response)

        tool_calls = list(response.tool_calls or [])

        # 模型不调用工具，表示它判断当前证据已经足够。
        if not tool_calls:
            content = _content_to_text(response.content)

            logger.info(
                "学习计划生成完成：mode=%s retrieval_calls=%d "
                "response_chars=%d",
                mode,
                used_retrieval_calls,
                len(content),
            )
            log_citation_summary("学习计划智能体", content)
            return content

        for call in tool_calls:
            # 一条AIMessage里的每个tool_call都必须对应一个ToolMessage。
            # 超出预算的调用不会真正执行，但仍返回明确拒绝结果，
            # 保证消息序列满足Function Calling协议。
            if used_retrieval_calls >= max_retrieval_calls:
                messages.append(
                    ToolMessage(
                        content=(
                            "补充检索预算已用完。"
                            "请基于现有分析和全部工具结果输出最终计划，"
                            "不得继续请求工具。"
                        ),
                        tool_call_id=call["id"],
                    )
                )
                continue

            signature = json.dumps(
                call.get("args", {}),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

            # 重复调用也计入预算，避免模型通过重复参数无限占用轮次。
            used_retrieval_calls += 1

            if signature in seen_retrieval_signatures:
                result = (
                    "拒绝重复检索：相同参数已经调用过。"
                    "请利用已有结果，或在仍有预算时改写为不同的信息缺口。"
                )
            elif call["name"] != retrieval_tool.name:
                result = (
                    f"未知工具：{call['name']}。"
                    f"当前只允许使用{retrieval_tool.name}。"
                )
            else:
                seen_retrieval_signatures.add(signature)

                try:
                    result = retrieval_tool.invoke(
                        call.get("args", {})
                    )
                except Exception as exc:
                    # 工具失败作为观察结果回传给模型，
                    # 不让一次检索异常直接破坏整个学习计划流程。
                    result = f"补充检索执行失败：{exc}"
                    logger.warning(
                        "规划检索工具异常：mode=%s call=%d error=%s",
                        mode,
                        used_retrieval_calls,
                        exc,
                    )

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"],
                )
            )

            logger.info(
                "规划补充检索：mode=%s used=%d limit=%d "
                "args=%s result_chars=%d",
                mode,
                used_retrieval_calls,
                max_retrieval_calls,
                call.get("args", {}),
                len(str(result)),
            )

        if used_retrieval_calls >= max_retrieval_calls:
            # 达到上限后改用没有绑定工具的模型。
            # 这是程序级终止条件，不只依赖提示词自觉停止。
            messages.append(
                HumanMessage(content=(
                    "补充检索次数已经达到上限。"
                    "请综合初始证据和所有工具返回结果，"
                    "现在直接输出最终完整学习计划。"
                    "不要再请求工具，也不要描述内部判断过程。"
                ))
            )

            content = _invoke_llm(
                create_llm(),
                messages,
                stage="学习计划智能体-强制收束",
            )

            logger.info(
                "学习计划强制收束完成：mode=%s retrieval_calls=%d "
                "response_chars=%d",
                mode,
                used_retrieval_calls,
                len(content),
            )
            log_citation_summary("学习计划智能体", content)
            return content

def _validate_material_dir(material_dir:str) -> str | None:
    """校验资料目录是否存在且包含可分析文件。

    合法时返回 None；目录不存在，或路径模糊/无法定位到有效资料时，返回一段
    面向用户的指导文字——调用方应据此在第一步直接终止流程，不再进入学习计划生成。
    """
    root = Path(material_dir)
    if not root.is_dir():
        return (
            f"未找到资料目录：{material_dir}.\n"
            "请确认路径是否正确（相对路径以当前运行目录为准），"
            "或使用 --materials 参数指定一个真实存在的目录后重试。"
        )
    file_list = list_learning_files.invoke({"root_dir": material_dir})
    if file_list.strip() == "没有找到可分析的资料文件。" or file_list.startswith("目录不存在"):
        return (
            f"目录 {material_dir} 中没有找到可分析的资料文件（支持 .txt/.md/.py/.pdf）。"
            "这个路径可能不是你想要分析的资料目录，请换一个更明确，且包含实际资料的路径后重试。"
        )
    return None

def run_learning_assistant(material_dir: str, learning_goal: str, days: int, mode: str) -> tuple[str, str, str]:
    """按顺序运行两个智能体，并返回“分析结果、学习计划、资料上下文”。

    若资料目录不存在或无法定位到有效资料，会在第一步（资料分析）就终止：
    返回的 analysis 是给用户的指导文字，plan 与 material_context 为空字符串，
    不会再调用 create_study_plan 进入计划生成。
    """
    guidance = _validate_material_dir(material_dir)
    if guidance is not None:
        logger.warning("资料目录校验未通过，提前终止：%s", guidance)
        return guidance, "", ""

    if mode == "agent":
        analysis, material_context = analyse_materials_agentic(
            material_dir,
            learning_goal,
        )
    else:
        analysis, material_context = analyse_materials(
            material_dir,
            learning_goal,
        )

    # 两种模式共用同一个规划入口和同一个多query检索工具，
    # 区别仅在程序允许的额外检索次数：
    # pipeline固定为1次，agent默认最多6次。
    plan = create_study_plan(
        analysis,
        material_context,
        learning_goal,
        days,
        material_dir=material_dir,
        mode=mode,
    )
    return analysis, plan, material_context
