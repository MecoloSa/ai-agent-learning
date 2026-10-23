"""两个职责清晰的智能体，以及它们之间的协作流程。
plan_evidence_trace仅用于本地实验审计，不影响原有返回值。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_exponential, before_log, before_sleep_log
)

from ai_agent.config import create_llm, create_llm_with_tools, get_settings
from ai_agent.resources.tools import (
    read_learning_file,
    list_learning_files,
    summarize_text_statistics,
    build_agent_tools,
    build_retrieve_learning_context_tool,
)
from ai_agent.retrieval.rag import LearningRAG, needs_rag
from ai_agent.runtime.plan_validation import finalize_plan_output
from ai_agent.runtime.prompt_rules import EVIDENCE_RULES
# 用于识别当前语料实际包含哪些文件类型，这里只读取文件路径。
from ai_agent.resources.materials import iter_material_files, read_material_excerpt

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

def _get_model_name(llm) -> str:
    """识别本次调用模型，包括绑定工具后的包装对象。"""
    # bind_tools() 的返回对象可能把实际模型保存在 bound 属性中。
    model = getattr(llm, "bound", llm)
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or "unknown"
    )

def _log_model_request(
    stage: str,
    messages,
    *,
    model_name: str = "unknown",
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
        model_name,
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
    *,
    model_name: str = "unknown",
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
        or model_name,
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

# 原来的 TOOL_MAP 为全局，现在的工具绑定具体的资料目录
settings = get_settings()
MAX_TOOL_ROUNDS = settings.agent_max_tool_rounds
MAX_PLAN_EVIDENCE_CHARS = settings.plan_max_evidence_chars

# 仅适用于一次资料分析任务，不限制整个交互会话。
# 动态评估：正常任务若频繁触发某工具上限，应检查是否确有证据缺口，
# 再局部调整额度；不要直接提高所有工具的预算。
ANALYSIS_TOOL_LIMITS: dict[str, int] = {
    "list_learning_files": 2,
    "read_learning_file": 6,
    "search_learning_files": 4,
    "retrieve_learning_context": 3,
    "summarize_text_statistics": 2,
}

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

class AnalysisExecutionError(RuntimeError):
    """程序确认分析未完成；调用方必须停止后续计划生成。"""

    def __init__(self, reason: str, message: str):
        # reason 是稳定的程序原因代码，便于日志、评测及后续状态图使用。
        self.reason = reason
        super().__init__(f"{reason}: {message}")


def _require_analysis_text(value: object) -> str:
    """分析阶段必须返回非空文本，不能把空值强制转成字符串冒充结论。"""
    if not isinstance(value, str) or not value.strip():
        raise AnalysisExecutionError(
            "invalid_analysis_output",
            "资料分析未返回非空文本；未生成学习计划。",
        )
    return value.strip()

def _build_analysis_budget_status(
    tool_names: list[str],
    tool_counts: dict[str, int],
    executed_calls: int,
    total_limit: int,
    remaining_model_rounds: int,
) -> dict[str, object]:
    """从程序真实计数生成模型可见预算，不让模型自行记账。"""
    remaining_total = max(0, total_limit - executed_calls)

    # 单工具实际可用次数同时受自身额度与总额度约束。
    # 只展示本次绑定的工具，避免暗示其他工具也可以调用。
    remaining_by_tool = {
        name: min(
            remaining_total,
            max(
                0,
                ANALYSIS_TOOL_LIMITS.get(name, 4)
                - tool_counts.get(name, 0),
            ),
        )
        for name in tool_names
    }

    notices = [
        "额度是上限，不必用满；批量请求中的每个执行分别计数。",
        "同参请求仍会被拒绝；剩余额度不代表允许重复执行。",
    ]

    if remaining_total == 0:
        notices.append(
            "工具总额度已用完，不要再请求工具；"
            "请依据已有证据回答，明确不足之处。"
        )
    elif remaining_total == 1:
        notices.append(
            "所有工具合计最多还可执行1次，"
            "请优先补足最关键的证据缺口。"
        )
    else:
        last_chance_tools = [
            name
            for name, count in remaining_by_tool.items()
            if count == 1
        ]
        if last_chance_tools:
            notices.append(
                "以下工具各自最多还可执行1次："
                + "、".join(last_chance_tools)
                + "。请优先处理必要的证据缺口。"
            )

    # 工具额度未耗尽，也可能先达到模型轮数上限。
    # 明确告知：最后一轮调用工具后，不会额外赠送一次模型收尾。
    if remaining_model_rounds == 1:
        notices.append(
            "本轮是最后一次模型决策；调用工具后没有下一轮生成分析。"
            "请依据已有证据作答，并明确不足之处。"
        )

    return {
        "remaining_total": remaining_total,
        "remaining_by_tool": remaining_by_tool,
        "remaining_model_rounds_including_current": remaining_model_rounds,
        "notice": " ".join(notices),
    }

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
        model_name = _get_model_name(llm)
        _log_model_request(
            stage,
            messages,
            model_name=model_name,
        )

        start = time.perf_counter()
        response = llm.invoke(messages)
        elapsed = time.perf_counter() - start

        _log_model_response(
            stage,
            response,
            model_name=model_name,
        )
        usage = getattr(
            response,
            "usage_metadata",
            None,
        )
        logger.info(
            "【%s】模型调用完成：model=%s 耗时=%.1f秒 tokens=%s",
            stage,
            model_name,
            elapsed,
            (
                dict(usage).get("total_tokens", "未知")
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

    logger.debug(
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
                data = read_material_excerpt(
                    material_dir,
                    str(relative_path),
                    settings.material_snippet_max_chars,
                )
                snippets.append(json.dumps(
                    {"ok": True, "data": data},
                    ensure_ascii=False,
                ))
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
        logger.debug("资料目录中没有Python源码。")
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

    logger.debug(
        "Python源码已全部完整读取：files=%d chars=%d",
        len(python_paths),
        total_chars,
    )

    return "\n\n---\n\n".join(blocks)

# 两种资料分析模式与计划生成共用规则，避免各阶段采用不同的学习标准。
# 动态评估：观察是否强加实践，以及明确要求实践时是否遗漏具体行动。
LEARNING_ACTIVITY_RULES = (
    "按用户目标和资料内容选择活动，不按文件后缀推断学习主题。"
    "阅读理解不强制代码或实践；明确要求实践时给出适合该领域的具体行动。"
    "完成标志须与目标匹配，不一律要求作品或测试。"
    "自行设计的活动标为“计划设计”，不冒充资料原有内容。"
    "预期结果不是已完成事实，已阅读或自报理解不等于客观掌握。\n"
)

def _build_broad_retrieval_queries(
    material_dir: str,
    learning_goal: str,
) -> tuple[tuple[str, str], ...]:
    """根据学习目标构造四个检索方向，不根据后缀推断学习主题。

    material_dir 保留用于兼容现有调用；文件读取与格式处理由上游负责。
    动态评估：比较目标相关证据覆盖、重复命中和补充检索次数。
    查询更通用不一定召回更好，技术资料也要保留对照案例。
    """
    goal = " ".join(learning_goal.split())

    # 保留原查询标签和四路检索流程，避免同时改变检索编排。
    # 标签仅用于区分查询，不代表用户必须学习实现或进行实践。
    return (
        (
            "goal",
            f"{goal}；直接相关的核心内容、关键概念与主要观点",
        ),
        (
            "organization",
            f"{goal}；知识结构、内容之间的关系与前置知识",
        ),
        (
            "implementation",
            f"{goal}；正文中的具体解释、例子、区别与适用条件",
        ),
        (
            "usage_validation",
            f"{goal}；资料中相关的学习指引、应用示例、练习或小结",
        ),
    )

def analyse_materials(material_dir: str, learning_goal: str) -> tuple[str, str]:
    """资料分析智能体：把原始资料压缩为学习重点。

    返回 (分析结果, 资料上下文)。资料上下文就是 collect_material_context
    收集到的文件清单/摘录/统计，供 interactive 模式直接复用，避免重复读文件。
    """
    logger.info(
        "开始资料分析：mode=pipeline",
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
        # 四个方向：目标内容、知识关系、具体解释、学习辅助依据。
        # 文件格式只影响读取方式，不决定学习主题或强制实践。
        # 动态评估：同时比较非技术资料和技术资料，避免改善前者却损害后者。
        retrieval_queries = _build_broad_retrieval_queries(
            material_dir=material_dir,
            learning_goal=learning_goal,
        )

        for query_id, _ in retrieval_queries:
            logger.debug(
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

        logger.debug(
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
            "你是资料分析智能体，只分析与学习目标相关的内容，不生成按天或按周的完整计划。\n"
            f"{LEARNING_ACTIVITY_RULES}{EVIDENCE_RULES}"
            "输出结构：\n"
            "1. 资料定位：主题、适用读者与覆盖范围；未说明的内容不要猜测。\n"
            "2. 知识结构：依据资料中的关系组织，不强套固定层级。\n"
            "3. 前置依赖：仅列出资料明确支持的先后关系，没有则说明。\n"
            "4. 核心内容：概念、作用和与目标相关的学习难点。\n"
            "5. 学习活动依据：区分资料原有活动与自行提出的学习建议。\n"
            "6. 推荐顺序：依据已有依赖和用户目标，不编造必须先学的关系。\n"
            "7. 证据覆盖与不足：区分本次未覆盖、本地缺少外部资源和资料明确不覆盖。\n"
            "缺口仅限与学习目标相关的内容，不为填满结构引入无关领域。"
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
        "开始资料分析：mode=agent",
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
            "你是资料分析智能体，通过工具获取与学习目标相关的证据。"
            "目录由程序绑定，不传root_dir或绝对路径；先用list_learning_files了解资料范围。"
            "小文件原文用read_learning_file，大文件、PDF或跨文件问题用retrieve_learning_context。"
            "查询只覆盖当前必要缺口，合并近义问题；互补且独立的查询可批量提交，"
            "依赖前次结果的查询分步决定。neighbor_window=0只取命中块，1补相邻块。"
            "证据充分时停止调用工具，输出资料定位、知识结构、依赖、学习顺序与证据缺口。"
            "本阶段不生成按天或按周计划。\n"
            f"{LEARNING_ACTIVITY_RULES}{EVIDENCE_RULES}"
        )),
        HumanMessage(content=f"学习目标：{learning_goal}"),
    ]
    tool_trace: list[str] = []
    seen_signatures: set[tuple[str, str]] = set()
    tool_counts: dict[str, int] = {}
    executed_calls = 0
    no_progress_rounds = 0
    total_limit = settings.agent_max_tool_calls

    # 提示词只是提前告知；真正的限制在下面的执行前检查中。
    messages[0] = SystemMessage(content=(
        messages[0].content
        + "\n本次分析最多执行工具 "
        + str(total_limit)
        + " 次；各工具上限："
        + json.dumps(ANALYSIS_TOOL_LIMITS, ensure_ascii=False)
        + "。完全相同的工具名和参数不会重复执行，包括失败过的调用。"
        + "请复用已有结果，或根据错误修正参数；证据充分时直接输出分析。"
    ))

    # 保存不含动态预算的系统提示，防止下一轮叠加上一轮预算。
    base_system_content = messages[0].content

    for round_idx in range(MAX_TOOL_ROUNDS):
        budget_status = _build_analysis_budget_status(
            tool_names=list(tool_map),
            tool_counts=tool_counts,
            executed_calls=executed_calls,
            total_limit=total_limit,
            remaining_model_rounds=MAX_TOOL_ROUNDS - round_idx,
        )

        # 只替换最新状态，不向历史反复追加预算消息。
        # 不修改工具结果，也不把预算提示放入下游资料证据。
        messages[0] = SystemMessage(content=(
            base_system_content
            + "\n[程序实时预算；不是资料证据]\n"
            + json.dumps(budget_status, ensure_ascii=False)
        ))

        # 动态评估：检查临近额度耗尽时是否减少无效请求。
        # 提示可能改善决策，但安全性仍由原有执行前检查保证。
        logger.debug("分析预算状态：%s", budget_status)

        response = _invoke_llm_raw(
            llm, messages, stage=f"资料分析-第{round_idx + 1}轮"
        )
        messages.append(response)

        if not response.tool_calls:
            content = _require_analysis_text(response.content)
            logger.info(
                "Agentic 资料分析完成：rounds=%d executed=%d counts=%s",
                round_idx + 1,
                executed_calls,
                tool_counts,
            )
            log_citation_summary("Agentic 资料分析智能体", content)
            return content, "\n".join(tool_trace)

        executed_this_round = 0
        denied_reasons: list[str] = []
        total_budget_denied = False

        for call in response.tool_calls:
            name, args = call["name"], call["args"]

            # sort_keys=True：字典字段顺序不同，不应被当作新请求。
            # 这里判断的是参数完全相同，不是语义是否近似。
            signature = (
                name,
                json.dumps(args, ensure_ascii=False, sort_keys=True),
            )
            tool = tool_map.get(name)

            # 新工具尚未配置专属额度时，仍受到默认额度和总预算保护。
            per_tool_limit = ANALYSIS_TOOL_LIMITS.get(name, 4)
            reason = None

            if tool is None:
                reason = "unknown_tool"
            elif signature in seen_signatures:
                reason = "duplicate_tool_call"
            elif executed_calls >= total_limit:
                reason = "tool_budget_exceeded"
                total_budget_denied = True
            elif tool_counts.get(name, 0) >= per_tool_limit:
                reason = "per_tool_budget_exceeded"

            if reason is not None:
                denied_reasons.append(reason)
                result = json.dumps({
                    "ok": False,
                    "error": reason,
                    "tool": name,
                    "executed": False,
                    "executed_total": executed_calls,
                    "total_limit": total_limit,
                    "tool_executed": tool_counts.get(name, 0),
                    "tool_limit": per_tool_limit,
                    "available_tools": list(tool_map),
                    "instruction": (
                        "复用已有结果、修正参数或使用其他工具；"
                        "证据充分时输出分析。"
                    ),
                }, ensure_ascii=False)

                logger.warning(
                    "分析工具拒绝：tool=%s reason=%s executed=%d/%d",
                    name, reason, executed_calls, total_limit,
                )
            else:
                # invoke 抛异常也消耗一次执行额度：失败同样有成本。
                # 在执行前登记签名，避免同参失败触发无意义的盲目重试。
                seen_signatures.add(signature)
                executed_calls += 1
                executed_this_round += 1
                tool_counts[name] = tool_counts.get(name, 0) + 1

                logger.debug(
                    "分析工具执行：tool=%s total=%d/%d tool_count=%d/%d",
                    name,
                    executed_calls,
                    total_limit,
                    tool_counts[name],
                    per_tool_limit,
                )

                try:
                    result = str(tool.invoke(args))
                except Exception as exc:
                    result = f"工具执行出错：{exc}"
                    logger.warning("工具 %s 执行异常：%s", name, exc)

                # 被拒绝的请求不是资料证据，不追加到下游资料上下文。
                tool_trace.append(
                    f"--- 调用 {name}({args}) ---\n{result}"
                )

            # 同批中被拒绝的调用也必须回传结果，不能漏掉 tool_call_id。
            messages.append(ToolMessage(
                content=result,
                tool_call_id=call["id"],
            ))

        # 处理完这一批的所有调用，再结束，保持调用与结果一一对应。
        if total_budget_denied:
            raise AnalysisExecutionError(
                "tool_budget_exceeded",
                "资料分析请求超过工具总执行预算；未生成学习计划。",
            )

        # 这里只判断有无实际执行，不等同于“获得了新知识”。
        # 动态评估：正常任务若频繁停止，应检查拒绝原因和工具上限。
        no_progress_rounds = (
            no_progress_rounds + 1
            if executed_this_round == 0
            else 0
        )

        if no_progress_rounds >= settings.agent_max_no_progress_rounds:
            reason = (
                "repeated_tool_call"
                if set(denied_reasons) == {"duplicate_tool_call"}
                else "tool_no_progress"
            )
            raise AnalysisExecutionError(
                reason,
                "资料分析连续多轮只有被拒绝的工具请求；未生成学习计划。",
            )

    logger.warning("达到最大模型轮数，停止资料分析")
    raise AnalysisExecutionError(
        "model_round_limit",
        "资料分析达到模型轮数上限，尚未生成结论；未生成学习计划。",
    )

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
    plan_evidence_trace: dict | None = None,
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
    logger.info("开始生成学习计划：mode=%s days=%d", mode, days)
    plan_evidence = _limit_evidence_context(material_context)

    if plan_evidence_trace is not None:
        # 保存真正放入模型输入的证据，不用未截断全文冒充实际输入。
        # 含资料正文，只写入本地实验结果，不打印到普通日志。
        plan_evidence_trace.update({
            "trace_version": "plan-evidence-v1",
            "initial_evidence": plan_evidence,
            "initial_context_chars": len(material_context),
            "initial_evidence_chars": len(plan_evidence),
            "tool_observations": [],
            "requests": [],
        })

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
        "需要补充证据时调用retrieve_learning_context。"
        f"最多调用{max_retrieval_calls}次，单次最多{settings.plan_max_queries_per_tool_call}条query。"
        "只查询可能改变计划的重要缺口，合并近义问题并按重要程度排序。"
        "互补且独立的缺口可批量提交；依赖前次结果的查询分步决定，不强制凑成同一批。"
        "query应明确所需资料信息，不能只要求生成计划；不重复已有查询。"
        "neighbor_window=0只取命中块，1补相邻块。证据充分就直接生成计划。\n"
        if retrieval_tool is not None else
        "当前未启用检索工具；依据已有证据制定计划，缺口如实说明。\n"
    )

    messages = [
        SystemMessage(content=(
            "你是学习计划智能体，依据用户目标、资料分析和原始证据制定计划。\n"
            f"{retrieval_rules}{LEARNING_ACTIVITY_RULES}{EVIDENCE_RULES}"
            "计划要求：\n"
            "1. 覆盖用户指定天数和限制；依据已证实的依赖组织顺序，"
            "没有明确依赖时按目标安排，不要编造“必须先学”的关系。\n"
            "2. 每天最多两个任务，每项写明学习内容、具体行动和与目标匹配的完成标志。\n"
            "3. 每个任务引用支持其资料内容的原始证据；教学安排标为计划设计，"
            "不将自行推导的内容描述成资料原文或官方要求。\n"
            "4. 证据不足时说明限制；最终仅输出完整计划及必要说明，不输出内部判断过程。"
        )),
        HumanMessage(content=(
            f"学习目标：{learning_goal}\n"
            f"计划天数：{days}\n\n"
            f"资料分析结果：\n{analysis}\n\n"
            f"当前可引用的资料证据：\n{plan_evidence}"
        )),
    ]

    def capture_evidence_before_request(stage: str) -> None:
        """记录本次准备提供的证据，不保存AI回复或私有思维链。"""
        if plan_evidence_trace is None:
            return

        observations = [
            {
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
            for message in messages
            if isinstance(message, ToolMessage)
        ]

        # 初始证据始终保留；工具观察包括成功结果、拒绝和失败反馈。
        # 同一轮的内容只保存一份，用数量标记每次请求看到的范围。
        plan_evidence_trace["tool_observations"] = observations
        plan_evidence_trace["requests"].append({
            "stage": stage,
            "state": "prepared_for_request",
            "tool_observation_count": len(observations),
        })

    # 没有RAG语料时，不绑定无效的语义检索工具。
    if retrieval_tool is None:
        capture_evidence_before_request("学习计划智能体")
        content = _invoke_llm(
            create_llm(),
            messages,
            stage="学习计划智能体",
        )
        content = finalize_plan_output(
            content,
            trace=plan_evidence_trace,
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
        stage = f"学习计划智能体-检索判断{used_retrieval_calls + 1}"
        capture_evidence_before_request(stage)

        response = _invoke_llm_raw(
            tool_llm,
            messages,
            stage=stage,
        )
        messages.append(response)

        tool_calls = list(response.tool_calls or [])

        # 模型不调用工具，表示它判断当前证据已经足够。
        if not tool_calls:
            content = _content_to_text(response.content)

            content = finalize_plan_output(
                content,
                trace=plan_evidence_trace,
            )
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

            logger.debug(
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

            capture_evidence_before_request("学习计划智能体-强制收束")
            content = _invoke_llm(
                create_llm(),
                messages,
                stage="学习计划智能体-强制收束",
            )

            content = finalize_plan_output(
                content,
                trace=plan_evidence_trace,
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

def run_learning_assistant(
    material_dir: str,
    learning_goal: str,
    days: int,
    mode: str,
    *,
    plan_evidence_trace: dict | None = None,
    orchestrator: str = "legacy",
    workflow_trace: dict | None = None,
) -> tuple[str, str, str]:
    """按顺序运行两个智能体，并返回“分析结果、学习计划、资料上下文”。

    若资料目录不存在或无法定位到有效资料，会在第一步（资料分析）就终止：
    返回的 analysis 是给用户的指导文字，plan 与 material_context 为空字符串，
    不会再调用 create_study_plan 进入计划生成。
    """
    if orchestrator not in {"legacy", "langgraph"}:
        raise ValueError(
            "orchestrator 必须为 legacy 或 langgraph"
        )

    if orchestrator == "langgraph":
        # 延迟导入：旧流程无需加载图模块，也避免初始化时循环导入。
        from ai_agent.runtime.orchestration import run_learning_graph

        return run_learning_graph(
            material_dir,
            learning_goal,
            days,
            mode,
            plan_evidence_trace=plan_evidence_trace,
            workflow_trace=workflow_trace,
        )

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

    # 同时保护 pipeline 和 agent，以及未来新增的分析实现。
    # 动态评估：分析无效时，create_study_plan 的调用次数必须为 0。
    analysis = _require_analysis_text(analysis)

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
        plan_evidence_trace=plan_evidence_trace,
    )
    return analysis, plan, material_context
