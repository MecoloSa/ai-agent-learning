"""集中管理模型配置，避免把密钥写进代码。"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
from pydantic import SecretStr

from .embeddings import BatchedDashScopeEmbeddings

# 运行产物应属于整个项目，而不是取决于终端/IDE 当时的工作目录。
# config.py 位于 <project>/src/ai_agent，因此 parents[2] 是 pyproject.toml 所在目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 优先加载本包目录下的 .env（无论从哪个 CWD 启动都能读到），找不到再回退系统环境变量。
load_dotenv(Path(__file__).resolve().parent / ".env")
logger = logging.getLogger("ai_agent.config")


def resolve_project_path(value: str | Path) -> Path:
    """把相对运行路径固定解析到项目根目录，绝对路径保持不变。

    动态评估：可从不同工作目录启动同一命令，日志中的最终路径应保持一致；
    如果项目目录整体移动，PROJECT_ROOT 会随代码位置自动变化。
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()

def _required_env(name: str) -> str:
    """读取必填环境变量，缺失时给出明确错误。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}，请检查项目根目录下的 .env。")
    return value

def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    value = int(os.getenv(name, str(default)))
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} 必须大于等于 {minimum}，当前值为 {value}")
    return value


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
        raise RuntimeError(f"{name} 不是合法布尔值：{value}")
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # 原有模型字段保留
    chat_api_key: str
    embedding_api_key: str
    chat_model: str
    chat_base_url: str
    embedding_model: str
    embedding_batch_size: int
    request_timeout: float

    llm_temperature: float
    llm_retry_attempts: int
    llm_retry_min_wait: float
    llm_retry_max_wait: float
    embedding_max_retries: int

    agent_max_tool_rounds: int
    plan_max_evidence_chars: int

    # Agent模式下，规划智能体最多允许额外调用几次语义检索工具。
    # pipeline模式固定为1次，因此不需要再设置单独环境变量。
    plan_agent_max_retrieval_calls: int
    # 每次工具调用允许同时提交的query数量。
    # 让模型集中表达当前证据缺口，减少“一条query调用一次”的低效行为。
    plan_max_queries_per_tool_call: int

    material_snippet_file_limit: int
    material_snippet_max_chars: int

    rag_data_dir: str
    rag_large_file_bytes: int
    rag_large_corpus_bytes: int
    rag_chunk_size: int
    rag_chunk_overlap: int
    rag_overview_max_pdf_pages: int
    rag_overview_max_chars: int
    rag_retrieval_k: int
    rag_min_relevance: float
    rag_relative_margin: float
    rag_neighbor_window: int
    rag_max_context_chars: int
    rag_min_hits_per_query: int

    tool_max_chars_per_file: int
    tool_file_list_limit: int
    tool_search_result_limit: int

    memory_summary_trigger_chars: int
    memory_window_size: int
    log_level: str
    log_payloads: bool
    log_preview_chars: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings(
        chat_api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
        embedding_api_key=os.getenv("DASHSCOPE_API_KEY_2", "").strip(),
        chat_model=os.getenv("CHAT_MODEL", "qwen3.7-plus").strip(),
        chat_base_url=os.getenv("CHAT_BASE_URL", "").strip(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "qwen3.7-text-embedding"
        ).strip(),
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 20, minimum=1),
        request_timeout=_env_float("REQUEST_TIMEOUT", 180),

        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        llm_retry_attempts=_env_int("LLM_RETRY_ATTEMPTS", 3, minimum=1),
        llm_retry_min_wait=_env_float("LLM_RETRY_MIN_WAIT", 2),
        llm_retry_max_wait=_env_float("LLM_RETRY_MAX_WAIT", 8),
        embedding_max_retries=_env_int("EMBEDDING_MAX_RETRIES", 3, minimum=0),

        agent_max_tool_rounds=_env_int("AGENT_MAX_TOOL_ROUNDS", 12, minimum=1),
        plan_max_evidence_chars=_env_int(
            "PLAN_MAX_EVIDENCE_CHARS", 90_000, minimum=1
        ),
        plan_agent_max_retrieval_calls=_env_int(
            "PLAN_AGENT_MAX_RETRIEVAL_CALLS",
            4,
            minimum=1,
        ),
        plan_max_queries_per_tool_call=_env_int(
            "PLAN_MAX_QUERIES_PER_TOOL_CALL",
            6,
            minimum=1,
        ),
        material_snippet_file_limit=_env_int(
            "MATERIAL_SNIPPET_FILE_LIMIT", 3, minimum=0
        ),
        material_snippet_max_chars=_env_int(
            "MATERIAL_SNIPPET_MAX_CHARS", 1_500, minimum=1
        ),

        rag_data_dir=str(
            resolve_project_path(
                os.getenv("RAG_DATA_DIR", ".agents_data/rag")
            )
        ),
        rag_large_file_bytes=_env_int(
            "RAG_LARGE_FILE_BYTES", 51_200, minimum=1
        ),
        rag_large_corpus_bytes=_env_int(
            "RAG_LARGE_CORPUS_BYTES", 204_800, minimum=1
        ),
        rag_chunk_size=_env_int(
            "RAG_CHUNK_SIZE", 900, minimum=2
        ),
        rag_chunk_overlap=_env_int(
            "RAG_CHUNK_OVERLAP", 150, minimum=0
        ),
        rag_overview_max_pdf_pages=_env_int(
            "RAG_OVERVIEW_MAX_PDF_PAGES", 10, minimum=1
        ),
        rag_overview_max_chars=_env_int(
            "RAG_OVERVIEW_MAX_CHARS", 30_000, minimum=1
        ),
        rag_retrieval_k=_env_int("RAG_RETRIEVAL_K", 8, minimum=1),
        rag_min_relevance=_env_float("RAG_MIN_RELEVANCE", 0.35),
        rag_relative_margin=_env_float("RAG_RELATIVE_MARGIN", 0.12),
        rag_neighbor_window=_env_int("RAG_NEIGHBOR_WINDOW", 2, minimum=0),
        rag_max_context_chars=_env_int(
            "RAG_MAX_CONTEXT_CHARS", 72_000, minimum=1
        ),
        rag_min_hits_per_query=_env_int(
            "RAG_MIN_HITS_PER_QUERY",
            3,
            minimum=1,
        ),

        tool_max_chars_per_file=_env_int(
            "TOOL_MAX_CHARS_PER_FILE", 5_000, minimum=1
        ),
        tool_file_list_limit=_env_int("TOOL_FILE_LIST_LIMIT", 50, minimum=1),
        tool_search_result_limit=_env_int(
            "TOOL_SEARCH_RESULT_LIMIT", 12, minimum=1
        ),

        memory_summary_trigger_chars=_env_int(
            "MEMORY_SUMMARY_TRIGGER_CHARS", 2_000, minimum=1
        ),
        memory_window_size=_env_int("MEMORY_WINDOW_SIZE", 6, minimum=1),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_payloads=_env_bool("LOG_PAYLOADS", False),
        log_preview_chars=_env_int(
            "LOG_PREVIEW_CHARS",
            800,
            minimum=100,
        ),
    )

    if settings.embedding_batch_size > 20:
        raise RuntimeError("EMBEDDING_BATCH_SIZE 不能超过 DashScope 上限 20")
    if not 0 <= settings.rag_min_relevance <= 1:
        raise RuntimeError("RAG_MIN_RELEVANCE 必须位于 0 到 1 之间")
    if settings.rag_chunk_overlap >= settings.rag_chunk_size:
        raise RuntimeError(
            "RAG_CHUNK_OVERLAP 必须小于 RAG_CHUNK_SIZE"
        )

    return settings

def create_embeddings() -> Embeddings:
    """创建支持自动分批的 DashScope Embedding 客户端。"""
    settings = get_settings()
    logger.debug(
        "创建 DashScope Embedding：model=%s, batch_size=%d",
        settings.embedding_model,
        settings.embedding_batch_size,
    )
    return BatchedDashScopeEmbeddings(
        model=settings.embedding_model,
        api_key=_required_env("DASHSCOPE_API_KEY_2"),
        batch_size=settings.embedding_batch_size,
        max_retries=settings.embedding_max_retries,
    )

def create_llm() -> ChatOpenAI:
    """创建聊天模型。"""
    settings = get_settings()

    logger.debug(
        "创建 LLM：model=%s base_url=%s "
        "temperature=%.3f timeout=%.1f "
        "outer_retries=%d streaming=%s",
        settings.chat_model,
        settings.chat_base_url,
        settings.llm_temperature,
        settings.request_timeout,
        settings.llm_retry_attempts,
        False,
    )

    return ChatOpenAI(
        model=settings.chat_model,
        api_key=SecretStr(_required_env("DASHSCOPE_API_KEY")),
        base_url=_required_env("CHAT_BASE_URL"),
        streaming=False,
        temperature=settings.llm_temperature,
        timeout=settings.request_timeout,
        max_retries=0,  # 仍由 tenacity 统一重试
    )

def create_llm_with_tools(tools: Sequence[BaseTool]):
    """由调用方注入工具，避免 config 反向依赖 tools。"""
    tool_list = list(tools)

    logger.debug(
        "绑定模型工具：count=%d tools=%s",
        len(tool_list),
        [tool.name for tool in tool_list],
    )

    return create_llm().bind_tools(tool_list)
