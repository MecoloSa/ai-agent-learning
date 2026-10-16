"""统一终端日志展示；不改变模型、工具、审计记录或业务状态。"""
from __future__ import annotations

import logging


def configure_console_logging(*, verbose: bool = False, log_level: str = "INFO") -> None:
    # 正常实验只看阶段与摘要；-v或LOG_LEVEL=DEBUG恢复决策细节。
    # 不强制重建已有handler，避免覆盖IDE/测试调用方自己的日志配置。
    level = logging.DEBUG if verbose else getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpcore", "urllib3", "chromadb", "openai", "dashscope"):
        logging.getLogger(name).setLevel(logging.WARNING)
