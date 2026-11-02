"""固定历史的记忆策略诊断；默认预检，不调用模型。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from typing import Annotated, Literal
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

import ai_agent.config as config_module
import ai_agent.persistence.memory as memory_module
from ai_agent.config import create_llm, get_settings
from ai_agent.persistence.memory import ConversationMemory

Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]


class FrozenTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user: Text
    assistant: Text


class MemoryProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: Text
    system_prompt: Text
    history: list[FrozenTurn] = Field(min_length=2)
    question: Text
    criteria: list[Text] = Field(min_length=1)


def load_probe(path: Path) -> MemoryProbe:
    return MemoryProbe.model_validate_json(path.read_text(encoding="utf-8-sig"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(case_path: Path, strategy: str, window_size: int, trigger_chars: int) -> dict:
    settings = get_settings()
    return {
        "schema_version": "memory-probe-v1",
        "strategy": strategy,
        "window_size": window_size,
        "trigger_chars": trigger_chars,
        "summary_max_chars": settings.memory_summary_max_chars,
        "chat_model": settings.chat_model,
        "llm_temperature": settings.llm_temperature,
        "request_timeout": settings.request_timeout,
        "llm_retry_attempts": settings.llm_retry_attempts,
        "llm_retry_min_wait": settings.llm_retry_min_wait,
        "llm_retry_max_wait": settings.llm_retry_max_wait,
        "chat_endpoint_sha256": hashlib.sha256(settings.chat_base_url.encode()).hexdigest(),
        "case_sha256": _hash(case_path),
        # 明确记录相关源码范围，不冒充完整依赖或服务端模型快照。
        "source_sha256": {
            "probe": _hash(Path(__file__)),
            "memory": _hash(Path(memory_module.__file__)),
            "config": _hash(Path(config_module.__file__)),
        },
    }


def run_probe(case: MemoryProbe, strategy: str, window_size: int, trigger_chars: int) -> dict:
    if strategy not in {"full", "window", "summary"} or window_size < 1 or trigger_chars < 1:
        raise ValueError("策略或实验参数无效")
    memory = ConversationMemory(
        system_prompt=case.system_prompt, strategy=strategy,
        window_size=window_size, max_chars=trigger_chars,
    )
    record = {
        "case_id": case.case_id,
        "execution_mode": "isolated_memory_probe",
        "status": "running", "error_type": None,
        "history_source": "fixed_fixture_not_live_responses",
        "source_history": [turn.model_dump() for turn in case.history],
        "question": case.question, "criteria": case.criteria,
        "assessment": "not_evaluated",
        "summary_calls": [], "memory_steps": [],
        "final_request_messages": [], "response": None, "final_usage": None,
        "final_latency_ms": None,
    }
    original = memory_module._request_summary

    def observe_summary(messages):
        event = {
            "request_messages": [{"type": m.type, "content": deepcopy(m.content)} for m in messages],
            "status": "running", "response": None, "usage": None, "error_type": None,
        }
        record["summary_calls"].append(event)
        started = time.perf_counter()
        try:
            response = original(messages)
            # returned只表示模型返回；是否采用由真实memory更新结果另行观察。
            event.update(status="returned", response=deepcopy(response.content),
                         usage=deepcopy(getattr(response, "usage_metadata", None)))
            return response
        except Exception as exc:
            event.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            event["latency_ms"] = round((time.perf_counter() - started) * 1000)

    try:
        # 仅观察真实摘要入口；固定历史不调用问答模型。禁止并发线程使用此适配器。
        with patch.object(memory_module, "_request_summary", observe_summary):
            for index, turn in enumerate(case.history, 1):
                before_count = len(memory.messages)
                memory.add_user(turn.user)
                memory.add_ai(turn.assistant)
                record["memory_steps"].append({
                    "round": index, "summary": memory.summary,
                    "retained_history_messages": len(memory.messages) - 1,
                    "summary_adopted": len(memory.messages) < before_count + 2,
                })
        messages = [*memory.get_messages_for_llm(), HumanMessage(content=case.question)]
        record["final_request_messages"] = [
            {"type": m.type, "content": deepcopy(m.content)} for m in messages
        ]
        started = time.perf_counter()
        try:
            response = create_llm().invoke(messages)
        finally:
            record["final_latency_ms"] = round((time.perf_counter() - started) * 1000)
        if not isinstance(response.content, str) or not response.content.strip():
            raise ValueError("最终回答不是非空文本")
        record.update(status="completed", response=response.content,
                      final_usage=deepcopy(getattr(response, "usage_metadata", None)))
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__)
    record["summary_adoptions"] = sum(step["summary_adopted"] for step in record["memory_steps"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--strategy", choices=["full", "window", "summary"], required=True)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--trigger-chars", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    if args.window_size < 1 or args.trigger_chars < 1:
        parser.error("窗口和实验触发阈值必须为正整数")
    try:
        case = load_probe(args.case)
        if args.output.exists():
            raise ValueError("结果文件已存在，请换名")
        before = build_manifest(args.case, args.strategy, args.window_size, args.trigger_chars)
        if not args.llm:
            print(json.dumps({"mode": "preflight_only", "manifest": before}, ensure_ascii=False, indent=2))
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # 模型调用前独占创建；不覆盖旧结果，不写业务学习存档。
        with args.output.open("x", encoding="utf-8") as stream:
            record = run_probe(case, args.strategy, args.window_size, args.trigger_chars)
            after, condition_error = None, None
            try:
                after = build_manifest(args.case, args.strategy, args.window_size, args.trigger_chars)
            except (OSError, ValueError) as exc:
                condition_error = type(exc).__name__
            record["experiment"] = {
                "before": before, "after": after,
                "conditions_unchanged": before == after if after is not None else None,
                "condition_error_type": condition_error,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, ValueError) as exc:
        parser.error(f"输入或结果处理失败：{type(exc).__name__}")
    print(f"{record['case_id']}: {record['status']} strategy={args.strategy} "
          f"summary_calls={len(record['summary_calls'])} adopted={record['summary_adoptions']}")
    if record["status"] != "completed" or record["experiment"]["conditions_unchanged"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

