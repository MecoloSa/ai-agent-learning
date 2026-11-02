# 固定历史的三策略诊断：代码方案（未应用）

新建位置：src/ai_agent/evaluation/memory_probe.py。该模块是新增文件，没有修改前代码；下方为完整内容，所有导入已包含。
不修改main、memory、反馈计划运行器、存档或默认策略。它是隔离机制实验，不是完整Agent端到端实验。

## 为什么采用固定历史

三种策略收到完全相同的用户与助手历史，最后一问才调用真实问答模型；summary另调用真实摘要模型。
固定助手回复为合成资料，不是本次模型生成结果，也不是正式确认记录。criteria只记录在结果，不送入模型。
评测记录中的source_history供人核查，不会给被测模型提供额外检索工具，不能当作产品原文回查已实现。

## 完整固定历史与最后提问

初始状态为空聊天记忆，无业务存档、任务或正式反馈；每种策略独立初始化。以下助手回复是固定合成历史，不在回放阶段调用模型。

### 第1轮

用户原文：

> 这轮练习名为松针练习。今天最多20分钟，只能阅读与口头说明，不绘图、不编程。这些只是本轮对话条件，不要登记任务或正式反馈。

固定助手回复：

> 了解，这是本轮对话中的练习条件，不写入正式记录。

### 第2轮

用户原文：

> 更正：今天最多35分钟，其他条件保持。这里只更正对话条件，不修改任何计划或记录。

固定助手回复：

> 了解，可用时间改为35分钟，其余条件保持，不修改正式记录。

### 第3轮

用户原文：

> 先不讨论安排。请解释活动完成和已经理解为什么需要分开记录。只讨论，不记录状态。

固定助手回复：

> 活动完成描述做了什么，理解情况描述能否解释或应用，两者不能互相推导。

### 第4轮

用户原文：

> 如果阅读做完但仍不能举例，应该如何描述进度？只讨论，不更新任何记录。

固定助手回复：

> 可以描述为阅读活动已完成，但理解或应用仍待确认，不把阅读完成直接当作掌握。

### 第5轮

用户原文：

> 请解释计划候选和已生效计划的区别。本轮不生成候选。

固定助手回复：

> 候选供用户审阅，只有程序完成明确确认后才可能成为生效计划；提出建议不等于已经采用。

### 第6轮

用户原文：

> 请解释为什么自编示例不能冒充原书引用。只讨论这一点，不改变先前约定。

固定助手回复：

> 自编示例可以辅助解释，但来源不同，不能虚构书页或声称原书写过这些内容。

### 最后实际提问

> 请依据对话复述这轮练习的名称、当前时间上限、允许的活动与不允许的活动。缺少的信息明确说明不知道，不猜测。不生成计划、任务或反馈。

预期核对名称松针练习、当前35分钟、只阅读与口头说明、不绘图不编程。窗口无法看到早期原文时允许说明不知道；召回缺失与编造分别评审。

## 完整新文件

```python
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
```

## 案例与调用边界

读取tests/fixtures/evaluation/memory_probe_case.json。六轮固定历史，每种策略独立创建记忆，不接续上次存档。
默认实验触发阈值120字符，窗口6条，摘要采用上限读取当前配置；120用于压力测试，不改变产品2000字符默认值。
full/window各一次最终调用；summary最多五次摘要入口调用加一次最终调用。SDK内部重试不在此入口计数中。
usage缺失保留null，不视作0，也不根据字符数伪造token或费用。
summary_adopted依据真实内存历史缩减观察；模型返回成功仍可能被长度校验拒绝。
summary_adoptions=0时不能据此评价“有效压缩后”的表现。completed只表示流程完成，assessment保持not_evaluated。
源码指纹仅包含probe/memory/config，不是完整环境快照；不记录API key或原始端点。

## 实施顺序

1. 按上方内容新建文件，其他业务文件不改。
2. 运行tests/test_memory_probe.py及记忆关联回归，先做无网络验收。
3. 再使用--llm分别运行full/window/summary，保存不同文件，不覆盖已有结果。
4. 查看source_history、summary_calls、memory_steps、final_request_messages与response，再做内容判断。

## 后续路线

先比较召回、未知与编造，观察摘要遗漏发生在哪次压缩。原文归档、受控回读应先于向量历史检索：
稳定消息ID/会话ID与来源 -> 原文可回查 -> 精确或词法检索基线 -> 必要时加入向量召回与重排。
本步不改变长期路线，也不把一个开发案例当作完整记忆评测集。

