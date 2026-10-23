# 下一步：只读查询窗口外的程序确认回执

日期：2026-10-22。上一轮确认原因修改已经直接完成；本文是下一步手动修改方案，不直接写入业务源码。

## 1. 本轮验收与为什么走这一步

已落地功能的实际源码全量942通过。在线真实模型能说明输入n否决了60分钟反馈候选，计划候选也能否决/采用并保留任务与反馈。但恢复后追问较早事件出现混淆：把60分钟反馈候选说成60分钟计划，并编造更正经过。

核对实际代码：ConversationMemory.get_messages_for_llm的window仅取rest[-window_size:]，export_state仍保存所有原human/ai消息。本次存档保留14条消息及3条程序确认回执。既不需要改成full策略，也不需要新增事件数据库或升级v7。

下一步只提供按需、有界查询，业务链路仍是：模型提出行动→用户确认→程序执行→程序回执→记忆保存。新增工具读取已保存回执，不重新执行操作。

## 2. 具体执行例子

原流程：恢复window会话→较早的60分钟回执不在当前模型窗口→模型只有当前40分钟和部分计划历史→可能错误补全原因。

新流程：用户追问旧候选→read_confirmation_history读取当前存档原AI消息尾部的程序回执→返回学习反馈候选60分钟被n否决、计划候选35分钟被n否决的独立证据→模型据此解释，正式状态不变。

已有read_feedback_history查询“正式反馈”，新工具查询“程序确认历史”。二者不能混用。否决60分钟不是确认今天只有60分钟，也不是取消某个学习任务。

## 3. 范围、复用与风险

- 只读当前已加载存档，不扫其他文件，不采用候选，不创建pending，共享现有工具预算。
- 查询出的AI候选及回执不能作为新反馈的用户原话来源；原source_id来源表及其权限边界不变。
- 最近已有回执足够时不必重复查询；相关证据不在窗口时再读取。
- 按登记消息倒序分页，默认3条、最多5条；来源消息索引仅用于当前存档定位，不是永久事件ID。
- 正式状态仍读当前LearningState；模型候选片段只帮助识别对象，不作为确认事实。
- 只解析既定程序标记后的JSON，不解析human消息、摘要或普通模型叙述。新标记已在上一轮防冒充规则中保护；这不是签名审计，用户手工篡改存档不在可验证范围内。
- summary已压缩掉的原回执可能确实不可查；不能从摘要补造，也不宣称已实现完整长期操作日志。
- 旧计划回执可能没有decision或context_revision，返回null，不从语言或当前修订号推测。
- 本步不改记忆策略、存储格式、模型或检索，也不增确认弹窗。

## 4. 新增模块全文

新增文件：`F:/LLM/projects/agent/src/ai_agent/learning/confirmation_history.py`。无修改前代码。

作用：独立的只读查询函数和参数schema，供绑定工具复用；不依赖模型客户端，也不写磁盘。必要边界注释已在代码中。

```python
"""按需读取当前存档保留的程序确认回执，而不是把全部历史塞进模型窗口。

不扫描其他文件、不写状态、不采用候选。window隐藏的原消息仍可查询；
summary已经压掉的原回执不能从摘要补造，空结果只说明本查询范围没有匹配。
"""
from __future__ import annotations

import json
from pydantic import Field
from ai_agent.learning.learning_state import LearningState, StateModel


class ConfirmationHistoryArgs(StateModel):
    offset: int = Field(default=0, ge=0, strict=True)
    limit: int = Field(default=3, ge=1, le=5, strict=True)


def read_confirmation_history(state: LearningState, request: ConfirmationHistoryArgs) -> str:
    """只读AI消息尾部的既有程序回执；普通回答、用户文本和摘要不是确认事实。"""
    snapshot = state.conversation_memory
    messages = snapshot.messages if snapshot is not None else ()
    markers = {
        "\n【程序学习操作结果数据】\n": "learning_action",
        "\n【程序确认结果数据】\n": "context_change",
    }
    reasons = {"user_confirmed", "user_declined", "user_not_confirmed", "input_ended", "not_recorded"}
    records = []
    unreadable = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role != "ai":
            continue
        present = [(message.content.rfind(marker), marker) for marker in markers if marker in message.content]
        if not present:
            continue
        position, marker = max(present)
        try:
            payload = json.loads(message.content[position + len(marker):].strip())
        except (ValueError, TypeError):
            unreadable += 1
            continue
        if not isinstance(payload, dict):
            unreadable += 1
            continue
        kind = markers[marker]
        if kind == "learning_action" and payload.get("context_type") != "learning_action_result":
            unreadable += 1
            continue

        # 不从自然语言猜测用户是否否决；旧回执没有该字段时保留未知。
        reason = payload.get("confirmation_reason", "not_recorded")
        if not isinstance(reason, str) or reason not in reasons:
            reason = "not_recorded"
        item = {
            "source_message_index": index,
            "receipt_kind": kind,
            "action": payload.get("action") if kind == "learning_action" else "context_change",
            "decision": payload.get("decision"),
            "confirmation_reason": reason,
            "decision_note": str(payload.get("decision_note") or "未保留具体确认原因，不推断。")[:500],
            "context_revision": payload.get("context_revision"),
        }
        proposal = payload.get("feedback_proposal")
        if isinstance(proposal, dict):
            text = str(proposal.get("text") or "")
            item["feedback_proposal"] = {
                "record_id": proposal.get("record_id"),
                "supersedes_record_id": proposal.get("supersedes_record_id"),
                "text": text[:1000],
                "text_truncated": bool(proposal.get("text_truncated")) or len(text) > 1000,
            }
        # 模型正文只帮助识别是哪份候选，不当作正式状态或确认原因的依据。
        preview = message.content[:position]
        item["assistant_candidate_excerpt"] = preview[:1000]
        item["assistant_candidate_excerpt_truncated"] = len(preview) > 1000
        records.append(item)

    page = records[request.offset:request.offset + request.limit]
    end = request.offset + len(page)
    has_more = end < len(records)
    # 动态评估：核对候选类型、确认原因及分页；不得把历史候选混成当前计划/正式反馈。
    return json.dumps({"ok": True, "data": {
        "search_scope": "current_loaded_state.conversation_memory.raw_ai_receipts",
        "memory_strategy": snapshot.strategy if snapshot is not None else None,
        "records": page,
        "total_matches": len(records),
        "has_more": has_more,
        "next_offset": end if has_more else None,
        "unreadable_receipt_count": unreadable,
        "note": (
            "仅查询当前存档仍保留的原始程序回执，不含其他存档或从摘要推测的事件。"
            "window只限制模型窗口，保留的原消息可按需查询；summary已压掉的回执可能缺失。"
            "空结果不代表从未发生；旧回执未知字段不能补造。"
            "assistant_candidate_excerpt是历史模型候选，不是正式计划或确认事实；"
            "当前生效值仍以最新学习状态为准。查询不确认、不重新执行任何操作。"
        ),
    }}, ensure_ascii=False)
```

## 5. feedback_runtime.py

文件：[feedback_runtime.py](F:/LLM/projects/agent/src/ai_agent/learning/feedback_runtime.py)

复用现有收件箱每轮绑定的真实LearningState；新增只读工具，不占用也不清除pending。

### 5.1 模块导入/声明，原第38行附近

作用：引入独立的参数和只读实现，避免重复实现状态读取或文件访问。

修改前，完整替换以下片段：

```python
from ai_agent.learning.feedback_history import (
```

修改后：

```python
from ai_agent.learning.confirmation_history import (
    ConfirmationHistoryArgs, read_confirmation_history,
)
from ai_agent.learning.feedback_history import (
```

### 5.2 __init__，原第159行附近

作用：新增确认历史工具schema和用途说明，不改变原正式反馈查询工具。

修改前，完整替换以下片段：

```python
        self.history_tool = StructuredTool.from_function(
```

修改后：

```python
        self.confirmation_history_tool = StructuredTool.from_function(
            func=self._read_confirmation_history,
            name="read_confirmation_history",
            description=(
                "只读查询当前存档保留的程序确认历史，用于核对哪条反馈/计划/任务候选被采用或否决。"
                "当相关回执不在当前窗口、需要解释此前候选为何未采用时调用；不要只凭讨论补造原因。"
                "不是已确认反馈查询，不修改状态；旧回执或summary缺失时须说明未知。"
                "返回候选摘要只是识别对象，不能当成当前正式计划或事实。"
            ),
            args_schema=ConfirmationHistoryArgs,
        )
        self.history_tool = StructuredTool.from_function(
```

### 5.3 _read_feedback_history，原第234行附近

作用：查询时使用本轮已经绑定的状态；不接收外部文件路径、不生成提议。

修改前，完整替换以下片段：

```python
    def _read_feedback_history(self, **kwargs) -> str:
```

修改后：

```python
    def _read_confirmation_history(self, **kwargs) -> str:
        if self._state is None:
            raise ValueError("尚未绑定本轮学习状态")
        return read_confirmation_history(
            self._state, ConfirmationHistoryArgs.model_validate(kwargs),
        )

    def _read_feedback_history(self, **kwargs) -> str:
```

## 6. main.py

文件：[main.py](F:/LLM/projects/agent/src/ai_agent/main.py)

工具必须同时绑定到模型schema及实际分发表；只改其中一处会出现“知道工具但执行不了”。提示词说明两类历史的不同职责，不用提示词假装恢复未读证据。

### 6.1 _build_interactive_system_prompt，原第350行附近

作用：让模型在较早回执不在当前窗口时有明确的查询入口，并禁止把当前40分钟反推成未验证的历史经过。

修改前，完整替换以下片段：

```python
        "空结果、分页和截断须如实说明；历史记录及unreplaced都不等于当前有效约束。"
```

修改后：

```python
        "空结果、分页和截断须如实说明；历史记录及unreplaced都不等于当前有效约束。"
        "查询此前候选为何采用或未采用、回执不在当前窗口时，调用read_confirmation_history；"
        "它查询保留的程序确认历史，不是正式反馈，不得把反馈候选混为计划候选。"
        "没有相关回执时说明无法核实，不能从当前40分钟倒推出60分钟曾被确认或更正。"
```

### 6.2 run_interactive_session，原第1173行附近

作用：分别补齐模型可见工具列表/真实执行分发表；二者必须同步，否则绑定与执行不一致。

修改前，完整替换以下片段：

```python

        feedback_inbox.cancellation_tool, feedback_inbox.history_tool,
```

修改后：

```python

        feedback_inbox.cancellation_tool, feedback_inbox.history_tool,
        feedback_inbox.confirmation_history_tool,
```

### 6.3 run_interactive_session，原第1288行附近

作用：分别补齐模型可见工具列表/真实执行分发表；二者必须同步，否则绑定与执行不一致。

修改前，完整替换以下片段：

```python

                feedback_inbox.cancellation_tool, feedback_inbox.history_tool,
```

修改后：

```python

                feedback_inbox.cancellation_tool, feedback_inbox.history_tool,
                feedback_inbox.confirmation_history_tool,
```

## 7. execution_guard.py

文件：[execution_guard.py](F:/LLM/projects/agent/src/ai_agent/runtime/execution_guard.py)

将新工具登记到现有结构化成功判定，沿用ok/data契约，不更改预算或循环保护。

### 7.1 模块导入/声明，原第83行附近

作用：新工具成功只代表查询返回了结构化结果，不代表历史完整或回答语义已被证明。

修改前，完整替换以下片段：

```python
    "read_feedback_history": _structured_result_succeeded,
```

修改后：

```python
    "read_feedback_history": _structured_result_succeeded,
    "read_confirmation_history": _structured_result_succeeded,
```

## 8. 测试与验证依据

测试已加入tests/test_confirmation_history.py，共13项；模块尚未应用时显式跳过，应用代码后自动启用。覆盖窗口外读取、分页/非法参数、空结果、畸形回执、human/摘要不作证据、pending不变、三种策略接线与保存恢复。

候选进程内全量955通过、1跳过；真实业务源码尚未应用这个新工具。单次在线复验使用相同失败存档、相同问题，模型自主调用read_confirmation_history一次，正确区分60分钟反馈候选和35分钟计划候选；2次模型响应、合计15657 tokens。这是针对性单次复验，不宣称统计稳定收益。原始对照见本地.evaluation-results/confirmation-reason-v1/online-9bf02a1733/REVIEW.md及.evaluation-results/confirmation-history-next/replay-9db0de3c3c/traces.json。

应用全部代码后执行：

```powershell
uv run python -m pytest tests/test_confirmation_history.py tests/test_confirmation_receipt_reason.py tests/test_feedback_history.py -q -p no:cacheprovider
uv run python -m pytest tests -q -p no:cacheprovider
```

预期关联59项通过；若当前代码未变，全量955通过、1项原有跳过。

## 9. 在线命令及交互

复用已为本轮创建的独立实验存档，无需另生成计划：

```powershell
uv run python -m ai_agent.main --resume-state "F:/LLM/projects/agent/.evaluation-results/confirmation-reason-v1/online-9bf02a1733/current.json"
```

先输入：

```text
依据已恢复的正式记录，说明当前生效计划、确认过的时间和学习方式限制；再说明刚才的60分钟候选及第一次计划候选为何未采用。不猜心理原因，不修改状态。
```

检查：

1. 先读程序确认历史，不从当前40分钟猜测旧事件；默认3条在这个存档足够。
2. 60分钟是被否决的反馈更正候选，不是60分钟计划，更没有成为正式反馈。
3. 第一次计划候选35分钟被输入n否决；后来35分钟候选经输入y采用，当前修订号2。
4. 正式反馈仍为40分钟、只读不写代码；任务和理解情况不因查询改变。
5. 查询不出现确认弹窗，/state的计划/任务/正式反馈/修订号不变；对话历史仍正常保存。

分页可再输入“请实际调用确认历史查询，每次只取1条，先查看最近一条，再翻到下一条；不修改状态。”分页结果不同不代表矛盾，应核对来源索引和候选类型。

缺失路径可在一个无程序确认回执的旧演示存档中查询，结果应说明本范围未找到或未记录原因，不能声称用户从未否决过。不要为了制造EOF而破坏当前问答输入流；EOF已由离线测试覆盖。

## 10. 验收后何时停止

这条复现问题通过后就收束本轮，不再扩成完整审计平台或额外记忆Agent。记忆查询保持精确、按需和可追溯；后续按既定规划进入一两个只读资料工具的MCP对照，再做工程收敛。若summary确实需要跨长期回溯，再单独评估持久化操作记录，不先预建。

