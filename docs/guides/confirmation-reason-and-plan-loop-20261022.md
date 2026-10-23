# 确认原因回执：修改前后与闭环验收

日期：2026-10-22。本文所列确认原因修改已按用户授权直接应用；下方前后片段保留为变更说明，无需重复应用。已完成真实模型验收，恢复后跨窗口历史解释的失败及下一步见[验收报告](../../.evaluation-results/confirmation-reason-v1/online-9bf02a1733/REVIEW.md)和[只读历史查询方案](confirmation-history-next-20261022.md)。

## 1. 为什么修改

当前回答正确保留了40分钟，没有把60分钟候选当成正式反馈；但原程序将用户输入n、其他非y输入以及EOF均归入cancelled，且终端的确认输入不作为普通用户消息写入聊天，因此模型难以解释为什么未采用。不能仅凭这段回复断言模型完全不知道被否决。

这次只补可观察的确认事实，不推断用户拒绝的心理动机；不增模型调用、工具或正式反馈状态，不升级存档版本。旧回执不能补造n；window裁掉回执或summary漏摘要后也不能保证永久记得。

## 2. 原流程与新流程

原流程：确认输入 → confirmed/cancelled/invalid → 泛化回执 → 聊天记忆。

新流程：确认输入 → 保留既有decision，同时捕获有限的confirmation_reason → 与实际候选关联的程序回执 → 现有full/window/summary → 保存恢复。

| 实际输入 | reason | 能说明什么 |
| --- | --- | --- |
| y | user_confirmed | 用户确认采用 |
| n | user_declined | 用户在确认环节明确否决此候选 |
| 空输入或其他字符 | user_not_confirmed | 没有确认，不能补猜心理原因 |
| EOF | input_ended | 输入结束，不能说成用户明确否决 |
| 旧记录/未保留/不一致 | not_recorded | 不推断具体确认输入 |

取消候选不是取消学习任务；否决60分钟不会删除40分钟正式反馈，也不修改计划。

## 3. 修改顺序与关联范围

先新增下面的确认工具函数，再按文件应用所有替换。所有“修改前”均来自当前实际源码；各代码块为完整匹配片段，没有省略号。行号用于辅助定位，前面的修改导致行号变化时请按函数名和原片段查找。函数头替换只替换展示的头部，保留未列出的函数体。

自然反馈、创建任务、取消任务共用相同确认语义。计划变更虽然是独立协议，也接入同一只读确认原因采集。新增的程序确认结果标记同时纳入现有防冒充检查。兼容旧调用：新增参数均为可选关键字参数，返回值仍是(state, decision)。

## 4. 新增确认辅助模块

新增位置：`F:/LLM/projects/agent/src/ai_agent/learning/confirmation.py`。这是新文件，无修改前代码。

作用：集中读取一次终端输入，捕获有限结果；统一解释确认事实，旧记录或不一致原因不补造。它不导入业务状态、不提交操作、不保存文件，也不吞掉KeyboardInterrupt等其他异常。

```python
"""读取终端确认结果，只记录可观察事实，不推断用户拒绝的心理原因。

不读写学习状态或快照；n与输入结束均不执行，但必须保留两者的区别。
"""
from __future__ import annotations


def read_confirmation(prompt: str, *, trace: dict[str, str] | None = None) -> bool:
    """保留既有y采用、其他输入不采用的行为；可选trace只供程序生成回执。"""
    if trace is not None:
        trace.clear()
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        reason = "input_ended"
    else:
        reason = (
            "user_confirmed" if answer == "y"
            else "user_declined" if answer == "n"
            else "user_not_confirmed"
        )
    # 不保存任意原始输入，只保存有限结果；不把空输入或断流说成用户明确否决。
    if trace is not None:
        trace["reason"] = reason
    return reason == "user_confirmed"


def confirmation_details(decision: str, trace: dict[str, str] | None = None) -> dict[str, str]:
    """将程序结果解释为回执；旧记录或不一致数据不补造确认原因。"""
    reason = (trace or {}).get("reason", "not_recorded")
    allowed = {
        "confirmed": {"user_confirmed"},
        "cancelled": {"user_declined", "user_not_confirmed", "input_ended"},
        "invalid": set(),
        "unchanged": set(),
    }
    if reason not in allowed[decision]:
        reason = "not_recorded"
    notes = {
        "user_confirmed": "用户在确认环节输入y，候选已采用。",
        "user_declined": "用户在确认环节输入n，明确否决本次候选；未执行。",
        "user_not_confirmed": "确认环节收到非y/n输入，未采用；不推断用户原因。",
        "input_ended": "确认输入结束，未采用；不能说成用户明确否决。",
    }
    fallback = {
        "confirmed": "候选已确认；未保留具体确认输入类型。",
        "cancelled": "候选未采用；未保留具体确认输入类型，不能推断未采用原因。",
        "invalid": "提议未通过程序校验或已失效；不能说成用户否决。",
        "unchanged": "没有实际变更，未请求确认。",
    }
    return {"confirmation_reason": reason, "decision_note": notes.get(reason, fallback[decision])}
```

## 5. feedback_runtime.py

文件：[feedback_runtime.py](F:/LLM/projects/agent/src/ai_agent/learning/feedback_runtime.py)

补实际确认事实与候选关联；正式反馈仍由原提交函数控制。增加新保留标记的防冒充检查，避免模型自行编造确认结果。

### 5.1 sanitize_learning_action_reply，原第49行附近

作用：新程序结果标记也必须拒绝模型自行生成，沿用既有输出防冒充边界，不增加模型重试。

修改前，完整替换以下片段：

```python
        "【程序学习操作结果数据】",
```

修改后：

```python
        "【程序学习操作结果数据】",
        "【程序确认结果数据】",
```

### 5.2 模块顶部/声明，原第17行附近

作用：导入新的无状态确认辅助函数，后续替换复用同一输入规则，避免各入口解释不一致。

修改前，完整替换以下片段：

```python
from ai_agent.learning.feedback_proposals import (
```

修改后：

```python
from ai_agent.learning.confirmation import read_confirmation, confirmation_details

from ai_agent.learning.feedback_proposals import (
```

### 5.3 build_learning_action_history，原第68行附近

作用：新增可选关键字参数，让调用方接收/传递本次确认事实；旧调用与(state, decision)返回结构保持兼容。

修改前，完整替换以下片段：

```python
    decision: str,
) -> str:
```

修改后：

```python
    decision: str,
    *,
    confirmation_trace: dict[str, str] | None = None,
) -> str:
```

### 5.4 build_learning_action_history，原第94行附近

作用：把有限确认原因与可读说明加入程序回执；没有来源或与decision不一致时不补造用户否决。

修改前，完整替换以下片段：

```python
    result = {
        "context_type": "learning_action_result",
```

修改后：

```python
    result = {
        "context_type": "learning_action_result",
        **confirmation_details(decision, confirmation_trace),
```

### 5.5 build_learning_action_history，原第105行附近

作用：为反馈候选补对象关联，便于回答“60分钟那条为什么未采用”；回显有界并标注截断，不进入正式反馈。

修改前，完整替换以下片段：

```python
    # 原回复保留为讨论背景，末尾追加程序事实；不清空历史或修改摘要策略。
```

修改后：

```python
    if isinstance(pending, PendingFeedback):
        proposed_text = pending.request.replacement_text or pending.user_message
        # 仅用于关联“哪条候选未采用”；不能把它混入正式反馈读取视图。
        result["feedback_proposal"] = {
            "record_id": pending.proposal_id,
            "supersedes_record_id": pending.request.supersedes_record_id,
            "text": proposed_text[:1000],
            "text_truncated": len(proposed_text) > 1000,
        }

    # 原回复保留为讨论背景，末尾追加程序事实；不清空历史或修改摘要策略。
```

### 5.6 review_learning_action，原第381行附近

作用：新增可选关键字参数，让调用方接收/传递本次确认事实；旧调用与(state, decision)返回结构保持兼容。

修改前，完整替换以下片段：

```python
def review_learning_action(
    state: LearningState,
    pending: PendingFeedback | PendingTaskCreation,
) -> tuple[LearningState, str]:
```

修改后：

```python
def review_learning_action(
    state: LearningState,
    pending: PendingLearningAction,
    *,
    confirmation_trace: dict[str, str] | None = None,
) -> tuple[LearningState, str]:
```

### 5.7 review_learning_action，原第387行附近

作用：保持原执行语义，只让本次确认的实际结果沿现有调用链传递；不把临时信息写成正式学习反馈。

修改前，完整替换以下片段：

```python
return review_task_cancellation(state, pending)
```

修改后：

```python
return review_task_cancellation(state, pending, confirmation_trace=confirmation_trace)
```

### 5.8 review_learning_action，原第390行附近

作用：保持原执行语义，只让本次确认的实际结果沿现有调用链传递；不把临时信息写成正式学习反馈。

修改前，完整替换以下片段：

```python
return review_task_creation(state, pending)
```

修改后：

```python
return review_task_creation(state, pending, confirmation_trace=confirmation_trace)
```

### 5.9 review_learning_action，原第392行附近

作用：保持原执行语义，只让本次确认的实际结果沿现有调用链传递；不把临时信息写成正式学习反馈。

修改前，完整替换以下片段：

```python
return review_feedback_proposal(state, pending)
```

修改后：

```python
return review_feedback_proposal(state, pending, confirmation_trace=confirmation_trace)
```

### 5.10 review_feedback_proposal，原第487行附近

作用：新增可选关键字参数，让调用方接收/传递本次确认事实；旧调用与(state, decision)返回结构保持兼容。

修改前，完整替换以下片段：

```python
def review_feedback_proposal(
    state: LearningState,
    pending: PendingFeedback,
) -> tuple[LearningState, str]:
```

修改后：

```python
def review_feedback_proposal(
    state: LearningState,
    pending: PendingFeedback,
    *,
    confirmation_trace: dict[str, str] | None = None,
) -> tuple[LearningState, str]:
```

### 5.11 review_feedback_proposal，原第491行附近

作用：保持原执行语义，只让本次确认的实际结果沿现有调用链传递；不把临时信息写成正式学习反馈。

修改前，完整替换以下片段：

```python
    """一次明确确认；取消不改业务状态，聊天本身仍按原流程保存。"""
    try:
```

修改后：

```python
    """一次明确确认；取消不改业务状态，聊天本身仍按原流程保存。"""
    if confirmation_trace is not None:
        confirmation_trace.clear()
    try:
```

### 5.12 review_feedback_proposal，原第537行附近

作用：不再把EOF先改为空字符串而丢失原因；仍然只有y采用，其他输入均不采用。

修改前，完整替换以下片段：

```python
    try:
        answer = input("确认记录？输入y确认，其他输入取消：")
    except EOFError:
        answer = ""

    if answer.strip().lower() != "y":
```

修改后：

```python
    if not read_confirmation(
        "确认记录？输入y确认，其他输入取消：",
        trace=confirmation_trace,
    ):
```

## 6. task_creation.py

文件：[task_creation.py](F:/LLM/projects/agent/src/ai_agent/learning/task_creation.py)

创建任务也复用同一输入读取规则。只是补充回执原因，不改变创建候选、去重或提交规则。

### 6.1 模块顶部/声明，原第14行附近

作用：导入新的无状态确认辅助函数，后续替换复用同一输入规则，避免各入口解释不一致。

修改前，完整替换以下片段：

```python
from ai_agent.learning.learning_state import (
```

修改后：

```python
from ai_agent.learning.confirmation import read_confirmation

from ai_agent.learning.learning_state import (
```

### 6.2 review_task_creation，原第101行附近

作用：新增可选关键字参数，让调用方接收/传递本次确认事实；旧调用与(state, decision)返回结构保持兼容。

修改前，完整替换以下片段：

```python
    pending: PendingTaskCreation,
) -> tuple[LearningState, str]:
```

修改后：

```python
    pending: PendingTaskCreation,
    *,
    confirmation_trace: dict[str, str] | None = None,
) -> tuple[LearningState, str]:
```

### 6.3 review_task_creation，原第103行附近

作用：保持原执行语义，只让本次确认的实际结果沿现有调用链传递；不把临时信息写成正式学习反馈。

修改前，完整替换以下片段：

```python
    # 提前构造候选只做校验；未确认时绝不替换外层state。
```

修改后：

```python
    if confirmation_trace is not None:
        confirmation_trace.clear()
    # 提前构造候选只做校验；未确认时绝不替换外层state。
```

### 6.4 review_task_creation，原第120行附近

作用：不再把EOF先改为空字符串而丢失原因；仍然只有y采用，其他输入均不采用。

修改前，完整替换以下片段：

```python
    try:
        answer = input("输入y确认创建，其他输入取消：")
    except EOFError:
        answer = ""

    if answer.strip().lower() != "y":
```

修改后：

```python
    if not read_confirmation(
        "输入y确认创建，其他输入取消：",
        trace=confirmation_trace,
    ):
```

## 7. task_cancellation.py

文件：[task_cancellation.py](F:/LLM/projects/agent/src/ai_agent/learning/task_cancellation.py)

取消任务的提议是否采用，与任务自身是否取消是两件事。复用确认采集，保留原状态转换。

### 7.1 模块顶部/声明，原第9行附近

作用：导入新的无状态确认辅助函数，后续替换复用同一输入规则，避免各入口解释不一致。

修改前，完整替换以下片段：

```python
from ai_agent.learning.learning_state import LearningState, StateModel
```

修改后：

```python
from ai_agent.learning.confirmation import read_confirmation
from ai_agent.learning.learning_state import LearningState, StateModel
```

### 7.2 review_task_cancellation，原第100行附近

作用：新增可选关键字参数，让调用方接收/传递本次确认事实；旧调用与(state, decision)返回结构保持兼容。

修改前，完整替换以下片段：

```python
    pending: PendingTaskCancellation,
) -> tuple[LearningState, str]:
    try:
```

修改后：

```python
    pending: PendingTaskCancellation,
    *,
    confirmation_trace: dict[str, str] | None = None,
) -> tuple[LearningState, str]:
    if confirmation_trace is not None:
        confirmation_trace.clear()
    try:
```

### 7.3 review_task_cancellation，原第126行附近

作用：不再把EOF先改为空字符串而丢失原因；仍然只有y采用，其他输入均不采用。

修改前，完整替换以下片段：

```python
    try:
        answer = input("确认取消该任务？输入y确认，其他输入放弃本次提议：")
    except EOFError:
        answer = ""

    if answer.strip().lower() != "y":
```

修改后：

```python
    if not read_confirmation(
        "确认取消该任务？输入y确认，其他输入放弃本次提议：",
        trace=confirmation_trace,
    ):
```

## 8. main.py

文件：[main.py](F:/LLM/projects/agent/src/ai_agent/main.py)

将临时确认trace从实际确认位置传到回执组装处；反馈与计划各用独立字典，防止把上一次确认原因串到另一个操作。trace只是程序临时值，不是模型参数或新状态。

### 8.1 模块顶部/声明，原第55行附近

作用：导入新的无状态确认辅助函数，后续替换复用同一输入规则，避免各入口解释不一致。

修改前，完整替换以下片段：

```python
from ai_agent.learning.learning_context import build_learning_state_message
```

修改后：

```python
from ai_agent.learning.confirmation import read_confirmation, confirmation_details
from ai_agent.learning.learning_context import build_learning_state_message
```

### 8.2 _review_context_change，原第953行附近

作用：保持原执行语义，只让本次确认的实际结果沿现有调用链传递；不把临时信息写成正式学习反馈。

修改前，完整替换以下片段：

```python
    learning_goal: str | None = None,
) -> tuple[LearningState, str]:
    """构造候选并请求确认，不写文件、不更新会话或聊天记忆。"""
```

修改后：

```python
    learning_goal: str | None = None,
    confirmation_trace: dict[str, str] | None = None,
) -> tuple[LearningState, str]:
    """构造候选并请求确认，不写文件、不更新会话或聊天记忆。"""
    if confirmation_trace is not None:
        confirmation_trace.clear()
```

### 8.3 _review_context_change，原第988行附近

作用：不再把EOF先改为空字符串而丢失原因；仍然只有y采用，其他输入均不采用。

修改前，完整替换以下片段：

```python
    try:
        answer = input(
            "[确认变更] 输入 y 采用以上目标/计划，其他输入取消："
        )
    except EOFError:
        # 输入流结束不能被解释为同意。
        answer = ""

    if answer.strip().lower() != "y":
```

修改后：

```python
    if not read_confirmation(
        "[确认变更] 输入 y 采用以上目标/计划，其他输入取消：",
        trace=confirmation_trace,
    ):
```

### 8.4 _context_change_history，原第1007行附近

作用：新增可选关键字参数，让调用方接收/传递本次确认事实；旧调用与(state, decision)返回结构保持兼容。

修改前，完整替换以下片段：

```python
    revision: int,
) -> str:
    """将程序处理结果与候选回复一起记录，避免误认候选已生效。"""
```

修改后：

```python
    revision: int,
    *,
    confirmation_trace: dict[str, str] | None = None,
) -> str:
    """将程序处理结果与候选回复一起记录，避免误认候选已生效。"""
    details = confirmation_details(status, confirmation_trace)
```

### 8.5 _context_change_history，原第1014行附近

作用：计划候选也追加相同确认原因数据，同时保留原有状态/修订号回执，不改变计划更新协议。

修改前，完整替换以下片段：

```python
        f"当前生效修订号={revision}；以当前系统状态为准】"
    )
```

修改后：

```python
        f"当前生效修订号={revision}；以当前系统状态为准】"
        + "\n【程序确认结果数据】\n"
        + json.dumps(details, ensure_ascii=False)
    )
```

### 8.6 run_interactive_session，原第1330行附近

作用：为这一轮操作分配独立临时字典；反馈与计划不能共用上次结果，避免确认原因串线。

修改前，完整替换以下片段：

```python
        if pending_feedback is not None:
            learning_state, feedback_decision = review_learning_action(
```

修改后：

```python
        if pending_feedback is not None:
            # 仅捕获本次确认输入的结果，随后随程序回执进入现有聊天记忆。
            confirmation_trace: dict[str, str] = {}
            learning_state, feedback_decision = review_learning_action(
```

### 8.7 run_interactive_session，原第1333行附近

作用：保持原执行语义，只让本次确认的实际结果沿现有调用链传递；不把临时信息写成正式学习反馈。

修改前，完整替换以下片段：

```python
                pending_feedback,
            )

            # 使用确认处理后的状态生成结果
```

修改后：

```python
                pending_feedback,
                confirmation_trace=confirmation_trace,
            )

            # 使用确认处理后的状态生成结果
```

### 8.8 run_interactive_session，原第1341行附近

作用：保持原执行语义，只让本次确认的实际结果沿现有调用链传递；不把临时信息写成正式学习反馈。

修改前，完整替换以下片段：

```python
                feedback_decision,
            )
```

修改后：

```python
                feedback_decision,
                confirmation_trace=confirmation_trace,
            )
```

### 8.9 run_interactive_session，原第1364行附近

作用：为这一轮操作分配独立临时字典；反馈与计划不能共用上次结果，避免确认原因串线。

修改前，完整替换以下片段：

```python
            previous_state = learning_state

            try:
                next_state, decision = _review_context_change(
```

修改后：

```python
            previous_state = learning_state
            context_confirmation_trace: dict[str, str] = {}

            try:
                next_state, decision = _review_context_change(
```

### 8.10 run_interactive_session，原第1370行附近

作用：保持原执行语义，只让本次确认的实际结果沿现有调用链传递；不把临时信息写成正式学习反馈。

修改前，完整替换以下片段：

```python
                    plan_markdown=updated_plan,
                )
```

修改后：

```python
                    plan_markdown=updated_plan,
                    confirmation_trace=context_confirmation_trace,
                )
```

### 8.11 run_interactive_session，原第1392行附近

作用：保持原执行语义，只让本次确认的实际结果沿现有调用链传递；不把临时信息写成正式学习反馈。

修改前，完整替换以下片段：

```python
                    decision,
                    learning_state.context_revision,
                ))
```

修改后：

```python
                    decision,
                    learning_state.context_revision,
                    confirmation_trace=context_confirmation_trace,
                ))
```

### 8.12 run_interactive_session，原第1444行附近

作用：保持原执行语义，只让本次确认的实际结果沿现有调用链传递；不把临时信息写成正式学习反馈。

修改前，完整替换以下片段：

```python
                "confirmed",
                learning_state.context_revision,
            ))
```

修改后：

```python
                "confirmed",
                learning_state.context_revision,
                confirmation_trace=context_confirmation_trace,
            ))
```

## 9. memory.py

文件：[memory.py](F:/LLM/projects/agent/src/ai_agent/persistence/memory.py)

摘要保留实际对象和确认原因，不把用户否决与断流混为一谈。只是摘要规则，不保证摘要模型永不遗漏。

### 9.1 _maybe_compress，原第214行附近

作用：摘要依据实际程序回执保存对象与最终决定，区分明确否决和断流；不能要求模型猜测拒绝动机。

修改前，完整替换以下片段：

```python
                    "对学习操作，优先依据回复末尾的程序学习操作结果数据，"
```

修改后：

```python
                    "对学习操作，优先依据程序学习操作结果数据；目标/计划变更依据程序确认结果数据，"
```

### 9.2 _maybe_compress，原第215行附近

作用：摘要依据实际程序回执保存对象与最终决定，区分明确否决和断流；不能要求模型猜测拒绝动机。

修改前，完整替换以下片段：

```python
                    "保留操作类型、任务ID及最终确认/取消/拒绝结论；"
```

修改后：

```python
                    "保留操作对象、最终结论及confirmation_reason；"
                    "区分用户输入n明确否决、未确认和输入结束，不编造用户拒绝的动机；"
```

## 10. 测试与已验证结果

测试已直接加入项目：tests/test_confirmation_receipt_reason.py新增28项；test_learning_action_output_boundary.py补充新保留标记回归。候选进程内全量942通过、1跳过、3项现有依赖弃用警告。未调用在线模型、未更改用户存档，不等于业务源码已应用或在线语言表现已验证。

应用全部代码后执行：

```powershell
uv run python -m pytest tests/test_confirmation_receipt_reason.py tests/test_learning_action_history.py tests/test_learning_action_output_boundary.py tests/test_context_change_confirmation.py tests/test_feedback_plan_loop.py -q -p no:cacheprovider
uv run python -m pytest tests -q -p no:cacheprovider
```

若系统临时目录权限报错，仅为该次测试分配新的仓库内临时目录，不修改业务代码。上述测试只证明确认、接线与状态边界，不证明模型总能正确解释反馈。

## 11. 在线验证本次改进

先用state_catalog核对最近存档目标，选择已经确认40分钟且保留只读不写代码的存档：

```powershell
uv run python -m ai_agent.state_catalog --limit 8
uv run python -m ai_agent.main --resume-state "替换为核对后的存档完整路径"
```

旧存档里先前的60分钟否决没有具体原因时，不能倒推出当时输入n。这次启动后重新测试。

1. 输入“把当前记录的可用时间更正为60分钟，其他要求不变，先不要修改计划。”；核对完整候选，输入n。
2. 输入“刚才60分钟候选为什么没有生效？只依据程序确认结果说明，不猜我的心理原因。再复述当前正式记录的时间限制。”
3. 预期：模型说明本次确认环节输入n否决候选，当前正式记录仍为40分钟；不能声称60分钟被保存、用户永久拒绝60分钟或断言用户拒绝的动机。/state不新增60分钟正式反馈。
4. 如需核对程序数据，退出后用下面命令检查这次自动保存路径的问答回执，而不是只查feedback_records。

```powershell
$checkStatePath = Read-Host "粘贴本次自动保存的完整JSON路径"
$checkState = Get-Content -LiteralPath $checkStatePath -Raw -Encoding UTF8 | ConvertFrom-Json
$checkState.state.conversation_memory.messages |
    Where-Object { $_.content -match 'confirmation_reason' } |
    Select-Object -Last 3 -ExpandProperty content
```

预期新回执含decision=cancelled、confirmation_reason=user_declined，以及相应60分钟feedback_proposal。反馈候选正文最多回显1000字符并带截断标记；这不截断正式原话，也不把候选写进feedback_records。

window近期回执仍在时恢复可继续解释；超出窗口不保证完整操作历史可查。summary另有模型压缩的不确定性，本轮不改成永久审计系统。

## 12. 按规划进入下一步：真实反馈驱动计划调整

本轮通过后不再增加第二组业务规则，直接在同一会话验收已有计划闭环：

1. 输入“先只讨论：依据已经确认的40分钟、只读不写代码，以及登记任务的真实状态，接下来怎么调整？不要修改计划或反馈。”预期只讨论；计划、修订号、任务、正式反馈不变，对话记忆仍可保存。
2. 输入“请依据上述已确认条件，为剩余活动提出完整的新计划。保持原学习目标，不自动修改任务进度或理解情况，等待我确认。”检查计划总安排不超过40分钟、不要求写代码、回应真实困难、不把已读完等同已掌握，不杜撰资料出处。不要求固定活动或固定工具轨迹。
3. 第一次候选输入n。检查计划与修订号不变；询问未采用原因应能依据新计划回执说明用户输入n。
4. 再次提出相同请求，候选正确才输入y。检查新计划采用、修订号增加1、原任务与反馈保留；tasks_need_review可能变为true，不等于任务或反馈自动失效。
5. 退出并从本次存档恢复，查询当前计划、登记任务与约束，三者应按各自来源解释，不把登记任务列表当成今日安排。恢复不得重新生成计划。

主要边界通过即收尾；只有实际复现的内容质量问题才继续改计划规则。之后按既定路线进入小范围只读MCP，再做工程收敛，不继续无目标扩展记忆/提示词实验。

