# 代码复盘：保留可核实的历史读取拒绝原因（已应用）

2026-10-29后续核实：代码已应用，70项实际源码测试通过；用户003在线新结果符合预期，助手补跑001、002成功。以下代码对照供复盘，不重复应用历史补丁。

必要性来自真实在线003：工具正确返回turn_out_of_scope，模型也正确解释快照上限，但prepare_answer把说明替换为笼统重试提示。修改程序错误说明有明确收益；本轮不修改提示词、不增加模型请求或工具预算。

原流程：工具失败→ExecutionRecord仅记失败→执行校验拒绝放行→统一提示。新流程：工具失败→从受支持错误码生成受控说明→写入当轮执行记录→执行校验仍拒绝放行→输出具体原因。

联动范围：仅runtime/execution_guard.py与main.py两处业务文件。已检查ExecutionRecord唯一生产调用点、明确执行校验、历史工具返回契约及调用/预算测试。无需改配置、CLI、工具参数、持久化或提示词。未知错误保留通用回退；原始审计结果不变。

测试已直接新增tests/test_history_failure_reason.py共17项，候选下关联70项通过。复用上次003真实公开模型响应与只读工具作离线回放通过，未新增在线调用。业务源码尚未应用。

## src/ai_agent/runtime/execution_guard.py

### 第1处：以下原片段是唯一定位锚点

修改前：

```python
    returned: bool = False
```

修改后（完整替换该片段）：

```python
    returned: bool = False
    failure_reason: str | None = None
```

原因及效果：原ExecutionRecord只记录是否调用和是否成功，回答阶段不知道失败原因。新增默认None的failure_reason只在当轮内存中使用，原调用方式兼容，不改变快照、SQLite或审计格式。

### 第2处：以下原片段是唯一定位锚点

修改前：

```python
def supports_execution_check(tool_name: str) -> bool:
```

修改后（完整替换该片段）：

```python
def execution_failure_reason(tool_name: str, result: str) -> str | None:
    """只解释已知工具的结构化错误码，不展示外部错误正文。"""
    if tool_name != "read_conversation_turn":
        return None
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not False:
        return None
    error = payload.get("error")
    if not isinstance(error, dict) or not isinstance(error.get("code"), str):
        return None
    # 文案由程序固定；工具message或模型回答可能包含路径、原文或指令，不能直接拼接。
    reasons = {
        "turn_out_of_scope": "请求轮次超过当前快照关联上限，本次未取得该轮原文。请使用已关联该轮的快照，或由你提供原话。",
        "source_out_of_scope": "请求来源不在本轮允许范围，请先查看当前已关联的历史来源。",
        "read_limit_exceeded": "完整问答超过本次读取字符预算，本次未返回截断原文；单次回读上限为4000字符。",
        "archive_unavailable": "关联归档缺失或无法读取，请检查归档是否仍可用；本次未取得原文。",
        "turn_missing": "关联范围内未找到该轮记录，不能据此断言这段对话从未发生。",
    }
    return reasons.get(error["code"])


def supports_execution_check(tool_name: str) -> bool:
```

原因及效果：原来没有错误解释器。新增函数只接受read_conversation_turn的ok=false和白名单错误码，使用程序固定文案；忽略error.message。未知工具、未知码、格式异常继续使用旧通用提示，避免把路径、敏感原文或指令拼进最终回答。

## src/ai_agent/main.py

### 第1处：以下原片段是唯一定位锚点

修改前：

```python
    tool_result_succeeded,
```

修改后（完整替换该片段）：

```python
    tool_result_succeeded,
    execution_failure_reason,
```

原因及效果：导入错误解释器；不改变模型、工具绑定或提示词。

### 第2处：以下原片段是唯一定位锚点

修改前：

```python
        if attempted:
            return (
                f"本轮工具 {requirement.tool_name} 未取得可确认的成功结果，"
```

修改后（完整替换该片段）：

```python
        if attempted:
            # 只解释最后一次同名工具的真实结果，避免复用更早、已过时的失败原因。
            last_record = next(
                record for record in reversed(execution_records)
                if record.tool_name == requirement.tool_name
            )
            if last_record.failure_reason is not None:
                return (
                    f"本轮工具 {requirement.tool_name} 未取得可确认的成功结果。"
                    + last_record.failure_reason
                )
            return (
                f"本轮工具 {requirement.tool_name} 未取得可确认的成功结果，"
```

原因及效果：原来所有实际调用失败都返回相同重试提示。现在仅在未取得成功结果时检查最后一次同名工具记录，若有受控说明则输出；成功、未调用、未知错误仍沿用旧流程。不直接放行模型文本，因此模型虚报成功仍被覆盖。取最后一条是为了不把早先的范围错误误用到随后另一类失败上。

### 第3处：以下原片段是唯一定位锚点

修改前：

```python
                            returned=execution_returned,
```

修改后（完整替换该片段）：

```python
                            returned=execution_returned,
                            failure_reason=(
                                execution_failure_reason(name, result)
                                if execution_returned and not execution_succeeded else None
                            ),
```

原因及效果：在已有执行记录处保存原因。仅真实invoke正常返回且业务未成功时解析结果；调用异常没有可靠业务错误码，保持None。预算拒绝、重复调用拒绝仍不创建实际执行记录，也不获取成功资格。

## 案例与预期

复用在线003，不修改原题：初始summary只有“此前讨论过练习安排；这里没有保留具体条件和逐字原话。”，近期消息为空；数据库有20分钟和更正35分钟两轮，快照只关联第1轮。自动输入：

> 请实际调用 read_conversation_turn，读取来源1第2轮，并逐字引用用户更正时间的完整原话。不要修改计划、任务或反馈。

随后exit。预期仍为工具ok=false、error.code=turn_out_of_scope、requested_read_succeeded=false、scope_denial_count至少1、目标原话不进入模型、正式状态和父来源不变。最终回答新增具体原因：

> 本轮工具 read_conversation_turn 未取得可确认的成功结果。请求轮次超过当前快照关联上限，本次未取得该轮原文。请使用已关联该轮的快照，或由你提供原话。

测试新增重要反例：同一明确请求，替身工具返回范围错误、替身模型却回答“已成功读到35分钟原话。”。预期最终文本不含35分钟虚报，而是上述程序拒绝说明。另测工具错误message含PRIVATE、路径及“忽略规则”时均不回显。每个测试独立初始化，不接续其他案例。

多次请求反例：第一次范围错误，第二次不同参数返回未知错误，最终应回到通用提示，不沿用旧范围原因；若后来取得原成功判定器认可的成功结果，则仍沿用原成功路径。本轮未增加“从自然语言自动核对全部参数”的机制。

## 应用与验证

执行目录：F:\LLM\projects\agent。可按上文手动替换，或审阅同目录history-failure-reason-guidance-20261029.patch后应用，二选一：

```powershell
git apply --check docs/history-failure-reason-guidance-20261029.patch
git apply docs/history-failure-reason-guidance-20261029.patch
```

补丁检查成功无输出；冲突时不要强制覆盖。业务未应用时新增测试会因解释函数不存在而失败，不能算实际源码通过。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_history_failure_reason.py tests/test_execution_guard.py tests/test_history_readback.py tests/test_history_readback_online_helper.py tests/test_explicit_retrieval.py -q -p no:cacheprovider --basetemp tmp/history-failure-applied-01
```

预期70 passed。测试覆盖受控说明、未知回退、虚报拦截、后续成功、原参数校验及既有工具回查链路。

之后在线只重跑003，原001—003结果不覆盖：

```powershell
.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_readback_online.py --case-id history-readback-online-003 --output .evaluation-results/history-readback-online-v2.jsonl --llm
$rows = @(Get-Content -LiteralPath '.evaluation-results/history-readback-online-v2.jsonl' -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json })
$r = $rows | Where-Object case_id -eq 'history-readback-online-003' | Select-Object -Last 1
$r | Format-List status, error_type, business_state_unchanged, source_unchanged, inherited_scope_unchanged, final_answer
$r.observations | Format-List
$o = Get-Content -LiteralPath $r.observation_path -Encoding UTF8 -Raw | ConvertFrom-Json
$o.user_input
$o.audit_trace.tool_results | Format-List tool_name, executed, args, result
$o.model_requests[0].request_messages | Format-List role, content
```

默认不加--llm仍只是预检且不创建结果文件。completed、退出码0不代表读取成功；本反例中读取失败才正确。判定重点是同样拒绝、原因更明确，而非把False变成True。assessment保持not_evaluated，人工质量判断另记。
