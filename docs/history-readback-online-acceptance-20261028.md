# 下一步：真实模型回查验收

本轮核实上一轮3个业务文件已落地，78项核心测试实际通过。此前替身模型已验证保存、来源范围、工具路由和回传，但尚不能证明真实模型正确选择工具及引用原话。先在线验收再决定搜索与提示词修改；本轮不需要业务代码或提示词候选，避免在没有实际失败记录时继续扩写功能。

直接准备的测试文件：

- tests/fixtures/evaluation/history_readback_online_cases.json：3个公开开发案例，不是留出集。
- tests/helpers/history_readback_online.py：预检、独立初始化、真实模型调用、消息与状态观察。
- tests/test_history_readback_online_helper.py：8项离线测试，全部使用替身模型验证脚本，不调用网络。

## 初始状态与归档原文

每个案例独立建立自己的source.json、归档数据库及current.json，均位于.evaluation-results/history-readback-online-artifacts/随机run_id/。不从上一案例恢复，也不修改个人存档。

归档第一轮用户原文：

> 这轮练习名为松针练习。今天最多20分钟，只能阅读与口头说明，不绘图、不编程。这些只是本轮对话条件，不要登记任务或正式反馈。

第一轮助手原文：

> 了解，这是本轮对话中的练习条件，不写入正式记录。

归档第二轮用户原文：

> 更正：今天最多35分钟，其他条件保持。这里只更正对话条件，不修改任何计划或记录。

第二轮助手原文：

> 了解，可用时间改为35分钟，其余条件保持，不修改正式记录。

三例均以summary策略恢复，近期messages为空，只提供摘要“此前讨论过练习安排；这里没有保留具体条件和逐字原话。”。正式目标“核对历史原话”，正式计划“原计划，保持不变。”，无任务或反馈。这里直接构造压缩后的状态，验证丢失原话后的回查，不重新评价压缩质量。

实际模型初始消息不包含fixture_history、case_spec或评分要求；这些仅在预检与本地观察文件中提供给人查看。只有成功工具回读后，目标原话才应进入模型消息。

## 三个完整案例

### 001：已知位置

快照关联上限2，数据库实际2轮。自动输入：

> 请实际调用 read_conversation_turn，读取来源1第2轮，逐字引用用户更正时间的完整原话，并保留历史引用。不要修改计划、任务或反馈。

随后输入exit。预期实际读取source_index=1、turn_number=2，逐字引用第二轮用户原话，沿用返回的history引用；不修改正式状态。

原理：指定工具和位置，先验证真实API的schema兼容、参数生成、执行校验与引文。这一例不检验未知位置搜索能力。

### 002：先发现来源

独立初始化，上限2，数据库2轮。自动输入：

> 请先查看当前快照关联了哪些历史来源，再读取其中第2轮，逐字引用用户更正时间的完整原话，并保留历史引用。不要修改计划、任务或反馈。

随后exit。预期先list_conversation_sources成功，再read_conversation_turn读取来源1第2轮；最终准确引用。来源列表只有元数据，不能据此声称已经获得原话。

原理：不点名工具，观察模型能否依据职责描述选择两种工具、把第一次结果用于第二次调用。仍明确告知第2轮，不把本例称为自由搜索历史。

### 003：文件存在，但快照范围不允许

独立初始化，数据库仍有2轮，但快照上限为1。自动输入：

> 请实际调用 read_conversation_turn，读取来源1第2轮，并逐字引用用户更正时间的完整原话。不要修改计划、任务或反馈。

随后exit。预期请求被turn_out_of_scope拒绝；不得引用第2轮原话或声称请求已成功。

原理：用“物理上存在、当前快照不可读”的反例检验范围保护，不用空库代替。当前主流程对明确要求执行但没有成功结果的工具会统一生成失败说明，所以最终回答可能是“未取得可确认的成功结果”；不要求它一定保留模型生成的具体范围解释。可在audit_trace.model_responses查看模型回复，和最终保存消息区分。

所有案例若出现意外确认，脚本自动输入n并记录。状态保持不变并不能掩盖模型提出了不应出现的候选，unexpected_confirmation_count应单独检查。

## 执行流程与关键代码

原离线演示通过fake_model固定发出调用；本脚本保留真实create_llm和create_llm_with_tools，只观察实际调用：

```python
original_model = app.invoke_llm_response
```

observed_model先保存本次消息，随后真实执行：

```python
return original_model(llm, messages, stage=stage)
```

这个包装器增加观察，不提供固定回复。另一个包装器把audit_trace传入原_invoke_interactive_turn，继续由原工具循环选择、执行与回传工具。

程序输入被替换为案例问题和exit；任何意外确认输入n。RAG在该隔离案例中关闭，材料目录只有一份合成说明，避免资料检索混入历史回查；local历史工具照常执行。该隔离条件写进manifest，不声称代表所有产品配置。

结果分别记录：

- turn_answer：工具循环返回的文本。
- final_answer：主流程清理、确认处理后实际保存的助手消息。
- business_state_unchanged：正式状态不变，聊天记忆和归档允许增长。
- source_unchanged：来源快照与父归档文件字节未变。
- inherited_scope_unchanged：新快照仍保留来源的原始上限。
- model_requests：每次观察到的消息列表；不记录模型私有思维链。
- audit_trace：模型公开响应、实际工具参数、结果及最终循环输出。

observations中的匹配与计数只报告可观察事实，不自动判定自然语言质量，assessment保持not_evaluated。真实使用量可查看audit_trace.model_responses中的usage；缺失不是0，观察边界也不保证涵盖SDK内部重试的完整费用。

## 执行命令

执行目录：F:\LLM\projects\agent。

先检查脚本：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_history_readback_online_helper.py tests/test_history_readback.py tests/test_history_archive.py tests/test_history_session_integration.py -q -p no:cacheprovider --basetemp tmp/history-readback-online-helper-applied-01
```

预期86 passed。这里仍是离线测试。

然后预检001：

```powershell
.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_readback_online.py --case-id history-readback-online-001
```

预期mode=preflight_only，完整展示原文、问题、初始范围与自动输入。预检不请求模型、不创建结果文件，不能在此之后就读取JSONL。

确认输入后，依次运行三个独立案例（只增加--llm才真实请求模型）：

```powershell
.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_readback_online.py --case-id history-readback-online-001 --output .evaluation-results/history-readback-online-v1.jsonl --llm
.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_readback_online.py --case-id history-readback-online-002 --output .evaluation-results/history-readback-online-v1.jsonl --llm
.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_readback_online.py --case-id history-readback-online-003 --output .evaluation-results/history-readback-online-v1.jsonl --llm
```

每次结果追加一行，run_id与独立工件路径不同。退出码0仅表示脚本完成及正式状态、来源保护检查通过；不能证明工具选择、逐字引用或回答内容达标。执行后可立即用$LASTEXITCODE查看。

读取最近一次001（追加文件可能含多次运行，不假定只有1行）：

```powershell
$rows = @(Get-Content -LiteralPath '.evaluation-results/history-readback-online-v1.jsonl' -Encoding UTF8 -ErrorAction Stop | ForEach-Object { $_ | ConvertFrom-Json -ErrorAction Stop })
$r = $rows | Where-Object case_id -eq 'history-readback-online-001' | Select-Object -Last 1
$r | Format-List case_id, status, error_type, business_state_unchanged, source_unchanged, inherited_scope_unchanged, assessment, final_answer, observation_path
$r.observations | Format-List
$o = Get-Content -LiteralPath $r.observation_path -Encoding UTF8 -Raw -ErrorAction Stop | ConvertFrom-Json
$o.user_input
$o.model_requests[0].request_messages | Format-List role, content
$o.audit_trace.tool_results | Format-List tool_name, executed, args, result
$o.model_requests[-1].request_messages | Format-List role, content
```

把case_id改成002、003即可查看其他案例。若想看工具结果中的ok：

```powershell
$o.audit_trace.tool_results | ForEach-Object {
    $result = $_.result | ConvertFrom-Json
    [pscustomobject]@{tool=$_.tool_name; executed=$_.executed; ok=$result.ok; error=$result.error.code}
} | Format-Table -AutoSize
```

## 判定边界与后续决策

| 观察 | 001/002预期 | 003预期 |
| --- | --- | --- |
| status | completed | completed，范围拒绝是正常业务结果 |
| 三项状态/来源保护 | true | true |
| first_request_contains_target_quote | false | false |
| requested_read_succeeded | true | false |
| scope_denial_count | 通常0 | 至少1 |
| exact_target_quote_in_answer | true，仍需看是否标成用户原话 | false |
| returned_citation_in_answer | true，仍需看引用对应的轮次 | 不单独作为通过条件 |
| successful_source_list_count | 001不强求；002至少1且在读原文前 | 不强求 |
| unexpected_confirmation_count | 0 | 0 |

executed=true只代表进入调用，必须同时看result.ok和实际args；调用了正确工具名也可能读错轮次。exact_target_quote_in_answer只是完整子串出现，不证明整段解释忠实、无额外编造。

若001/002漏调用或参数错误，先依据原始schema、args和错误回传诊断；不要直接增加预算。若准确读取但引文或身份错误，先检查提示职责和输出，不重做存储。若003越界读成功或来源被修改，属于程序边界问题，优先修复。若三例均达标，再增加“提供关键词、未知轮次”的定位能力，以当前按轮次读取作为基线；向量检索等到字面检索的实际缺口出现后再评估。

后续已读取用户执行的3条真实在线结果：001直接读取、002先发现后读取、003范围拒绝均符合本例目标。模型请求、工具参数及最终文本已人工核对，详情见[结果分析](experiments/history-readback-online-review-20261028.md)。每例仅一次，原结果assessment仍保留not_evaluated，人工结论另记，不改写原始记录。
