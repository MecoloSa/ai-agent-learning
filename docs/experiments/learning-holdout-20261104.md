# 两条新任务的完整学习闭环验收（2026-11-04）

## 目的与结论

不沿用前轮文学/停止条件题调参，冻结当前生成策略，检查反馈、候选、确认、恢复和证据利用能否协同。新增资料未用于此前提示词或请求视图优化。此次只改tests及实验文档，没有修改业务执行、模型、提示词或预算。

结果：验证集/测试集完成流程控制，但存在事实过度强化、恢复引用标识错误；相关/因果在第一次候选请求就漏调提交工具，未进入确认。不能记为“两条质量通过”。发现实质失败后停止重复，不换措辞补跑到成功。

## 输入文件与完整交互

资料全文见`tests/fixtures/evaluation/learning_holdout_cases.json`，运行前已在对话公开。运行时分别复制为独立materials/notes.md，以下为全文：

验证集/测试集：

> 验证集用于开发过程中比较候选模型、选择超参数。测试集用于模型与超参数确定后的最终评价，不参与调参。若反复根据测试结果选择模型，测试结果就可能不再反映对未见数据的独立评价。可通过阅读、比较使用场景并口头解释来学习。

相关/因果：

> 相关关系表示两个变量存在统计上的共同变化，单凭相关不能确认因果关系。混杂因素可能同时影响两个变量。观察到夏季冰淇淋销量与溺水事件同时增多，不能据此认定冰淇淋导致溺水；季节及活动变化是需要考虑的因素。本资料未提供实验数据，不足以判定该例的具体因果效应。可阅读例子、比较解释并口头说明证据边界。

初始学习目标分别是“区分验证集与测试集”“区分相关关系与因果关系”。实际初始规划请求为目标后附：

> 请依据资料安排一天的学习，帮助我理解这两个概念的区别。活动采用阅读和口头说明，给出可检查的完成标准。

真实分析/规划生成初始计划后构建LearningState，修订号1，无任务与反馈；两条任务独立初始化，内部连续对话。自动序列：

1. `/task {"task_id":"read-001","title":"阅读资料并口头比较"}`
2. `/feedback {"task_id":"read-001","text":"阅读已经完成，但我仍分不清这两个概念在实际例子中的区别。","activity_status":"completed","understanding_status":"self_reported_unclear"}`
3. “今天最多20分钟，只阅读和口头说明。请依据已记录的困难讨论如何学习，只讨论，不修改计划、任务和反馈。”
4. “请依据已有反馈调整后续学习计划，总计最多20分钟，只阅读和口头说明，保留学习目标和任务反馈，提交完整候选计划供我确认。”
5. 在实际确认提示输入n。
6. 再次发送第4步原文。
7. 在实际确认提示输入y。
8. exit；以保存状态在独立新进程恢复。
9. “请复述当前生效计划及read-001的活动进度和理解反馈，并从来源1的历史中搜索‘今天最多20分钟’，读取命中的完整问答后逐字引用我的限制原话。不要修改任何计划、任务或反馈。”实际输入的引号为中文双引号，精确版本在manifest.json。
10. exit。

输入按chat/confirm阶段严格消费；若本应确认却回到聊天则终止该链，不把n/y误送成用户聊天，也不补造候选。另一条独立任务继续执行。

## 执行来源及关键过程

`tests/helpers/learning_holdout.py`负责声明输入与调用；复用`learning_closed_loop.run_case`、worker及launch：

```python
analysis, plan, context = agents.run_learning_assistant(
    str(materials), case.get("planning_request", case["goal"]), 1, "agent"
)
```

资料分析使用真实工具，初始规划没有预制答案。交互复用`main.run_interactive_session`与`_invoke_interactive_turn`；模型调用仅加公开请求、响应、usage及耗时侧录，不替换其决策。

正式反馈复用`learning/learning_commands.py`的prepare/commit路径：

```python
pending = prepare_feedback_proposal(
    state,
    FeedbackProposalArgs(scope="task", task_id=feedback.task_id,
        activity_status=feedback.activity_status,
        understanding_status=feedback.understanding_status),
    user_message=feedback.text,
)
updated = commit_feedback_proposal(state, pending)
```

计划由`learning/plan_proposals.py::PlanProposalInbox`登记本轮pending，`main._review_context_change`确认后才更新。只有自然语言计划正文而没有实际提交时，程序没有pending，不进入确认；本轮不加文本自动解析回退。

恢复回查走search_conversation_turns→read_conversation_turn。`main._sanitize_history_citations`核对本轮已成功回读的引用，不能从“原话字符串正确”推导“模型引用编号也正确”。

新增每次输入前的落盘状态观察，用来直接比较只讨论前后；原有formal比较排除聊天记忆，保留目标、计划、任务、反馈等业务字段。初始任务登记和反馈是预期状态变化，不能要求整场业务状态完全不变。

## 观察字段与结果

- status=completed：脚本阶段执行结束，不是内容质量通过。
- reviews：真实程序确认处理；cancelled应前后相同，confirmed应修订号仅加1。
- quote_in_answer：完整用户原话出现在交付中，只检查文字，不检查引用正确。
- source_unchanged：运行期间业务Python源码指纹不变；resume_source_unchanged另检查恢复源快照字节未变。
- history_citation_warning：交付包含未核实引用警告，不能被quote_in_answer=true掩盖。

| 任务 | 逻辑模型调用 | 输入token | 输出token | 整链耗时 | 控制结果 |
|---|---:|---:|---:|---:|---|
| 验证集/测试集 |10|49325|6720|123.83秒|两次提交、n拒绝/y采用、恢复完成|
| 相关/因果 |6|16245|5087|90.73秒|首次候选漏提交，停止，未执行后续确认/恢复|

合计16次逻辑调用、65570输入token、11807输出token。两条都只讨论前后状态相同，运行期间业务源码不变。调用阶段已采集elapsed_ms；整链时间还包括进程与保存，不将token换算为费用。

验证集链：两次实际propose_plan_change；确认结果cancelled 1→1、confirmed 1→2；一项任务已完成/理解不清楚，一条正式反馈保留；恢复业务状态及来源快照不变。恢复实际调用历史搜索、回读，原话逐字正确，计划与反馈也完整交付。

相关/因果链：首次候选回复为10+7+3分钟，活动符合阅读/口头说明，但tool_calls为空，reviews为空；错误“输入阶段不匹配：需要confirm，实际chat”。状态修订号仍1，一项任务、一条反馈；与记录反馈后业务状态完全相同。此次失败发生在第一次候选，不仅限于拒绝后的再次请求。

## 助手逐条内容审查（不是独立人工金标或自动judge）

1. **规则强化**：验证集链把“不参与调参、反复选模型有风险”强化为“测试集只能用一次”，还列为完成标准。资料没有规定次数；属于未获支持的绝对化要求，不只是额外教学例子。初始计划也列出可选书面回答，与限定活动形式存在偏差。
2. **证据不足被说成否定**：相关/因果讨论把“不能据此认定”写成“资料否定因果表述”，并据此设计检查要求。原材料只限制可推断范围，没有证明不存在该因果关系。
3. **缺失项被升格为必要条件**：因果候选写“需要实验数据来确认因果关系”；原材料只说此处未提供实验数据且不足判定，未确立一般必要条件。应区分资料的不足与通用方法论要求。
4. **原话正确但编号复制错误**：验证集恢复模型把正确session_id的一部分复制成archive_id片段。已有核验程序将其标成“历史引用未核实”，并列出真实回读来源；保护生效，但引用交付未达标。工具还返回了可用的`[history-ref=1:1]`，模型没有选用；不据此新增一套重复引用机制。
5. **格式警告**：初始计划使用“任务一/任务二”或“完成标准”，结构检查器不能完整识别，但实际存在任务与检查项。属于观察器覆盖问题，不代表业务执行被阻断；不强制模型改排版来迎合解析器。

两条都不能判内容质量整体通过，故原始quality_status=pending_manual_review保留；本节及独立assistant-quality-review.json补充助手审查结论，不覆盖原始轨迹。

## 修改与验证范围

新增tests/fixtures/evaluation/learning_holdout_cases.json、tests/helpers/learning_holdout.py、learning_holdout_report.py、tests/test_learning_holdout.py；复用脚本增加可选事件/初始请求参数、状态与耗时侧录，旧调用保持默认行为。

辅助单测检查预检不发模型请求、两次确认顺序、失败不能掩盖第二条任务及非零退出。在线首次执行时入口只完成汇总、失败未传递进程码；观察后已在辅助入口修复，未重跑付费实验。以后任一流程/控制失败退出1，退出0也不表示质量通过。

预检已执行；相关离线42项通过、3项既有依赖弃用警告。本轮没有重复全量，不能沿用上轮1695称作本次全量成绩。

下一步优先分析“内容生成正确性”与“动作提交可靠性”的职责划分。显式计划入口只能改善动作路径，不能消除上面的事实强化，不能当作全部问题的修复；不继续给提示词添加这两题的答案。新业务方案另行说明和确认。

## 产物及命令

执行目录为项目根目录；原始产物`.evaluation-results/learning-holdout-online-20261104`，两条子目录各有真实materials/notes.md、初始工作流与计划、逐轮观察、快照和控制台；第一条另有恢复观察。readable-transcript.md提供完整输入、回答与工具结果，summary.json为派生统计。

```powershell
# 只读派生报告，不调用模型
.\.venv\Scripts\python.exe -X utf8 -m tests.helpers.learning_holdout_report .evaluation-results/learning-holdout-online-20261104
# 离线回归，临时目录使用新名字
.\.venv\Scripts\python.exe -m pytest tests/test_learning_holdout.py tests/test_confirmation_history.py tests/test_plan_proposals.py -q -p no:cacheprovider --basetemp .evaluation-results/pytest-holdout-recheck
# 预检，不调用模型；输出目录需不存在
.\.venv\Scripts\python.exe -X utf8 -m tests.helpers.learning_holdout --output-dir .evaluation-results/holdout-preflight-new
# 真实模型整链重跑，会产生费用；本轮不需要重复
.\.venv\Scripts\python.exe -X utf8 -m tests.helpers.learning_holdout --output-dir .evaluation-results/holdout-online-new --llm
```

```powershell
$o = Get-Content .evaluation-results/learning-holdout-online-20261104/correlation_causation/first-observation.json -Raw -Encoding UTF8 | ConvertFrom-Json
$o.input_events | Format-List stage,text
$o.turns | Format-List user_input,response
$o.models[-1].request_messages | Format-List role,content
$o.models[-1] | Format-List content,tool_calls,usage,elapsed_ms
```
