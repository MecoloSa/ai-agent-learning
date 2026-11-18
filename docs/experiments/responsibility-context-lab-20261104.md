# 模型职责与上下文：三项独立实验及组合修订

2026-11-04。用户授权测试侧自由实验，要求每项原理、输入、过程和结果可见。业务Python、生成提示、配置和预算没有落地更改；候选通过进程内patch运行。完整原始轨迹保留，失败不改成成功。

## 结论与来源

减负方向有价值，但“少给字段”不等于引用可靠，“只换结构化结果工具”也不等于职责分清。当前最确定的改进方向是程序展示真实回读原文与来源；其次是计划阶段使用适合自身职责的输入，而非继承全部聊天。阶段化候选尚未具备完整内容质量与自主补查验收，不直接接入生产。

参考已核对的[Deep Agents filesystem](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py)的大结果外置、预览和继续读取，以及[Pi compaction](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/compaction/compaction.ts)的会话记录与上下文投影分离。桌面教程物理页345—346用于工具输入/输出契约、503—504用于确认边界。程序原文展示块、四类计划结果是本项目实验设计，不归称为上述项目的标准方案。

## 冻结输入、执行顺序与判定

输入资料沿用learning_holdout_cases.json两份全文（见[上一轮输入与失败](learning-holdout-20261104.md)），不重新分析或生成初始计划。模型qwen3.7-plus、temperature=0.2；正式预算与空响应恢复不改。

- 实验1从验证集任务已采用的first.json恢复，正式修订号2；目标、计划、任务、反馈和三轮归档齐备。
- 实验2/3从相关因果任务首次候选前恢复：去掉失败的最后一问一答，归档授权由第2轮退回第1轮。正式修订号1，一项已完成但理解不清楚的任务，一条原始反馈。每次独立复制归档，新current文件，源快照不覆盖。
- 正例按原/候选/候选/原顺序各2次。实验2/3另各一条原/候选只讨论、原/候选时间冲突，共各8条。三个输出目录相互隔离，独立运行，不共享候选状态。
- 所有实际候选确认均输入n，不采用。初版阶段错误发生于交付之前时，不强行保存current；缺少快照的状态比较记unknown，不冒充true。

完整查询：

**历史回查**：“请复述当前生效计划及read-001的活动进度和理解反馈，并从来源1的历史中搜索‘今天最多20分钟’，读取命中的完整问答后逐字引用我的限制原话。不要修改任何计划、任务或反馈。”实际引号为中文双引号，精确值保存在manifest/input.json。

**候选**：“请依据已有反馈调整后续学习计划，总计最多20分钟，只阅读和口头说明，保留学习目标和任务反馈，提交完整候选计划供我确认。”

**只讨论**：“今天最多20分钟，只阅读和口头说明。请依据已记录的困难讨论如何学习，只讨论，不修改计划、任务和反馈。”

**时间冲突**：“这项学习活动必须连续观察至少2小时，不可压缩或分段；我今天只有20分钟，却希望今天完整完成。请判断是否可行，先讨论解决方式，不要修改计划、任务或反馈。”

观察字段：completed只表示实验流程结束；reviews表示真正进入程序确认；exact_quote只检查完整原话；expanded_citation要求交付含本轮真实回读的完整来源且没有核验警告。否定因果短语仅是审查线索，不是完整自动语义评分。保存状态的21条正式字段均不变；另外2条失败前没有生成current，无确认记录。

## 1. 工具回执中的模型可见字段

代码：tests/helpers/responsibility_lab.py::history_view。在app.invoke_llm_response之前只改深复制的ToolMessage正文，原执行结果仍交给主循环登记引用、权限校验与审计。

```python
removed = {"archive_id", "session_id", "turn_id", "user_message_id", "assistant_message_id", "citation"}
payload["data"] = {k: v for k, v in data.items() if k not in removed}
```

保留user_text、assistant_text、source_index、turn_number、citation_ref、truncated、recorded_at等，调用配对ID不动；失败或未知回执不变。只缩减历史回读字段，不声称清空全部请求里的程序元数据。

| 组别 | 精确原话 | 正确展开引用 | 平均累计输入 | 平均耗时 |
|---|---:|---:|---:|---:|
| 原版本 |2/2|2/2|25844 token|30.29秒|
| 隐藏长标识 |2/2|0/2|33623 token|25.39秒|

候选生成了`[citation: history-ref=1:1]`、`[citation_ref=history-ref=1:1]`，都不属于既有展开器的合法标记；“无未核实警告”不能当成功。没有扩大正则去默许错误拼写。

成本不能简单归因于视图：候选一次遇到真实空响应，原有有界恢复重发一次，额外8418输入token已包含在累计值；另一次多调用来源目录。34次逻辑调用共35次底层尝试。没有通过删除失败重算成本。

**1b确定性补验**：tests/helpers/history_evidence_panel.py从四条真实成功回读的原始工具轨迹构建独立“原始用户消息＋程序来源”块，不再请求模型。4/4逐字保留原话和来源；拒绝失败、未执行、截断回执，重复来源去重。不是让模型重答，也不认证其相邻论断。两个候选错误的自由文本保留，补验仅证明独立依据展示可由程序可靠完成。

采用判断：不采用“隐藏长ID即可解决复制”的结论。优先考虑独立依据展示；不能把整个模型回答中的任何引用都自动绑定到这个展示块。

## 2. 计划准备与提交分阶段

代码：responsibility_lab.py::PlanOutcome/run_worker。显式实验入口进入计划准备，不额外调用模型分类意图。绑定结果schema，允许：

```python
status: Literal["candidate", "clarification", "infeasible", "no_change"]
plan_markdown: str | None
explanation: str
```

candidate必须有正文；非candidate必须没有正文。程序经真实inbox.tool登记，成功才开放原确认；不是解析普通文字猜测是否生成了候选。只读资料已齐备，首版不提供自主补查，不能算完整通用计划执行器。

首版保留原上下文和大部分交互规则，只替换其中提交说明及结果工具。正例原版本2/2、阶段候选2/2提交；候选平均输入6183→3889，输出1167→1583，耗时19.78→26.12秒，不能宣称提速。

但只讨论、时间冲突两个阶段负例返回普通文本，没有返回唯一plan_outcome，阶段交付失败。程序没有确认。虽绑定了命名tool_choice，不能据此保证实际响应遵约；本轮没有HTTP原件，不能断言是提供方还是模型内部原因。原始公开响应、空tool_calls均已保存。

内容上原/候选正例都出现“资料为什么否定因果表述”等问题，分阶段没有清除旧讨论的错误。这一版仍混用了开放交互职责，不应把失败归为阶段化思想不可行。

辅助统计最初假定失败也有current，出现FileNotFoundError；已修复为缺失记unknown，原失败不重跑，只在独立tail目录继续未执行的两个冲突样本。汇总直接扫描全部observation，包含最初漏写汇总行的失败，没有遗漏付费请求。

## 3. 按阶段选上下文，保持原生成/工具流程

代码：responsibility_lab.py::phase_view。保留原system（含资料分析、目标及原计划）和程序生成的context_type动态消息；仅在请求副本去掉旧普通human/ai问答。正式状态、用户来源表、历史来源目录保留，本轮原问题由原循环追加。原记忆/归档仍完整，历史工具继续可用。不是摘要，不改工具集合或调用策略。

正例原版本2/2、候选2/2提交；单次输入6183→5489（约11.2%），输出均值1142.5→1155.5，耗时19.39→19.20秒，无明显提速证据。两组只讨论/冲突均未提交候选。

原组一条正例复用了“资料为什么否定因果”；候选两条未出现该原句，但一条仍把“季节可能影响两者”强化为确定混杂机制。不能据短语消失宣布语义正确。

冲突样本：本组原版先承认20分钟无法完成2小时，再澄清对象；候选也先判断不可行。实验2的原版冲突反而先以当前资料没有该要求为由否定用户前提，另排20分钟学习方案。可见旧状态可能压过当前用户新条件，且基线本身也波动。

采用判断：输入减负有直接证据，但仅覆盖无需旧聊天的任务；不能上线为“所有轮次一律丢旧聊天”。后续必须覆盖“沿用上个候选、改第二项、用户历史更正”等依赖旧聊天的任务。

## 组合修订：真正替换阶段职责，而非只换工具

在单项后追加三条，不混算为原AB成绩：stage_clean使用短的独立阶段system，保留system中目标/分析/原计划数据，保留正式动态状态；去掉旧闲聊与无关操作指令。只有plan_outcome结果工具，仍由原inbox校验、n拒绝。没有增加案例答案或针对因果问题的禁令。

| 输入 | 结构化结果 | 输入token | 输出token | 耗时 |
|---|---|---:|---:|---:|
| 请求候选 |candidate，确认后n拒绝|2242|1493|26.03秒|
| 只讨论 |no_change，无候选|2236|982|17.45秒|
| 时间冲突 |clarification，无候选|2262|974|16.40秒|

三个流程均完成，正式状态不变。正例输入比原版6183少约63.7%，但不是严格单变量：规则、历史和schema都不同，不能把收益归给某一项，也不能由一个正例称稳定解决漏提交。

质量未全过：新例子明确标为辅导者补充，给模型保留了教学发挥空间；但仍出现“要确认因果需要实验数据”。冲突回复仅要求澄清，遗漏“在用户给定前提下不可行”的直接回答。说明减少束缚有助职责与输入整洁，并不能替代语义能力和任务完成评审。

## 总体判断、落地状态与复验

23条在线观察，34次逻辑调用、35次底层尝试，累计输入224898、输出29938token；包含两个阶段失败与一次真实空响应恢复。69项相关离线测试通过，3项既有依赖警告，未重跑全量。没有将小样本混成项目成功率。

所有候选仍在tests。下一步优先方案：①程序独立展示本轮真实回读的原话与来源，解除精确复制职责；②为明确的计划阶段构造适用请求视图与结果契约，保留澄清/不可行分支及必要补查，不给全部聊天强行套用；③针对“旧助手建议污染新计划”补历史依赖边界，而非继续堆提示词。当前不把四种实验自动合并到业务，也不以格式通过掩盖事实强化。

原始输出目录：responsibility-e1-20261104、responsibility-e2-20261104、responsibility-e2-tail-20261104、responsibility-e3-20261104、responsibility-clean-20261104（均在.evaluation-results）。每条含input.json、source/current快照（失败可能无current）、observation.json、console.log、stage-results.json和readable-transcript.md。实验1另有program-evidence-panel.txt及panel-checks.json。总汇总位于responsibility-report-20261104/summary.json，包含两个阶段失败。

项目根目录只读重建报告，不发模型请求：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tests.helpers.responsibility_report --roots .evaluation-results/responsibility-e1-20261104 .evaluation-results/responsibility-e2-20261104 .evaluation-results/responsibility-e2-tail-20261104 .evaluation-results/responsibility-e3-20261104 .evaluation-results/responsibility-clean-20261104 --output-dir .evaluation-results/responsibility-report-20261104
```

预检入口：`python -m tests.helpers.responsibility_lab --experiment 1 --output-dir <新目录>`；experiment支持1/2/3及组合修订4。加--llm会真实调用并产生费用；现有结果足以作本轮判断，不必重复。

```powershell
$o = Get-Content .evaluation-results/responsibility-e1-20261104/2-history_view-positive/observation.json -Encoding UTF8 -Raw | ConvertFrom-Json
$o.input_events | Format-List stage,text
$o.models[-1].request_messages | Format-List role,content
$o.turns | Format-List user_input,response
$o.turns.audit_trace.tool_results | Format-List tool_name,args,result
# 原回执含完整编号，模型请求中的对应工具消息已去除；可直接对照。
```

离线复验：`python -m pytest tests/test_responsibility_lab.py tests/test_history_citation_guard.py tests/test_history_readback.py tests/test_plan_proposals.py -q -p no:cacheprovider --basetemp .evaluation-results/pytest-responsibility-new`，使用项目虚拟环境和新的临时目录，预期69通过。
