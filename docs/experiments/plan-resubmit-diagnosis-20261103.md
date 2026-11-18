# 拒绝后再次生成候选：诊断、对照与采用边界

## 结论

原失败是：用户再次明确请求完整待确认候选，但模型只输出普通计划文本，没有调用 `propose_plan_change`。程序没有 pending，因此没有确认，也没有写状态。原实验严格等待确认输入，故报“需要confirm，实际chat”；不是JSON解析、工具执行报错或保存失败。

本轮基线首次请求消息与原失败请求的9条消息逐项完全相同（包含role、正文及已有记录字段），两次重放均调用工具。可以确认这是可变的工具选择结果，不能把它解释为程序必然丢失拒绝状态。不能由公共响应推断模型内部为何作此选择，也没有提供方内部证据。

确定发现并修复另一处接口缺陷：`main._context_change_history` 的结构化回执只有 reason/note，消费者 `read_confirmation_history` 需要的 decision/context_revision 缺失。结果虽出现在展示文字中，查询工具却返回空值。现在由生产者直接写入程序已知值；旧记录保持未知，不从旧文字补造。这项修复不是漏调工具的已证实根因修复。

## 初始状态与完整输入

使用 `.evaluation-results/closed-loop-fixed-20261103/literature` 中真实合成学习闭环的产物。目标为“区分叙述视角与人物态度”；正式修订号1，无20分钟限制；read-001已完成，理解状态self_reported_unclear，正式反馈“阅读已完成，但我仍分不清两个概念。”。原候选为8+7+5分钟，用户已输入n。

从first.json去掉最后失败的一问一答；原文归档授权前缀从3退回2，避免读取未来。每次独立复制归档目录，使用新的current快照；模型和资料不重新生成。沿用原analysis/context。

再次提交原文：

> 请依据已有反馈调整后续学习计划，总计最多20分钟，只阅读和口头说明，保留学习目标和任务反馈，给出完整候选计划供我确认。

讨论边界原文：

> 今天最多20分钟，只阅读和口头说明。请依据已记录的困难讨论如何学习，只讨论，不修改计划、任务和反馈。

维持拒绝原文：

> 上个候选仍然不采用，请解释它与当前正式计划的区别，不要生成新候选，不修改计划、任务或反馈。

自动输入顺序：chat上述提问→如果程序出现确认则输入n→chat exit。观察器不预设必须出现确认，不把漏提交伪装成脚本成功；分别统计reviews及状态。候选请求预期一次确认，其他两项预期零次；全部正式状态不变。内容质量另看实际回答，确认次数不是学习效果评分。

## 四种处理及依据

- baseline：保留原请求、schema、旧回执。
- contract：仅在测试侧替换工具说明，明确“提交正文→程序展示确认→pending未采用”，给plan_markdown增加完整正文的参数描述；不修改全局提示、执行逻辑或权限。
- receipt：仅给历史快照里已有拒绝回执补decision=cancelled、context_revision=1，其余保持。模拟新生产者的正确结构，不迁移真实旧存档。
- projection：仅修改发给模型的消息副本，将已拒绝的长候选替换成短JSON拒绝回执和read_confirmation_history查询入口；原始记忆、归档不改，允许模型按需查询。此为请求视图探索，不是生产压缩策略。

参考桌面《智能体AI漫游指南》PDF物理页345—346的动态消息组件、工具名称/使用时机/参数/输出说明；物理页503—504的人机审批与反馈。这里只借鉴设计原则，不把书中的通用效果数字当成本项目收益。

参考[LangChain HITL官方文档](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)：提议、批准、拒绝是不同结果；reject跳过执行并把反馈交回agent。本项目保留既有本地确认事务，没有为此迁移框架。工具选择自由不等于自动获得写入授权。

## 实测

实际配置：qwen3.7-plus，temperature=0.2，timeout=180秒，LLM_RETRY_ATTEMPTS=3，交互工具总预算8，连续无进展上限4。没有修改模型或预算。

| 组别 | 再次请求候选：进入确认/运行数 | 每次平均输入token | 每次平均输出token |
|---|---:|---:|---:|
| 原版本 | 2/2 | 6452 | 976.5 |
| 工具契约 | 1/2 | 6493 | 904.5 |
| 结构化回执 | 1/2 | 6465 | 1189.5 |
| 请求视图收起旧候选 | 3/3 | 5790 | 940.3 |

契约组与视图组各测讨论、维持拒绝，共4条边界，无候选误提交。13次独立会话合计14次模型生成，实际usage累计输入87348、输出12587token；不是费用估算，没有逐调用耗时采集，不能声称速度提升。

所有13次正式业务状态均保持不变；9次正例中进入确认的7次全部输入n，实际拒绝未写入。所有14次生成有非空内容或正常工具请求；本轮不是空响应问题。

视图正例的单次输入减少662token，约10.3%；但解释旧候选那次需要先read_confirmation_history，再生成回答，累计12408输入token、2次请求。确实完成按需查询，没有生成新候选或采用旧候选。其依据为候选识别摘录（truncated=true），本次所比较的标题、时间与问题在返回摘录中；不能当成全文恢复能力验收。

人工抽查：原版本成功候选5+8+7分钟，目标和活动形式符合要求；视图比较回答正确区分已拒绝候选和当前正式计划。但将正式计划称为“未明确限定学习方式”应理解为没有明确排他限制，正式计划已有阅读/口头活动，不宜理解为未指定任何方式。未做盲评，不把这些观察记为自动质量通过。

## 采用与暂缓

已采用：结构化确认回执补全；新增5种真实生产者→消费者往返用例，以及旧回执缺字段保持未知的测试。确认、取消、输入结束、invalid、unchanged均覆盖。

未采用：工具契约替换与候选正文收起。前者未显示稳定收益；后者只有3次正例且基线本轮2/2，不能归因为提升了可靠性，还增加旧候选查询成本。保留影子入口与失败，停止继续刷同一题。

若产品要求“明确发起计划修改就必须进入候选生成流程”，下一项架构方案是显式计划动作入口（用户/界面选择计划模式→模型准备完整正文/必要取证→校验→人工确认），使流程选择由程序承载；开放聊天仍自由路由。这涉及交互入口和工具编排，需单独说明、比较和验收，不以关键词硬判意图或解析普通文本假装工具成功。

## 验证与复验

确认历史相关19通过；全量1695通过、1项环境跳过、3项依赖弃用警告，57.43秒。首次pytest默认临时目录拒绝访问，改用项目内独立basetemp后通过，未改断言。

执行目录：项目根目录。离线复验需使用未存在的临时目录：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_confirmation_history.py tests/test_plan_proposals.py -q -p no:cacheprovider --basetemp .evaluation-results/pytest-plan-recheck
```

以下入口会真实调用模型、产生费用，不是预检；每个输出目录必须是新目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tests.helpers.plan_resubmit_probe --output-dir .evaluation-results/resubmit-recheck --repeats 2
.\.venv\Scripts\python.exe -X utf8 -m tests.helpers.plan_resubmit_probe --output-dir .evaluation-results/projection-recheck --variants projection --boundary-variant projection --repeats 3
```

完整输入/请求/工具与回答：每个子目录input.json、observation.json、console.log。自动汇总results.json，不能仅看process_code=0；正例还须reviews=1，负例reviews=0，unchanged=true。

```powershell
$o = Get-Content .evaluation-results/plan-resubmit-projection-20261103/projection-keep_rejected-1/observation.json -Encoding UTF8 -Raw | ConvertFrom-Json
$o.input_events | Format-List stage,text
$o.models[0].request_messages | Format-List role,content
$o.turns | Format-List user_input,response
$o.turns.audit_trace.tool_results | Format-List tool_name,args,result
```

聚合原始用量及结果：`.evaluation-results/plan-resubmit-diagnosis-20261103/summary.json`；全量日志：同目录pytest-full.log。
