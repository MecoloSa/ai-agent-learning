# 历史压缩策略核对（2026-10-28）

## 结论与证据范围

保留单模型渐进摘要，不新增第二评审模型、分层摘要或记忆Agent。
单次摘要调用一个模型，与只用摘要保存全部记忆不是同一回事。本项目已将任务、正式反馈及更正关系独立持久化，
每轮重新组装当前状态；摘要负责聊天背景。两个模型也不能从已经丢失的原文恢复信息。
这是根据当前代码与来源形成的工程判断，不是单模型质量已被在线证明。

实际源码已应用采用预算与故障降级，关联177项离线回归通过。本次未请求在线模型。

## 桌面书籍

文件：C:/Users/21363/Desktop/agentic-ai-guide-zh.pdf。

- 第318页17.3.2：渐进式摘要使用旧摘要与新信息生成新摘要；同时介绍分层压缩，说明摘要有损，精确事实与用户原话宜保留原文。
- 第319—320页17.4.1：不同记忆粒度应结合，来源可追溯；有损摘要无法恢复原子事实。
- 第341—342页18.2：上下文预算还包括system、工具、历史与输出；旧轮次可由专用摘要模型压缩，另有选择性保留与分层方案。
- 第374页19.10：周期摘要、滑动窗口、重要性保留、外部记忆和子Agent隔离各有取舍，不能从架构更复杂推断效果更好。

书中有关固定空间的说明需要以摘要长度受控为条件；本项目失败回退会保留历史，不能宣称总空间或总请求长度恒定。

## 官方项目对照

以下是查阅当日的文档与主分支实现，不把示例配置当成本项目最优参数。

| 项目 | 实际机制 | 对本项目的意义 |
| --- | --- | --- |
| LangMem | 单个摘要模型更新RunningSummary，跟踪已摘要消息ID；分别配置触发、摘要及最终上下文预算 | 渐进摘要可保留，需承认字符阈值尚非完整token预算 |
| Deep Agents | 摘要与近期原文并用，被移出历史写入后端供回查；计数包含system及工具 | 更大的差距是原文可回查及总预算，不是模型数量 |
| LlamaIndex Memory | FIFO短期队列加静态、事实抽取、向量等记忆块 | 不同内容分层保存，不应只靠聊天摘要承担正式状态 |
| AutoGen | Buffered及TokenLimited上下文可直接限制消息视图 | 不采用摘要也是可比较基线，不必所有策略同时开启 |

来源：

- [LangMem API](https://langchain-ai.github.io/langmem/reference/short_term/)
- [LangMem摘要源码](https://github.com/langchain-ai/langmem/blob/main/src/langmem/short_term/summarization.py)
- [Deep Agents摘要源码](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/summarization.py)
- [LlamaIndex Memory文档](https://github.com/run-llama/llama_index/blob/main/docs/src/content/docs/framework/module_guides/deploying/agents/memory.mdx)
- [AutoGen上下文参考](https://microsoft.github.io/autogen/stable/reference/python/autogen_core.model_context.html)

LlamaIndex旧ChatSummaryMemoryBuffer已标为弃用，不将其作为新迁移目标。
LangMem默认以system消息注入摘要，Deep Agents使用HumanMessage；角色处理并不统一，不能照抄视作安全或效果保证。

## 当前实现审计

保留：旧摘要与新增历史分隔、明确更正及未改条件、冲突不猜、程序确认结果优先、原引用不补造、
长度程序校验、失败保留数据、正式状态独立保存。无需为对标项目而扩写为庞大的通用提示词。

局部提示词提案（未修改业务源码）：

1. “不重复系统中已有的当前目标和计划”缺少实际输入前提：摘要模型并未收到业务system或当前状态，只有摘要专用system及待压缩数据。
   改为不重抄整份计划，同时保留理解历史决定所需的目标、对象与理由。
2. 在原有用户意图保留规则中明确约束优先，不增加案例数值、文件名或标准答案。

内存候选的相关105项回归通过；不代表真实模型的遗漏率已改善。无新增案例、无额外请求。

仍有差距：成功摘要后普通聊天原文不完整保留在当前存档；旧存档、console日志不是正式可查询的完整原文库。
仅保留最近一问一答可能对指代过于激进；字符阈值不统计全部system、工具和摘要；变更提示也可让摘要字段超过采用上限。
摘要以SystemMessage注入，虽然主提示词已声明摘要仅作背景，仍非程序级可信隔离。

## 下一决策依据

先做相同多轮输入的full/window/summary对照，报告更正、限制、指代、确认结果、恢复、耗时及额外摘要请求。
出现原文追问无法核查时，优先讨论原文归档与受控回读；出现指代丢失时比较近期保留范围；
出现摘要遗漏时先区分输入丢失、摘要错误、下游未使用三类原因，再决定独立模型离线评审或受限重试。
不把默认增加第二模型当作验收方式。
