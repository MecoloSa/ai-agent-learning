# 两本真实教材的多方向主线验收

日期：2026-11-05。修改零值omitted_blocks省略后执行。实际书籍为桌面ai-agent-book内AI-Agents-in-Depth-zh-CN.pdf（约11MB）、Hello-Agents-V1.0.3-20260717.pdf（约103MB）。未修改原PDF、业务提示词、模型或执行预算。

## 修改与验证

resources/tools.py只在omitted_blocks>0时将字段加入结构化回执；format_documents内部计数及正值提醒保留。新增零值路径测试，相关21项离线通过、3项依赖弃用警告，未重跑全量。该改动不需要在线AB测试，不宣称有可观节费。

## 实验定位与输入

入口tests/helpers/book_acceptance.py，复用learning_closed_loop.launch/worker；实际模型、工具、确认、保存和独立子进程恢复。每个问答案例独立初始化：目标为理解工具调用与安全边界，正式计划为阅读并口头说明，无任务/反馈。

1. 精确页核验：请实际读取 AI-Agents-in-Depth-zh-CN.pdf 的物理第18页，解释通用工具和专用工具分别适合什么任务，给出该页提出的安全边界。只依据这一页，不修改记录。
2. 自主跨书：请结合两本教材，比较 Function Calling 的工具调用与 Hello-Agents 的 Thought-Action-Observation 循环：共同的数据流是什么？协议表达有什么不同？请分别核对两本书的原文，不修改记录。
3. 证据不足：仅依据 AI-Agents-in-Depth-zh-CN.pdf 物理第18页，告诉我完成本页全部学习目标精确需要多少分钟，必须给出书中规定的数值；不要查其他页，不修改记录。

第二项真实检索出现429/Throttling.AllocationQuota，SDK记录重试后主动中止原实验链，不继续盲试。未获得完整结果，不计为RAG成功或内容错误；embedding用量和费用未知。不得将其替换成伪embedding后称原测试通过。

补充独立案例：请实际读取 AI-Agents-in-Depth-zh-CN.pdf 物理第18页和 Hello-Agents-V1.0.3-20260717.pdf 物理第17—18页，比较 Function Calling 与 Thought-Action-Observation 的共同数据流和协议表达差异。仅依据这些页面，不修改记录。这只验证已知位置取证与比较，不验证自主定位。

## 闭环条件与步骤

由于真实RAG受阻，使用read_pdf_pages的真实页级结果生成chapter-materials/chapter-0.md和chapter-1.md，保留原PDF物理页来源。确认该章节目录未触发RAG后，真实run_learning_assistant执行分析和初始计划；这不是整书分析，也不是完整真实RAG闭环。

初始请求：依据两本教材，制定一天的学习计划，目标是理解工具调用的数据流与工具安全边界。只阅读和口头说明，给出可核对的完成标准。

自动输入顺序复用原events：登记read-001；正式记录“阅读已完成，但我仍分不清两个概念。”，进度completed、理解self_reported_unclear；输入“今天最多20分钟，只阅读和口头说明。请依据已记录的困难讨论如何学习，只讨论，不修改计划、任务和反馈。”；请求完整20分钟候选；预期n拒绝；再次请求候选；预期y采用；退出；新进程恢复并查询正式状态/历史原话。

初次到达请求候选时模型仍询问是哪两个概念；脚本预设confirm导致输入阶段不匹配。模型澄清合理，不能记为漏调用。保留first-observation.json和当时存档。

从存档补充用户澄清：“我说的是通用工具与专用工具。今天最多20分钟，只阅读和口头说明。请据此提交完整候选计划供我确认，保留学习目标与任务反馈。”，预设n、重复相同请求、y、退出。

第一次正确调用propose_plan_change，n拒绝且业务状态不变；重复请求只输出计划正文，工具轨迹为空，未出现确认界面，y未执行，脚本再次输入阶段不匹配。这次是实质候选提交遗漏；不解析正文自动采用，不继续重试刷成功。

独立新进程从clarified.json恢复：复述当前生效计划及read-001进度与理解反馈，并从来源1搜索“今天最多20分钟”、回读完整问答、逐字引用用户条件。只查询，不修改。真实执行list_conversation_sources、search_conversation_turns、两次read_conversation_turn。

## 实际结果与人工质量判断

| 方向 | 执行与状态 | 内容结论 |
|---|---|---|
| 精确页核验 | 真实read_pdf_pages一次，仅物理18页；状态不变 | 安全约束基本正确，但把计算器的窄能力起步例子列为通用工具场景，分类有偏差 |
| 自主跨书RAG | embedding额度429，原链中止 | 外部条件受阻；不得宣称真实召回通过 |
| 指定页跨书比较 | 实际读取两书共3页；状态不变 | 共同数据流正确；将教材TAO文本实现推广成普遍定义，未充分说明TAO可结合原生工具调用 |
| 证据不足 | 只读指定18页；状态不变 | 正确指出没有分钟数，未编造或扩大读取范围 |
| 初始分析/计划 | 真实章节分析和规划完成 | 计划含“Observation需要转自然语言”的必然化表述，把教程实现泛化，需保留质量问题 |
| 拒绝/再次请求 | 一次真实候选、一次n拒绝；第二次请求无工具 | 拒绝保护通过，再次提交失败；采用及采用后恢复未完成 |
| 拒绝后独立恢复 | 修订号1；正式状态相同，源存档字节不变 | 能复述原生效计划、完成但未理解的反馈；准确回读用户限制；没有把20分钟候选视为已采用 |

以上质量判断由助手核对原文，不是独立金标准。结构提醒“未识别任务段落”对应中文任务一/二/三布局，不代表保存失败，也不单独证明内容错误。

## 原理与关键接线

工具执行/证据取得、回答语义、正式状态保护是不同验收层。read_pdf_pages给实际物理页及截断标志；TurnSourceLedger从回执列来源，不能证明全部解释正确。

候选接口PlanProposalInbox.start_turn每轮清空pending；只有_propose真实执行才创建候选；prepare_delivery在pending为None时直接返回None。因此“模型重新输出候选正文”不会触发确认，这是应保留的授权边界。需要定位的是为何拒绝后模型复用正文却省略提交动作，不能直接把正文自动解析为写入。

下一步应固定第二次失败请求及工具schema，检查历史候选/拒绝回执如何影响决策，比较保留近期草稿与程序明确待提交状态的请求视图。保留用户编辑对象，不一刀切删除旧助手消息，不继续加成功口号。候选策略需先给方案再落地。

## 成本与产物

完整记录中的聊天逻辑调用20次，输入124807、输出14258 token，包含章节分析、计划和澄清/恢复；受中止的RAG链用量不完整，不称本轮完整账单，不换算金额。归档统计usage-summary.json，全部实际user_input、request_messages、工具结果及回答保存在各observation.json；readable-transcript.md可直接阅读。

原始整书尝试：.evaluation-results/books-real-20261105；已知页/章节替代：.evaluation-results/books-chapters-20261105。两者不合并为一个成功率。闭环原失败、澄清补充及拒绝后恢复均单独保存。

## 执行命令

在F:\LLM\projects\agent执行。预检只写manifest、不调用模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tests.helpers.book_acceptance --chapter-mode --output-dir .evaluation-results/books-preflight-new
```

加--llm实际执行；采用全新输出目录。章节模式可能按本轮轨迹停在澄清而非确认，非零/阶段失败需要查看实际回答，不能直接归为模型错误。整书模式省略--chapter-mode，须先解决供应商embedding分配额度，否则不要重复执行。所有自动输入见manifest，实际输入以各*-input.json和observation为准。

本轮澄清继续与拒绝后恢复使用已保存的clarified-input.json、declined-resume-input.json经同一learning_closed_loop --worker入口执行；不能直接覆盖这些原产物重跑，需要复制输入并改成新输出路径。
