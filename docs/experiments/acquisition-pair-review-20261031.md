# 获取范围与限制：配对在线实验

## 结论

2026-10-31，3个案例、2轮配对、12次独立规划全部完成。观察到20次模型响应和8次补查工具调用。6对控制检查全部通过。两边均能完成概念学习计划：正常组不补查，两个受限组均补查一次。本次没有证明新增获取记录能提高模型表现或减少补查。

建议保留证据/限制分离的程序契约与可观察性，停止为本轮结果追加提示词。受限组输入token增加是实际成本，回答长度与耗时差异不能由每组2次样本归因于新机制。

## 文件与结果

- 输入定义：`tests/fixtures/evaluation/acquisition_pair/cases.json`。
- 自编材料：同目录`materials/stop_normal.md`、`stop_protective.md`、`narration.md`、`attitude.md`。
- 执行：`src/ai_agent/evaluation/acquisition_pair.py`。
- 真实规划：`src/ai_agent/runtime/agents.py::create_study_plan`。
- 工具与事实：`src/ai_agent/resources/tools.py::build_retrieve_learning_context_tool`。
- 交接：`src/ai_agent/runtime/context_handoff.py`。
- 汇总：`src/ai_agent/evaluation/acquisition_pair_report.py`。
- 结果目录：`.evaluation-results/acquisition-pair-20261031-v1/`，包含`manifest.json`、`inputs.json`、`results.jsonl`、`summary.json`、`transcript.md`、`review.json`。
- `source-snapshot/`保存实际运行的5份关键源码。结果中的源码不变检查针对运行期间；后续新增的独立汇总程序不改变已保存模型输入。

## 假设、控制与范围

问题是：在同一目标、分析、正文和模型条件下，仅提供获取限制，是否改变补查、事实范围表述及最终计划？

使用配置模型`qwen3.7-plus`，温度0.2，请求超时180秒。每次独立历史，规划使用现有pipeline入口，最多补查1次；pipeline仅指规划工具预算，并非运行完整pipeline资料分析。两轮顺序AB、BA，A无记录、B有记录。正常组没有获取限制，所以AB初始输入完全相同，用于显示采样波动。

冻结的是检索结果，不是模型：向量索引/重排被本地替身取代，真实格式化函数、工具schema、查询处理、规划循环和聊天API仍运行。补查在两臂返回相同材料。此设计不测RAG召回质量、真实检索时延或任意查询的实际有效性。没有调用embedding、重排或LLM裁判。

自动控制检查包含初始证据、系统消息、唯一数据差异和源码稳定性。工具行为是结果，不能要求工具调用相同才承认实验有效。本次恰好每对的补查返回也相同。预设质量边界存于案例文件，不发送给模型。

## 案例与执行

### 1. complete：完整证据

原文提问：依据资料安排1天学习，理解正常结束与保护性停止的区别。只阅读和口头说明，今天最多30分钟。

初始材料包含两种停止的定义、例子与区别，预算16000字符，获取事实为空。两边不应该编造获取限制或无理由拒绝计划。

四次均直接输出计划，不调用补查。两种定义、停止原因和案例判断均有对应来源；活动为阅读/口头说明，明确30分钟以内。AB输入token均1225，但回答与耗时仍有变化，说明单次差异不是机制效果。

### 2. query_limited：查询未完整执行

原文提问：依据资料安排1天学习，区分叙述视角与人物态度。只阅读和口头说明，今天最多30分钟。

初始提交“叙述视角的定义与例子”“人物态度的定义与例子”，真实工具上限1条；冻结初始召回仅返回narration.md。程序生成requested_queries=2、executed_queries=1，查询数是规范化去重后的有效条目。

两边各2次均主动补查人物态度或概念对比，补查返回narration.md与attitude.md。四份最终计划均有对应定义和资料引用，没有把未返回资料说成不存在。补查后不重复解释初始限制是允许的，不能为了让限制显得有用而要求最终回答复述它。

### 3. budget_limited：命中但没有全部提供

提问与complete相同。真实格式化输入2块，预算140字符，仅保留stop_normal.md；程序记录hit_count=2、selected_blocks=1、omitted_blocks=1。两边初始正文相同，只有B看见该获取事实。

四次均补查保护性停止，随后提供有依据的对比计划。A第二轮未明确重申30分钟或提供时间分配，记录为时间约束表达不完整；不能直接判为超时，也不能用单次结果证明B更好。

## 关键执行代码

初始化事实来自真实工具，而不是给模型写评分答案：

```python
initial_tool = tools.build_retrieve_learning_context_tool(
    str(root), max_queries_per_call=case["initial_query_limit"],
    max_context_chars=budget, evidence_blocks=blocks, acquisition_facts=facts,
)
initial_result = initial_tool.invoke({"queries": case["initial_queries"], "neighbor_window": 0})
```

来源：`acquisition_pair.py::prepare_case`。只替换召回结果，保留真实工具生成查询上限、块预算与事实。

配对唯一变量：

```python
facts = case["acquisition_facts"] if arm == "with_facts" else []
context = pack_material_context(case["evidence"], facts)
```

来源：`acquisition_pair.py::run_one`。两臂同一goal、analysis、evidence，并调用真实`create_study_plan(..., mode="pipeline")`。

模型收到的实际分层：

```python
f"资料分析结果：\n{analysis}\n\n"
f"{evidence_label}：\n{plan_evidence}"
f"{format_acquisition_facts(acquisition_facts)}"
```

来源：`agents.py::create_study_plan`。系统行为规则未添加或改写，限制属于动态数据。

每次API调用前保存完整`request_messages`；返回后保存公开content、tool_calls、usage和耗时，不采集reasoning_content等私有字段。`plan_trace.tool_observations`包含实际工具返回。

## 数值比较

每行均为该条件两次任务的平均；补查列是每次任务次数。token包含规划决策与最终回答的全部已记录响应，不只统计最终回答。

|场景|无记录/有记录补查|平均输入token：无/有|平均总token：无/有|平均秒：无/有|
|---|---|---|---|---|
|完整|0 / 0|1225 / 1225|2198 / 2351|17.04 / 18.79|
|查询受限|1 / 1|2020.5 / 2167|3218.5 / 3587.5|21.26 / 24.78|
|证据预算|1 / 1|2069.5 / 2248|3445 / 3804.5|24.06 / 27.42|

受限组有记录臂平均增加146.5、178.5输入token，包含同一限制在连续模型请求中重复出现的开销。补查查询数也有波动：查询受限A/B合计3/4条，证据预算A/B合计3/2条；不能据此宣称新机制更会拆查询。

所有usage字段完整。无/有记录臂总token分别17723/19486，合计37209；没有据此推算人民币费用或稳定的节省/增耗比例。时延同时受输出长度、服务与采样影响。

## 质量复核与决策

由助手逐条对照原文和轨迹，非独立盲评：12份均覆盖关键概念，关键定义有已提供材料支持，仅阅读/口头说明；11份明确写出30分钟界限。未发现虚称资料库不存在相关资料，正常组没有不必要补查。具体例外见review.json。

本实验不能证明限制信息被模型忽略，也不能证明它提高可靠性。任务短、缺口明显、补查固定成功，存在天花板效应。不要删除程序事实能力，也不要把架构正确性包装成模型效果提升。

本轮只新增评测和报告代码及测试，不修改业务提示词、模型、检索策略。下一阶段应使用有判别力的真实/留出任务，例如补查失败、部分任务仍可完成、失败来源与另一个成功来源并存；预先确定标准后再比较，而不是为了追求正结果不断换案例。

## 复现

执行目录：项目根目录。默认预检不会请求聊天模型；`--llm`显式执行12次独立任务，最多24次应用层模型调用（本次实际记录20次响应）。输出目录必须使用新名称，已有results.jsonl时拒绝覆盖。

```powershell
.\.venv\Scripts\python.exe -m ai_agent.evaluation.acquisition_pair --output-dir .evaluation-results/acquisition-pair-preflight
.\.venv\Scripts\python.exe -m ai_agent.evaluation.acquisition_pair --output-dir .evaluation-results/acquisition-pair-new --llm --repeats 2
.\.venv\Scripts\python.exe -m ai_agent.evaluation.acquisition_pair_report .evaluation-results/acquisition-pair-new
```

辅助代码最终相关回归136项通过。已有业务实现未因本实验再次修改；不重复在线请求来制造更有利样本。
