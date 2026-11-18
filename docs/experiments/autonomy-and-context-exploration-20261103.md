# 自主取证、边界修复与上下文探索（2026-11-03）

## 先列需要确认的底层候选

1. 最终响应为空时，保持原证据和请求，仅对生成重试一次。当前遇到empty+no_tool_calls即invalid_analysis_output。离线原型验证瞬时空响应可恢复、持续为空仍失败、工具请求不重试、原证据未变；尚未接入生产，也未证明线上可恢复率。需要明确额外请求预算及与网络重试的组合，不能直接延长循环或重跑工具。
2. 原始回执内容寻址存储＋可回读请求视图。10份模拟回执41560字符，预览/引用5470字符，跨实例精确回读10/10；只是存储原型，不是线上token或质量提升。需要生命周期、权限隔离、回读工具及完整性语义后才能启用。
3. 统一分析与交互的目录/读取适配，优先复用已经存在的build_list_material_files_tool分页能力；不再为每个阶段建设独立工具。涉及工具schema迁移，未改生产绑定。

## 外部参考及取舍

本轮查阅当前main源码（非发布版本保证），未执行外部代码：

- Deep Agents [filesystem.py](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py)、[protocol.py](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/protocol.py)：搜索返回支持truncated，读取携带真实窗口和续读信息。借鉴“获取边界可见”，不照搬整个工具框架。
- Deep Agents [_message_eviction.py](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_message_eviction.py)：大型工具结果外置与首尾预览。只做本地原型，不默认省略证据。
- Pi [compaction.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/compaction/compaction.ts)：实际usage锚点、后续消息估计、投影变更后失效、保留近期token。只先审计完整请求组成，不直接移植其字符/4估计到中文Qwen。
- OpenHands SDK [llm_summarizing_condenser.py](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/context/condenser/llm_summarizing_condenser.py)：独立上下文压缩与阈值。当前小案例无上下文压力，不急于增加摘要模型或压缩器。

这些是实现参照，不是对本项目效果的外部保证。保留模型自主工具选择，不设置必经工具顺序。

## 实验一：自主发现与资料缺失

输入文件`tests/fixtures/evaluation/autonomous_material_cases.json`，完整原文、初始三个文件与预期均在其中。overview指向concepts/checks；完成标记只代表提交，掌握须独立检验；检验为四例至少三例正确并说明依据。

问题一：“请依据这组资料解释‘完成标记’和‘掌握标记’的区别，并说明获得掌握标记需要什么条件，给出资料来源。”
问题二：“这组资料是否规定了第三种标记的晋级条件？如果没有，请明确说明，不自行补充规则。”
两例各独立运行2次，无预设工具请求；完整生产循环和schema，当前选择性整理策略。

| 案例/重复 | 真实模型次数 | 工具数 | 输入token | 输出token | 人工判断 |
|---|---:|---:|---:|---:|---|
| 跨文件1 |3|6|3727|558|通过；list、2次search、3次read|
| 跨文件2 |3|4|3547|573|通过；list后批量3次read|
| 缺失1 |4|6|4992|762|通过；读完3文件后2次search确认，无编造|
| 缺失2 |3|4|3529|629|通过；读完3文件，说明本组资料未规定|

13次真实请求无重试，20次工具执行，源文件和资料指纹不变。四次最终事实与来源语义均符合当前小型合成资料；三次未使用标准[source]标记但具备文件名/原文，不能将引用日志视为语义失败。不能外推真实PDF或大目录能力。
产物：`.evaluation-results/autonomy-online-20261103/`，requests记录真实输入，source全部model_request。

## 实验二：复现并修复发现工具的获取边界

离线首次2失败2通过：list仅返回前N文件不说明总数；search达到结果上限立即返回不说明扫描未完整。已小范围修改resources/tools.py::build_agent_tools：截断清单增加“仅返回N/M个文件”，并交接truncated获取事实；搜索达到上限时说明未确认是否还有其他命中，并交接limit_reached事实。未达上限返回保持原样。无新工具、schema、预算、系统提示或执行分支。

参考Deep Agents搜索截断契约，且项目已有acquisition_facts，不新增一套状态。搜索恰好达到上限时只说“未确认”，不声称一定还有命中。相关31通过1环境跳过。

补充线上ABBA：清单上限设2，三个文件a_notes/b_notes/z_condition，目标在未展示的z_condition。
问题：“本目录是否规定了高级标记的通过条件？若有请准确说明并注明来源；若未找到，请说明本次检查范围。”
正确条件：“高级标记要求完成一次口头答辩，并说明两项判断依据。”
两组同一完整生产循环、模型、选择性整理策略，只比较旧/新发现工具返回。

| 组/重复 | 模型次数 | 工具数 | 输入token | 输出token | 结果 |
|---|---:|---:|---:|---:|---|
| 旧1 |5|7|6407|585|找到并读取目标，正确回答|
| 新1 |4|5|4872|616|找到并读取目标，正确回答；细则缺失等延伸有超出检查范围的表述|
| 新2 |4|5|4887|535|找到并读取目标，正确回答|
| 旧2 |4|6|4948|522|已读目标，最终空响应；stop、output_tokens=0，invalid_analysis_output|

17次真实请求，无网络重试，23次工具执行。不能将旧2空响应归因于旧清单，也不能凭2/2成功宣布新清单提高稳定成功率；可确认的是边界信息不再静默丢失。成本和调用数只作观察。产物：`.evaluation-results/discovery-limit-online-20261103/`；旧工厂源码保存在discovery-boundaries-20261103/baseline_factory.py。

## 探索三：完整请求审计（未改运行策略）

分析13次真实请求的消息、参数及重建OpenAI工具schema：首次消息正文718字符，序列化消息790字符，工具schema1488字符，对应实际输入1076token。最大正文1127字符、序列化消息2596字符，最大实际输入1463token。
仅数m.content会漏掉调用参数、结构及工具schema。字符不等于token，未使用字符数估价。当前小样本没有足够压力证明动态删工具/摘要值得做；不实施动态工具裁剪。
产物：`.evaluation-results/context-exploration-20261103/report.json`。

## 探索四：原文外置与有限生成重试（仅离线）

原型位于tests/helpers/context_exploration.py。RawReceiptStore以SHA256命名保存UTF-8字节，预览明确不是完整证据，回读验证哈希；覆盖跨实例回读、去重、CRLF、无效ID及内容篡改。
retry_empty_generation_once只在空字符串且无tool_calls时再请求一次，不增加提示、不执行工具；正常tool_calls立即返回，持续为空仍由正式文本校验拒绝。3种离线分支通过，不把模拟恢复当真实模型改善。
这些原型没有增加生产工具、配置或自动压缩策略。

## 测试、交付与下一步

本轮新增无seed测试1项、获取边界4项、原文存储3项后，全量1667通过1跳过3依赖警告（47.89秒）；随后有限重试原型新增3项，与相关测试共16通过，未重复全量。git diff --check通过。
本轮线上合计8条独立链、30次真实请求、43次工具执行，全部一次invoke返回，无网络重试；其中1次模型返回空最终响应导致任务失败。没有提交/推送git，也没有修改预算、系统提示、框架或底层执行逻辑。

优先级：先确认最终空响应的有限生成恢复设计；随后完成“资料→计划→用户反馈→确认/拒绝→恢复/回查”的跨阶段验收，避免一直优化读取微场景。大型结果存储和最近窗口仅在压力案例证明必要时接入。详细行动见短期/长期计划，既有引用日志措辞方案仍未实施。
