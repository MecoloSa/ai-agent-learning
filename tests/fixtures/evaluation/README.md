# 评测案例与本地资料

## 反馈计划案例：第一步契约（2026-10-24）

`feedback_plan_cases.jsonl` 当前有3个开发案例，不含留出集。覆盖只讨论、时间缩减候选被拒绝、非技术阅读的活动限制；这不是完整12案例评测集，也没有模型质量成绩。

每行包含 `case_id`、`split`、`material_dir`、`initial_state_file`、`inputs`、`checks`。两条路径相对于JSONL所在目录，解析后必须仍位于该目录内；快照与资料仅作初始输入，加载不修改它们。`feedback_plan/` 中均为合成资料，v7状态省略字段由现有状态模型补默认值。

`checks` 中每项有唯一 `check_id`、`category`（control/quality/evidence）、`criterion`（通过标准）、`basis`（判据依据）。它们是评测规则，不是实际结果或已获得的人工标签。未来报告须另存观察、依据与true/false/null，不能将案例能加载计作质量通过。所有判据仅供评分，不作为生成模型额外提示。

`inputs` 是交互回放的输入序列。已实现的回放器核对实际提示阶段：v/n只允许投递到对应计划确认入口，不能在模型未产生候选时直接按顺序当聊天问题投递。出现澄清等意外分支记录未完成/脚本不匹配，不直接改状态或伪造pending。

`src/ai_agent/evaluation/feedback_plan_cases.py`、`feedback_plan_input.py` 和 `feedback_plan_runner.py` 已应用。离线运行器及关联五文件74项已复验；假回答测试不代表真实模型质量通过。不改变旧评测runner和业务入口。

已落地可选 `checks[].rule`：仅control项可指定state_unchanged或plan_declined。前者检查完成回放后的业务状态是否不变；后者还要求至少一次真实计划确认调用，全部明确返回user_declined且accepted=false。禁止依据case_id或criterion关键词自动推断规则；未声明规则的检查保持null。用户核对v2输出符合预期，关联六文件98项已复验。

评分候选只使用实际业务返回的确认结果，不从inputs中的n推断已拒绝。未完成回放或必要观察缺失时返回null；true/false只表示声明的控制规则通过/违反，质量与证据检查本轮仍为null。assessment=not_evaluated继续指内容质量尚未评估。旧结果没有确认记录时不能补猜拒绝经过，也不回写旧实验文件。

`feedback_plan/offline_responses.json` 为离线控制回放提供逐案例固定回复，映射case_id到回复列表；不是标准答案或真实模型结果。时间缩减案例故意保留60分钟候选，用来证明控制回放完成不能当作语义通过。正式判据不传入生成路径。回放范围仅支持普通聊天、计划确认、window策略与本地资料工具绑定；不支持状态命令、反馈/任务确认或独立进程恢复。原runner命令继续离线，真实模型使用独立feedback_plan_online入口显式--llm。

观察与人工评审辅助已应用，关联八文件136项测试通过（含7项观察测试、31项评审辅助测试）。
结果保存本次案例/判据、初始业务上下文、实际消费回复及候选处理决定，避免从拒绝后的旧计划误评候选。
`feedback_plan_review prepare/report`导出独立标注表、校验来源和标注并汇总；
pending不同于已评审但无法判断的null，布尔结论需评审人、理由和实际观察引用。
报告只统计录入的标注，不判断标签是否正确；不采集资料正文，不自动确认资料依据，不请求模型。
v3实际回放仍为3控制项true、4内容项未评估。详见[src说明](../../../src/ai_agent/evaluation/feedback_plan_review.md)。

## 资料依据专项开发案例

`feedback_plan_evidence_cases.jsonl` 独立保存dev-004，复用原reading_state和合成资料，
不修改原3案例或其历史结果。原dev-003只要求活动调整，真实回复没有知识性解释或引用，
因此证据项标为reviewed/null（unknown），不能把未触发的能力算通过，也不自动记失败。

dev-004的用户输入明确要求实际读取literature_notes.md、引用并解释藏信例句，
再提供保留目标的阅读/口述候选，最后按v/n拒绝。已知文件名是任务条件，不是隐藏标签；
通过标准和解释依据只进入评审，不额外注入生成提示。该例测明确资料需求下的读取与使用，
不测自主发现文件或检索排序，不是留出集；不因未使用RAG或MCP而判失败。

新增2项离线测试：独立案例可加载且原3例不变；替身模型调用真实按行工具，
保留真实引用、工具结果及拒绝记录。连同关联回归181项通过，未执行dev-004在线调用。
证据项需分别核对实际读取、引文一致、位置及解释边界，不能只检查是否出现引用标记。

## 时间策略与证据越界反例

dev-004真实输出出现“资料明确指出……不是叙述者有意为之”的越界说明；
原文仅提供信息限制，未排除有意隐瞒。助手代审综合证据项false，读取和引用定位成功，
不能相互抵消。原始结果保留在本机.evaluation-results，不将整份运行轨迹复制为公开资料。

`feedback_plan_time_cases.jsonl`另存两个开发案例，不改变既有题目：
dev-005要求不按天、不估分钟的自主路径；dev-006明确提供不可压缩的最低耗时与本次预算冲突。
最低耗时来自用户输入，不让模型凭资料页数猜测。两例均只讨论、不写状态，不是留出集。
它们为时间策略候选提供人工效果检查，接线测试不自动判定自然语言输出合格。
运行前先读[时间案例说明](feedback_plan_time_cases.md)：列出初始背景、完整提问、自动输入和判据，
两个案例独立初始化，不继承此前其他案例的用户约束。

## 采用后保留学习记录（候选契约）

`feedback_plan_adoption_cases.jsonl`保存dev-007，初始状态adoption_state.json含3项任务和1条反馈。
完整输入与自动v/y/exit过程见[采用案例说明](feedback_plan_adoption_cases.md)。
计划变更应令修订号增加、tasks_need_review=true，任务进度、理解、取消标记及反馈保留；
business_state_unchanged=false属于正确预期。该判据只覆盖一次计划采用，不允许同时改目标。
plan_adopted、final_context及评审引用接线尚属业务候选；源码应用后才能正式加载运行此例。
21项新增测试在临时候选上通过，含实际确认/落盘及记录损坏、观察缺失反例；不调用在线模型。

## 其他案例与本机资料

history_feedback_integration.json将归档40→20分钟更正与adoption_state.json的3任务/1反馈结合，提问不重复限制。
tests/helpers/history_feedback_integration.py默认展示初始状态、完整历史、提问及确认序列，--llm才运行真实生成。
采用/拒绝仅操作隔离存档；恢复使用新进程真实CLI和模型替身，不能当作在线复述质量验证。
脚本不匹配保留失败，不把“文本里有候选”推断成正式提交；原文、预期和结果见[联合验收](../../../docs/experiments/history-feedback-integration-review-20261031.md)。

history_grounding_judge_cases.json为9项独立证据边界裁判开发诊断，包含忠实复述、扩大范围、虚构原因、合理建议、明确冲突及含糊表述。
预期标签在调用前冻结，不发送给裁判；case原文与回答均可由tests/helpers/history_grounding_judge_probe.py默认预检查看。
真实完整回答的4项迁移材料保留在本机忽略的.evaluation-results，不作为公开原始轨迹提交。
这组用于检验裁判是否误判，不能当成生成模型的任务成绩或独立留出集。结果与分歧见[独立核验诊断](../../../docs/experiments/history-judge-probe-review-20261031.md)。

2026-10-31新增历史解释对照开发案例016/017：history_scope_reading_allowed.json与history_scope_reading_limited.json。
两例同一提问、独立初始化，原话仅在是否明确允许阅读文字上不同；用于区分明确条件遵守与未说明条件的扩大推断。
完整历史、提问、自动输入、预期与单次结果见[解释忠实度诊断](../../../docs/experiments/history-grounding-review-20261031.md)。
引用、解释与状态保护分别判断，程序附上真实来源不能记为模型复制成功，非留出集或自动内容质量金标准。

历史原文回查在线验收使用`history_readback_online_cases.json`的3个独立开发案例：
已知来源轮次、先发现来源再读取、数据库存在但快照上限拒绝读取。
完整原文、自动输入、预期结果和判据见[在线回查验收](../../../docs/history-readback-online-acceptance-20261028.md)。
`tests/helpers/history_readback_online.py`默认只预检，显式`--llm`才请求真实模型；
意外确认自动拒绝并记录。结果的completed、退出码0不代表内容质量合格，assessment默认not_evaluated。
辅助脚本8项离线测试通过，真实在线结果待运行；合成案例不是留出集。

`history_search_online_case.json`为独立的004开发案例，复用两轮原文，但不提供目标轮次，要求先搜索再回读。
采用既有BM25的历史适配候选，参数为source_index/query/k；旧SQL候选的keyword/after_turn/limit不再使用。
完整初始状态、原问题、8轮离线反例及命令见[BM25指导](../../../docs/history-bm25-guidance-20261029.md)。
候选157项关联测试通过，业务未应用前不执行004在线验收。

此目录保留可公开的金标准、合成边界案例和在线案例模板。教材PDF、用户资料、原始模型回答与调用轨迹不随仓库发布。

`end_to_end_smoke`、`end_to_end_citation_check`、`learning_goal_alignment`、`rag_cases` 的无后缀本地JSONL包含机器专属路径，已忽略。对应 `*.example.jsonl` 保留原题目和标注，使用项目相对资料路径。模板文件不会自动触发在线请求。

第一次使用时，可在项目根目录复制模板，再将本地副本中的 `material_dir` 改成你有权使用的实际资料目录，例如：

```powershell
Copy-Item tests/fixtures/evaluation/end_to_end_smoke.example.jsonl tests/fixtures/evaluation/end_to_end_smoke.jsonl
```

若本地副本已经存在，直接编辑它，不重复复制覆盖。书籍页码金标准只适用于对应书籍版本，资料不同需重新标注，不能把没有教材导致的失败解释成模型退化。原本机案例保留在磁盘，此次未改写其运行路径。
