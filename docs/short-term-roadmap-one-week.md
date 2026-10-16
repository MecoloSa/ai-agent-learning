# 项目短期完善计划：一周建立可量化基线

> 适用时间：2026-09-30 起的一周  
> 项目目标：本地学习资料分析、学习计划生成与多轮问答 Agent  
> 本周原则：先修正确性与补评测，再增加新框架；保留 `pipeline` 作为确定性对照组。

## 1. 本周结束时应达到的状态

本周不追求同时实现 LangGraph、长期记忆、MCP 和多 Agent。最有价值的交付是让当前系统从“能运行”变成“可验证、可比较、可复盘”：

- 当前关键流程没有已知的类型或分支错误；
- Agent 工具循环、RAG、Memory 和交互检索都有离线测试；
- 有一小套固定评测案例，可同时运行 `pipeline` 与 `agent`；
- 每次实验会保存答案、引用、工具轨迹和质量/成本/延迟指标；
- README 的配置和命令与代码一致；
- 能用 3～5 分钟解释一次完整执行链路，以及为什么暂时不急着引入 LangGraph。

建议投入 2～4 小时/天。时间不足时，优先完成第 1～4 天；第 5～7 天可以顺延，但不要牺牲测试去赶新功能。

## 2. 当前基线（以代码为准）

### 2.1 已实现能力

| 能力 | 当前代码位置 | 当前状态 |
| --- | --- | --- |
| Function Calling | `src/ai_agent/agents.py::analyse_materials_agentic`、`create_study_plan`，`src/ai_agent/main.py::_invoke_interactive_turn` | 已有工具选择、参数回传、异常转为观察、调用上限和重复调用保护 |
| 确定性基线 | `src/ai_agent/agents.py::analyse_materials`、`run_learning_assistant` | `pipeline` 与 `agent` 可对照 |
| 短期记忆 | `src/ai_agent/memory.py::ConversationMemory` | 已有 `full/window/summary`；目前缺少系统对比测试 |
| RAG | `src/ai_agent/rag.py::LearningRAG` | 已有加载、切分、向量化、增量索引、阈值过滤、多查询合并、邻居扩展、上下文预算和引用元数据 |
| 可观测性 | `src/ai_agent/agents.py`、`rag.py`、`main.py` | 已记录阶段、耗时、token、检索候选/命中和引用摘要，但还没有统一实验结果结构 |
| 安全边界 | `src/ai_agent/materials.py`、`tools.py` | 已限制资料根目录和支持的文件类型 |

### 2.2 已确认的缺口

1. 现有 `tests/` 共 26 个用例；在工作区临时目录下运行全部通过，但主要只覆盖文件工具和运行路径。
2. `agents.py`、`rag.py`、`memory.py` 和交互工具循环没有核心行为测试，修改提示词或控制流时容易静默回归。
3. 当前工作区的 `main.py::run_interactive_session` 中，`max_context_chars=(settings.interactive_retrieval_max_context_chars,)` 会构造元组；无 RAG 分支中的 `retrieval_tool = None,` 也会构造单元素元组。前者会在进入交互检索时造成类型问题，后者会让“没有检索工具”的判断失真。
4. README 与 `config.py` 的部分默认值已经漂移，例如聊天模型和 RAG 上下文预算；文档不能继续作为可靠运行契约。
5. 还没有固定评测集、检索标注、统一指标和可比较的实验报告，因此无法证明某次优化真的更好。
6. `langgraph`、`langchain-mcp-adapters` 已在依赖中，但当前代码没有对应的状态图或 MCP Client/Server 实现。依赖存在不等于能力已落地。

## 3. 一周执行安排

### 第 1 天：冻结基线并修复已知回归

**目标**：确保后续评测测的是有效程序，而不是已知类型错误。

**修改位置**：

- `src/ai_agent/main.py::run_interactive_session`
- `tests/test_interactive.py`（新增）
- `src/ai_agent/README.md`

**任务**：

1. 去掉上述两个误加的尾逗号，并为“需要 RAG”和“不需要 RAG”两个分支各加一个测试。
2. 使用假的 LLM 和假的检索工具测试 `_invoke_interactive_turn`，不调用真实 API。
3. 同步 README 的当前默认配置，不趁机重构业务代码。
4. 记录基线提交号、Python 版本、模型配置名和测试命令。

**验收**：

- `pytest` 全绿；
- 测试能在错误重新出现时稳定失败；
- 无 RAG 时不会创建或调用检索工具；有 RAG 时字符预算是 `int`。

**动态评估**：观察失败究竟发生在模型、工具还是控制流边界。若测试必须依赖真实模型才能复现，说明依赖注入边界还不清晰，应先拆出纯控制逻辑。

### 第 2 天：为 Function Calling 循环补行为测试

**目标**：掌握 Agent 的“行动—观察—再决策”闭环，而不只看成功输出。

**修改位置**：

- `src/ai_agent/agents.py::analyse_materials_agentic`
- `src/ai_agent/agents.py::create_study_plan`
- `tests/test_agent_loop.py`（新增）

**至少覆盖**：

- 模型不调用工具，直接返回最终答案；
- 一次合法工具调用后收束；
- 未知工具会得到结构化错误观察；
- 工具抛异常时流程不崩溃，模型能收到失败结果；
- 连续三次相同调用提前终止；
- 达到最大轮数或检索次数后强制收束；
- 相同参数的重复检索仍计入预算。

**验收**：测试完全离线、结果确定，不消耗 API token。

**动态评估**：记录 `tool_calls_total`、`tool_calls_unique`、`tool_errors`、`termination_reason`。重复率 `1 - unique / total` 持续偏高，说明工具描述、状态反馈或终止提示需要调整，而不是单纯提高轮数上限。

### 第 3 天：为 Memory 建立对照实验

**目标**：能解释三种短期记忆策略的语义和代价。

**修改位置**：

- `src/ai_agent/memory.py`
- `tests/test_memory.py`（新增）
- `src/ai_agent/eval/cases/memory_cases.jsonl`（新增）

**任务**：

1. 测试 `full` 保留全部历史、`window` 只保留最近窗口、`summary` 达阈值后压缩且保留系统状态。
2. 加入“用户先声明目标，数轮后修改天数/约束”的案例，检查旧目标不会覆盖新目标。
3. 保存每种策略的输入字符数、估算 token、关键信息保留率和摘要次数。

**验收**：相同会话可以一条命令跑三种策略并生成对比表。

**动态评估**：

- `full` 质量高但 token 线性增长，是正常现象；
- `window` 丢失早期关键约束，说明窗口不适合单独保存长期状态；
- `summary` 多次压缩后事实漂移，说明摘要提示或“结构化状态 + 自由摘要”的边界需要调整。

### 第 4 天：建立 RAG 离线评测最小集

**目标**：把“检索看起来不错”变成可计算结果。

**修改位置**：

- `src/ai_agent/eval/cases/rag_cases.jsonl`（新增）
- `src/ai_agent/eval/rag_eval.py`（新增）
- `tests/test_rag.py`（新增）

**任务**：

1. 从项目实际资料中人工制作 10～20 条问题，每条标注 `gold_source`，能标页码或 chunk 时一并标注。
2. 对 `retrieve`、`retrieve_many`、阈值过滤、去重、邻居扩展和字符预算做离线测试；向量库接口使用假对象或小型固定 embedding，避免网络波动。
3. 输出 Recall@K、MRR、无结果率、上下文字符数和检索耗时。

**指标解释**：

- Recall@K：正确证据是否出现在前 K 条，先解决“有没有找回来”；
- MRR：第一个正确证据排得是否靠前，衡量排序质量；
- 无结果率：阈值是否过严；
- 上下文字符数：质量提升是否靠无节制塞上下文换来。

**验收建议**：首周不强行设行业阈值，以当前实现跑出的结果作为 `baseline-v1`。以后只有在同一数据集上比较，结论才有效。

### 第 5 天：搭建端到端评测骨架

**目标**：对 `pipeline` 和 `agent` 做相同输入下的公平比较。

**修改位置**：

- `src/ai_agent/eval/schema.py`（新增）
- `src/ai_agent/eval/runner.py`（新增）
- `src/ai_agent/eval/cases/end_to_end_cases.jsonl`（新增）
- `src/ai_agent/eval/results/`（新增，生成结果默认不提交或只提交脱敏样例）

**建议结果字段**：

```text
run_id, git_commit, case_id, mode, model, config_snapshot,
success, latency_ms, input_tokens, output_tokens,
tool_calls_total, tool_errors, termination_reason,
retrieved_sources, cited_sources, answer_path
```

**首批 8～12 个案例**：正常小资料、大文件触发 RAG、PDF、混合代码和文档、空目录、无关问题、证据不足、需要多轮工具调用、更新目标、更新计划。

**规则评分先行**：格式完整、天数正确、引用可解析、引用来源真实存在、失败时有明确原因。LLM-as-Judge 暂时只作为补充，不替代可确定的规则。

**验收**：同一批案例可以分别跑两种模式，失败案例不会中断整批评测，并输出 JSONL/Markdown 汇总。

### 第 6 天：做第一次对照实验与人工抽检

**目标**：用数据形成第一份可讲述结论。

**固定变量**：同一模型、temperature、资料、问题和评测脚本；只改变 `mode`，或一次只改变一个 RAG/Memory 参数。

**比较维度**：

| 维度 | 建议指标 |
| --- | --- |
| 任务完成 | 成功率、计划结构通过率、天数正确率 |
| 证据忠实 | 引用格式通过率、引用存在率、人工支持度 |
| Agent 稳定性 | 工具错误率、重复调用率、强制终止率 |
| RAG | Recall@K、MRR、无结果率 |
| 成本与性能 | 总耗时、模型调用次数、token、检索上下文字符数 |
| 一致性 | 同一案例重复 3 次的规则得分方差 |

**验收**：生成 `docs/experiments/baseline-v1.md`，明确记录“结论、证据、局限、下一步”，不能只粘贴日志。

### 第 7 天：文档收口和面试演练

**目标**：让项目成果既能复现，也能被清楚讲出来。

**任务**：

1. README 增加测试与评测命令、两种模式对比和已知限制。
2. 画一张不依赖框架术语的执行链路图：输入 → 路由 → 检索/工具 → 分析 → 计划 → 交互 → 保存。
3. 准备 3～5 分钟项目介绍，至少回答：
   - 为什么保留 `pipeline`？
   - Agent 怎样选择、执行和停止调用工具？
   - RAG 的阈值、邻居扩展和上下文预算解决什么问题？
   - `full/window/summary` 如何取舍？
   - 你如何证明一次改动有效？
4. 将本周未完成事项移入长期路线，不临时堆进主分支。

**验收**：新同学按 README 能跑测试、读懂报告，并能区分“已实现”“实验中”“规划中”。

## 4. 本周完成定义（Definition of Done）

- [ ] 已知交互分支错误有回归测试并已修复。
- [ ] Agent 循环、Memory、RAG 各有正常、边界、失败路径测试。
- [ ] 固定评测案例有版本号，案例与结果可追溯。
- [ ] `pipeline` 与 `agent` 至少完成一次同条件对比。
- [ ] 报告同时包含质量、稳定性、延迟和 token；没有只报“感觉更好”。
- [ ] 日志和结果中没有 API Key、完整敏感资料或模型私有思维链。
- [ ] README 与代码配置一致，清楚标明未实现的长期记忆、LangGraph、MCP 和多 Agent。
- [ ] 新增关键逻辑含适量中文注释，解释设计原因、失败边界和应观察的指标；不重复解释显然代码。

## 5. 本周明确不做

- 不为了简历关键词把当前流程整体重写成 LangGraph；
- 不在没有单 Agent 基线前拆多 Agent；
- 不先接入 LLM-as-Judge 就跳过规则评测和人工抽检；
- 不把向量库本身称为长期记忆；只有形成“提取—存储—检索—更新/遗忘—注入—评测”闭环才算长期记忆；
- 不记录或要求模型输出私有思维链。可解释性使用计划、工具调用、观察结果、状态变化和评分结果实现。

## 6. 推荐的首周提交顺序

1. `fix: correct interactive retrieval branch types`
2. `test: cover agent tool-loop termination and failures`
3. `test: compare conversation memory strategies`
4. `eval: add versioned RAG cases and retrieval metrics`
5. `eval: add pipeline versus agent benchmark runner`
6. `docs: record baseline-v1 results and current limitations`

每个提交只解决一类问题，便于回退，也便于面试时展示架构如何逐步演进。
