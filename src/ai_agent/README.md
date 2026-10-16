# 证据驱动的本地学习智能体

本项目面向研究生和自学者：根据本地资料与学习目标生成有来源依据的学习路径，并通过多轮对话解释资料、调整目标和更新计划。学习内容不限于编程；阅读理解、观点比较、技能练习和代码实现应采用不同的学习活动与完成标志。

长期目标是形成“资料依据 → 初始路径 → 用户反馈 → 必要调整 → 进度延续”的闭环。当前已实现资料分析、计划生成、会话内目标/计划更新、手动任务登记与确认反馈、状态变更后自动保存快照及状态管理恢复。恢复不包含聊天历史或资料问答；自动局部重规划仍待实现，不能将当前聊天记忆等同于这些能力。

当前支持 `.txt`、`.md`、`.py` 和 `.pdf`。所有文件访问都限制在用户指定的资料目录内；PDF 通过 `pypdf` 读取文本层，扫描版 PDF 暂不支持 OCR。

文件格式决定读取方式，不应决定学习主题或强制实践。例如 Markdown 可以是文学笔记，PDF 可以是编程教材，Python 源码也可以只用于理解算法而不要求执行。

## 当前能力

- **两种执行模式**：`pipeline` 保留固定流程基线；`agent` 自主选择只读工具。资料分析阶段同时限制模型轮数、工具总执行次数和单工具执行次数；同参请求在执行前拒绝，包含 ABAB、ABCABC 中的重复请求。连续无实际执行进展或请求突破总预算时，以明确失败停止后续计划生成和导出。预算按一次分析任务重置，执行异常也占用执行额度。
- **自适应资料处理**：小型文本资料直接读取文件清单和有限摘录；存在 PDF、单个大文件或大规模语料时，自动切换到 RAG。
- **可解释 RAG**：使用 Chroma 增量索引，业务多查询经 `retrieve_with_strategy()` 选择 `dense` 或 `hybrid`，默认保留向量召回。重排默认使用套餐聊天端点的 DeepSeek，支持切回专用重排或关闭。混合路径采用向量＋BM25 → RRF → 重排 → 位置去重；聊天后端只排序，专用后端额外按相关性分数筛选。索引、融合或重排异常时回退原向量检索，不再次请求重排，也不自动切到付费模型。专用重排正常筛选后无合格结果返回空。原 `retrieve()` 默认不重排，保留评测向量基线。
- **关键词检索基础模块**：`lexical_retrieval.py` 提供中文双字词／英文词 BM25 和内存倒排索引；旧评测入口复用同一个类，保持公式和同分排序规则。`LearningRAG.retrieve_lexical()` 已接入按实例延迟构建的缓存，资料增改删时在更新向量库前失效，返回候选副本以隔离临时标记，相关离线测试通过。关键词候选不要求 PDF 页码，独立关键词入口不改变原向量检索基线。
- **通用融合候选入口**：`retrieval_fusion.py` 与 `LearningRAG.retrieve_hybrid_candidates()` 按 `source/chunk_index` 融合两路排名，保留同页不同块及原位置；同位置正文或版本冲突显式报错。默认每路最多 32 块，业务融合池上限复用重排候选配置（默认 16）；仅关键词召回的块不伪造向量分数。业务路径分别保留向量与 RRF 分数、重排排名及后端；只有专用后端产生重排分数。邻居扩展让真正命中块继承这些查询标记，普通邻居不继承。

关键词缓存仅在当前实例内复用：每次查询仍先通过 `sync()` 检查文件哈希，未变化时不重复读取全库和分词；资料变化或更新中途失败后，旧缓存不可继续使用。缓存构建校验正文、元数据和块位置一致性，全部成功后才发布。进程重启需重建内存关键词索引，但未变化资料不因此重新向量化。倒排索引和缓存只改善执行方式，不自动提升召回或证明候选有充分依据。

`retrieve_many()` 接受任意数量的有效查询，包括单条。内部逐条检索，再按各结果列表的名次轮询合并；这不是额外的聊天模型决策，也不强制交互使用初始分析的四类查询。多查询会增加向量查询和重排工作，拆成多次工具调用还会增加模型往返及历史上下文重读。交互的 `INTERACTIVE_MAX_RETRIEVAL_CALLS` 当前限制本轮所有资料工具请求，而非仅限制语义检索；改变批量方式时不应同时盲目提高总预算。

受控在线验证已覆盖：单条明确查询，以及两条互补查询的一次批量提交；日志确认查询数量、邻居窗口、实际工具执行与关键词缓存复用，未被替换成 PDF 读取。显式检索请求采用 `returned` 确认，表示工具正常返回，不等于命中充分或答案正确；其他已注册的结构化工具仍按 `succeeded` 判定。以上验证不证明模型自主选择查询拆分始终最优，首轮回答还存在超出窄问题展开过多的现象。
- **多角度初始检索**：`pipeline` 模式下将大资料检索拆分为 `goal`、`organization`、`implementation`、`usage_validation` 四类查询，并将确定性资料概览与正文证据同时交给资料分析智能体。当前查询按文件后缀追加部分技术类关键词，通用学习适配待修正；这些 ID 是查询标签，不应成为所有领域固定的学习要求。
- **证据驱动计划**：学习计划智能体同时接收分析结论和带来源的证据，要求任务标注引用；资料不足时应明确说明，而不是补写无依据内容。
- **多轮交互与记忆**：`--interactive` 支持 `full`、`window`、`summary` 三种短期记忆策略，并配上”目标更新“与”计划更新“内嵌功能。大资料场景会依据每轮用户问题重新检索，计划或目标更新后同步写回会话状态，退出后保存最终版本。
- **受控调试日志**：默认只保留启动模式/编排方式、分析与规划阶段、简短模型耗时/token、异常和保存路径；预算明细、查询、阈值、逐块命中、引用列表、上下文规模和重复节点日志在 `-v` 下展示。HTTP、OpenAI、Chroma 等依赖保持 WARNING 级别。

## 执行流程

1. 校验资料目录及支持的文件类型；目录无效时提前结束，不调用计划智能体。
2. 根据资料规模选择直接摘录或 RAG；RAG 索引按文件哈希增量同步。
3. 资料分析智能体基于概览、工具结果或检索证据生成知识结构分析。
4. 学习计划智能体结合分析结果与原始证据生成指定天数的计划。
5. 可选进入交互模式；退出后将最终目标、资料分析和学习计划写入 Markdown。

`pipeline` 模式通常有资料分析和计划生成两个主回答请求，但检索还可能调用 Embedding 和聊天重排；这不等于整次运行只有两次聊天请求。`agent` 的主回答请求次数受工具轮数影响，独立重排请求同样需要另计。

### LangGraph 迁移状态（2026-10-16）

当前 `run_learning_assistant()` 仍为普通函数编排，尚无 `--orchestrator` 参数或图入口。下一步拟增加可选 `legacy|langgraph` 外层编排，仅拆出目录校验、分析、规划三个节点；`pipeline|agent` 仍决定节点内分析方式，DeepSeek 重排继续复用原业务函数，不另建专用重排客户端。首步不包含 checkpoint、失败续跑或交互循环迁移；新增代码与测试尚待用户实施，不标记为已完成。

## 主要模块

| 文件 | 职责 |
| --- | --- |
| `main.py` | 命令行入口、日志初始化、交互会话和结果保存 |
| `agents.py` | 模型调用、重试、两种资料分析模式、学习计划生成和引用汇总 |
| `rag.py` | RAG 路由、资料概览、增量索引、阈值检索和邻居扩展 |
| `embeddings.py` | DashScope Embedding 分批适配，单批最多 20 条 |
| `memory.py` | `full`、`window`、`summary` 对话记忆及目标/计划状态历史 |
| `tools.py` | 文件列举、受限读取、关键词检索、文本统计和语义检索工具 |
| `materials.py` | 支持类型、目录遍历和路径越界保护 |
| `config.py` | `.env` 加载、参数校验以及聊天/Embedding 客户端创建 |

## 配置

项目要求 Python 3.14 及以上，依赖由项目根目录的 `pyproject.toml` 和 `uv.lock` 管理。先在项目根目录安装依赖：

```powershell
uv sync
```

将 `src\ai_agent\.env.example` 复制为 `src\ai_agent\.env`，然后至少配置聊天模型所需参数：

```env
DASHSCOPE_API_KEY=your-chat-api-key
CHAT_BASE_URL=your-openai-compatible-base-url

# 使用 RAG 时必填；若同一密钥同时具备权限，也可以填写相同值
DASHSCOPE_API_KEY_2=your-embedding-api-key
```

常用可选参数及代码默认值如下：

| 分类 | 环境变量与默认值 |
| --- | --- |
| 模型分工 | `CHAT_MODEL=qwen3.7-plus`、`FEEDBACK_MODEL=qwen3.8-flash`、`ENHANCED_MODEL=qwen3.8-max` |
| 模型调用 | `EMBEDDING_MODEL=qwen3.7-text-embedding`、`LLM_TEMPERATURE=0.2`、`REQUEST_TIMEOUT=180` |
| 重试与 Agent | `LLM_RETRY_ATTEMPTS=3`、`LLM_RETRY_MIN_WAIT=2`、`LLM_RETRY_MAX_WAIT=8`、`EMBEDDING_MAX_RETRIES=3`、`AGENT_MAX_TOOL_ROUNDS=12`、`AGENT_MAX_TOOL_CALLS=12`、`AGENT_MAX_NO_PROGRESS_ROUNDS=2` |
| RAG 路由 | `RAG_LARGE_FILE_BYTES=51200`、`RAG_LARGE_CORPUS_BYTES=204800`、`RAG_DATA_DIR=.agents_data/rag` |
| RAG 检索 | `RAG_CHUNK_SIZE=900`、`RAG_CHUNK_OVERLAP=150`、`RAG_RETRIEVAL_K=8`、`RAG_MIN_RELEVANCE=0.35`、`RAG_RELATIVE_MARGIN=0.12`、`RAG_NEIGHBOR_WINDOW=2`、`RAG_MAX_CONTEXT_CHARS=72000` |
| 重排 | `RAG_RERANK_ENABLED=true`、`RAG_RERANK_BACKEND=chat`、`RAG_RERANK_CHAT_MODEL=deepseek-v4.1-flash`、`RAG_RERANK_CANDIDATE_LIMIT=16`、`RAG_RERANK_TIMEOUT_S=15`；专用后端单独保留 `RAG_RERANK_MODEL=qwen3.7-text-rerank` |
| 混合策略 | `RAG_RETRIEVAL_STRATEGY=dense`、`RAG_HYBRID_MIN_RERANK_SCORE=0.5`；此阈值只在 `dashscope` 专用重排时使用，聊天重排为 `rank_only`；关闭重排时混合路径回退向量检索 |
| 记忆 | `MEMORY_WINDOW_SIZE=6`、`MEMORY_SUMMARY_TRIGGER_CHARS=2000` |
| 日志 | `LOG_LEVEL=INFO`、`LOG_PAYLOADS=false`、`LOG_PREVIEW_CHARS=800` |

自然语言反馈解析使用 Flash；资料分析、计划生成、普通交互和对话摘要使用 Plus。Max 仅预留配置与创建能力，页面切换尚未实现，当前不提供 `/max` 命令。模型分工不增加额外路由请求；实际提速、质量与成本变化仍需在线对照验证。以上为代码默认值，环境变量可覆盖；修改配置后需重启程序。

其余资料概览、文件工具和计划证据长度参数可在 `config.py` 的 `Settings` 与 `get_settings()` 中查看。密钥不要写入代码或提交到版本库。

### DeepSeek重排开关（主线与evaluation共用）

默认配置与本地 `.env` 已启用：

```dotenv
RAG_RERANK_ENABLED=true
RAG_RERANK_BACKEND=chat
RAG_RERANK_CHAT_MODEL=deepseek-v4.1-flash
```

`chat` 使用现有 `CHAT_BASE_URL` / `DASHSCOPE_API_KEY`（本项目配置为Token Plan）；关闭思考、一次排列候选、不自动重试。切回专用按量重排只需设 `RAG_RERANK_BACKEND=dashscope`，使用 `RAG_RERANK_MODEL` / `NORMAL_DASHSCOPE_URL` / `DASHSCOPE_API_KEY_2`。完全关闭用 `RAG_RERANK_ENABLED=false`。变更环境配置后重启程序。

聊天输出索引与原片段对应，原文和页码不由模型改写。唯一合法的遗漏列表按原候选顺序补尾；重复、越界、截断、超时等导致主线恢复原向量路径。日志记录模型、补尾数量、token和耗时，不记录正文或思考内容。邻居扩展只让真正命中块继承 `rerank_rank/backend/model`。观察补尾率、失败回退率、Recall@K及延迟，避免把“接口成功”等同“答案正确”。

聊天重排没有 `rerank_score`，不与0.5阈值比较，不伪造概率。`dense` 路径仍保留现有向量筛选；`hybrid + chat` 的候选经过排序与位置去重，但没有专用重排的低分拒绝能力，应补充无关/无答案问题评测。初步模型对照见 `evaluation/experiments/chat_rerank_20261016/DEEPSEEK_RESULTS.md`；已接入运行时不等于无答案过滤已验证。

2026-10-16 核对：18题、每题两轮中，DeepSeek 排序 P50/P95 为1.289/2.187秒，严格 Hit@1 为86.11%，MRR@5为0.9139；36次均返回完整合法排列。专用重排历史同池对照更快且指标略高，但不是本轮同时段重测。采用聊天后端是套餐额度与延迟/质量的工程折中，不宣称免费或全面更优。当前代码的重排、分析失败、评测入口和状态恢复关联测试本次复核73项通过；未重复调用在线模型。

PDF评测 `python -m ai_agent.evaluation.pdf_rag_eval` 默认随配置开启重排并自动开启融合对照，默认候选上限16；顶层指标仍报告纯向量基线，`hybrid` / `rerank` 单独报告。支持 `--rerank-backend chat|dashscope`、`--rerank-model 模型名`、`--rerank-candidate-limit 数量`，参数覆盖只影响该次评测。使用 `--no-rerank` 恢复不调用重排的基线；加 `--compare-hybrid` 可单独观察RRF。

```powershell
# 默认DeepSeek，保留原向量与RRF对照；资料路径填已有索引对应目录。
python -m ai_agent.evaluation.pdf_rag_eval --materials <资料目录> --cases tests/fixtures/evaluation/book_retrieval_holdout_v1.jsonl --output .evaluation-results/deepseek-rerank.jsonl
# 同一命令追加 --rerank-backend dashscope 切专用重排；追加 --no-rerank 关闭。
```

历史独立实验脚本明确指定了模型与端点，以保留原始可复现对照；不会随业务默认值改写历史轨迹。交互评测共用主线检索，配置快照现在包含后端与聊天重排模型。主回答token摘要不包含独立重排调用，不能当作完整账单；PDF评测在 `rerank_total_tokens` 中记录重排用量，并记录后端和补尾数。

混合运行时初步验收：`.evaluation-results/locate-hybrid-runtime-v1.jsonl` 三例均完成检索定位、指定页读取及页级引用支持检查；实际返回的命中标记为 hybrid，未发现回退标记，人工对照记录中的页文本确认关键答案一致。目录题目标页出现于仅关键词召回的命中中，体现补充候选的实际作用。本次聊天总用量 75,965 tokens，旧向量＋重排记录为 68,222；部分查询表达及邻居参数不同，不能视为严格单变量收益结论。仅覆盖三个正例，尚不证明通用质量、阈值已校准或在线失败回退已经验收。

### 分析阶段的工具预算

`AGENT_MAX_TOOL_ROUNDS` 限制外层模型决策轮数，不等于工具执行次数，也不包含模型客户端内部重试。`AGENT_MAX_TOOL_CALLS` 限制实际工具执行尝试总数，一轮批量调用中的每个执行分别计数。单工具额度集中定义在 `agents.py::ANALYSIS_TOOL_LIMITS`：列目录 2、读取文件 6、关键词搜索 4、语义检索 3、文本统计 2；未单独配置的工具默认 4，所有工具同时受总预算约束。

重复判断使用工具名和排序后的参数 JSON，字段顺序不影响判定；不同文件或不同参数不直接视为重复。失败调用也登记签名，以避免同参盲目重试；这项规则适用于当前静态资料分析工具，不应直接套用到未来的实时查询或轮询工具。被拒绝的请求仍回传对应 `ToolMessage`，但不进入下游资料证据。连续无实际执行进展不等于连续没有新知识，也不是连续工具失败熔断。

验收分工：离线测试验证批量总预算、单工具预算、交替重复拒绝、正常不同参数调用和额度用满后收尾；在线输出验证正常分析与计划链路，成品中出现引用不等于引用已经逐条核验。当前已向模型告知初始上限，并在每次模型决策前更新剩余总额度、单工具有效剩余额度和含本轮在内的剩余模型轮数；更新系统消息而非累计过期提示，不污染下游资料证据。剩余额度提示已通过离线测试，在线决策收益尚未确认；硬限制不依赖模型遵守提示。

### 首次 API 连接慢：可选 IPv4 策略

若首次聊天或 Embedding 请求有约 42 秒的额外等待，而后续请求很快，可在本地 `.env` 设置 `API_FORCE_IPV4=true` 后重启。当前网络曾出现两个 IPv6 地址各等待约 21 秒再回退 IPv4；开启后，同步聊天客户端和 DashScope Embedding session 都绑定 IPv4 源地址，使 IPv6 候选立即失败后尝试 IPv4。它不修改系统 DNS、不硬编码服务 IP，也不影响查询/文档编码、证据策略和模型重试。

默认 `false` 保留原有连接行为；设回 `false` 并重启即可回退。日志会显示两类客户端的 IPv4 策略；比较新进程首次调用和空闲后重连耗时，确认约 42 秒等待消失。聊天客户端保留 API 地址对应的环境/系统代理和 `NO_PROXY` 选择，Embedding 保留 Requests 的代理行为；连接到代理时约束的是代理连接。当前项目使用同步 `invoke`，该开关不覆盖未来新增的异步聊天调用。它解决连接等待，长上下文和长输出的生成时间仍需单独优化。

## 运行

请在项目根目录执行。固定流程示例：

```powershell
uv run python -m ai_agent.main --materials .\book --goal "深入理解资料并生成适合新手的学习计划" --days 3 --output .\plans\study_plan.md --mode pipeline
```

Agent 模式并开启可解释调试日志：

```powershell
uv run python -m ai_agent.main --materials .\book --goal "深入理解资料并生成适合新手的学习计划" --days 3 --output .\plans\study_plan.md --mode agent -v
```

生成后进入多轮对话：

```powershell
uv run python -m ai_agent.main --materials .\book --goal "深入理解资料并生成适合新手的学习计划" --days 3 --output .\plans\study_plan.md --mode pipeline --interactive --memory-strategy summary
```

`--days` 允许 `1` 至 `30`；`--memory-strategy` 可选 `full`、`window`、`summary`。交互模式中输入 `exit`、`quit` 或 `q` 后，程序才会把最终状态保存到输出文件。

`--output` 和 `RAG_DATA_DIR` 的相对路径始终基于包含 `pyproject.toml` 的项目根目录解析，不受终端或 IDE 当前工作目录影响。默认计划输出到 `./plans/study_plan.md`，RAG 索引写入 `./.agents_data/rag`；传入绝对路径时仍使用指定位置。

默认情况下，即使使用 `-v` 也不会打印完整向量、资料正文或完整模型消息。仅在确需排查提示词内容时临时设置 `LOG_PAYLOADS=true`；日志只输出受 `LOG_PREVIEW_CHARS` 限制的预览，但仍可能包含资料片段，调试完成后应恢复为 `false`。

默认不在终端重复打印已保存的整份分析与计划；文件内容保持完整，需要终端全文时追加 `--show-output`。交互模式仍展示初始计划、正常问答、反馈确认和状态保存结果，避免隐藏用户需要确认的信息。

`evaluation.runner`、`evaluation.interactive_runner`、`evaluation.pdf_rag_eval`、`evaluation.review` 和 `evaluation.calibration` 也支持 `-v`，共用日志级别配置。交互评测默认保留每题状态、耗时、已记录聊天token；逐项引用检查、位置传递与预算详情在 `-v` 下显示。校准保留进度、每题一致性摘要与最终指标，逐项判定/规则变化在 `-v` 下显示。JSON/JSONL结果、审计轨迹和评测判定没有精简或删除。

本地 `.env` 的调试默认值已恢复为 `LOG_LEVEL=INFO`、`LOG_PAYLOADS=false`；显式设置 `LOG_LEVEL=DEBUG` 仍会开启详细日志。修改后重启进程。日志中的token是单次已记录模型响应，不是整次Agent运行的完整费用。此次输出调整记录见 `docs/experiments/console-output-2026-10-16.md`。

## 开发协作

### 独立模型评审：先读取原始资料

评审不参与生成或自动重写。`review` 默认从案例的 `material_dir` 读取资料，也可通过
`--materials` 指定一个原始文件或目录。`cited` 模式直接提取最终计划引用的 PDF 原页与
相邻页，以及被引用的 MD/TXT/Python 文件正文，不使用生成者摘要或向量检索替代原文。

先检查实际证据，不调用聊天或 Embedding 模型：

```powershell
uv run python -m ai_agent.evaluation.review --results .evaluation-results/react-e2e-v1.jsonl --cases tests/fixtures/evaluation/end_to_end_smoke.jsonl --output .evaluation-results/react-e2e-v1-evidence-v2.jsonl --materials src/ai_agent/eval/ai_agent_book --dry-run
```

确认输出中的 `source_evidence.documents` 有原文，并检查 `issues` 后，再执行在线评审：

```powershell
uv run python -m ai_agent.evaluation.review --results .evaluation-results/react-e2e-v1.jsonl --cases tests/fixtures/evaluation/end_to_end_smoke.jsonl --output .evaluation-results/react-e2e-v1-llm-review-v2.jsonl --materials src/ai_agent/eval/ai_agent_book --llm
```

`--evidence-mode full` 会读取指定文件/目录中所有 PDF、MD、TXT、Python 正文；不是默认模式。
默认原文证据预算为 150000 字符（`--max-evidence-chars`），超出时停止，不静默截断。
字符预算不等于模型 token 上限，长书应拆分评审，不应盲目加大预算。

原文、实际读取页、文件指纹、缺页/提取失败信息保存在本地评审记录中。PDF 页从 1 开始计数，
chunk 编号未验证，图像和扫描页未做 OCR。没有证据或证据不全时不能将所有引用标为通过；
邻页支持也不能证明被引用页支持。模型建议仍需人工确认，不是金标准。
在线评审会将提取的正文发送给配置中的模型服务；源码和私密资料应先检查敏感内容。
评审 JSONL 含原文，不应公开提交；现有 `.evaluation-results/` 已被 Git 忽略。

项目的分支约定、日常提交、同步、合并、冲突处理和发布流程参见 [Git 与 GitHub 分支工作流操作手册](../../docs/git-workflow.md)。

### PDF 检索评测标签

`gold_evidence` 保留原始指定页和历史 recall/MRR 口径；
`acceptable_evidence` 记录经原文核实、每页均可独立回答的替代页。
评测另报 `acceptable_mrr_at_k`，不将替代页混入旧 recall 的分母。
页级命中不等于最终回答正确；重排已接入业务检索，混合策略仍保留开关和向量基线，排序收益不自动等于最终回答质量提升。

## 完善路线

当前优先级以以下步骤为准；旧路线图中的日期和“未完成”描述不代表当前状态。

1. **近期：通用学习适配（待实施）**。取消文件后缀对主题的推断；资料分析按目标选择学习活动，不固定要求实践项目；计划中的完成标志可以是口头解释、阅读自检、观点比较、作品或运行结果，而不是一律要求代码测试。补充非技术资料与“同一资料、不同目标”的对照案例。
2. **随后：最小反馈闭环（规划中）**。记录用户确认的已完成内容、当前困难和约束变化，只调整必要任务，并保留调整理由。区分“已阅读”“用户自报理解”和“有检查依据的理解情况”；不凭模型猜测认定掌握。
3. **下一阶段：显式状态编排（待实施）**。现有学习状态已支持快照与恢复；先以可选 LangGraph 入口拆出目录校验、资料分析、计划生成节点，保持原返回值、异常、工具预算和检索实现，并保留 legacy 基线。首步不承诺 checkpoint 或失败续跑，随后再按真实恢复需求完善。MCP 用于资料工具的标准化接入，而非为增加框架数量。

验证时优先比较目标满足、引用支持、反馈调整是否误改已完成任务，以及延迟和调用成本。`runner` 的 `success` 仅表示生成非空计划且未抛出异常，不代表计划质量达标或用户已学会。Judge 结果仍需人工抽检；宣称提高学习效果需要另外的真实用户研究。

- [一周短期计划：建立可量化基线](../../docs/short-term-roadmap-one-week.md)
- [长期路线：从可运行 Agent 到可评测的工程作品](../../docs/long-term-roadmap.md)



### 评审器校准（实验中）

固定正反例位于 tests/fixtures/evaluation/judge_calibration_cases.jsonl。
先执行离线证据检查：

    uv run python -m ai_agent.evaluation.calibration --output .evaluation-results/judge-calibration-evidence-v1.jsonl

再按需执行在线校准，会调用配置中的聊天模型：

    uv run python -m ai_agent.evaluation.calibration --output .evaluation-results/judge-calibration-online-v1.jsonl --llm

agreement_rate衡量与人工标签一致的比例；false_pass_rate衡量错误被放过；
false_reject_rate衡量正确结果被误拒；unknown_rate与judge_failed单独展示。
未在线评审时相关比例为null，不代表0错误。人工标签不发送给模型。
PDF仅做文本/页级核对，chunk与图像未验证；JSONL含原文，不公开提交。

```
在线评审保存规则处理前的模型结构化判断、最终判断和判定变化记录。
校准报告分别统计同一批案例的原始成绩与最终成绩。
旧实验缺少原始快照，不能用于还原纯模型成绩。
评审标准 v4 明确区分目标要求与计划内部矛盾，并约定实践维度的
false/null 边界。没有实践要求且未安排实践时记 true，理由注明不适用，
不表示实践已执行。校准集包含 16 条案例、22 个人工标注项；
准确率与稳定性以实际在线实验结果为准。
校准入口向 stderr 即时输出准备、证据读取、模型评审和保存阶段，
展示每案例耗时、最终判断、标注分歧及规则变化；指标仍输出到 stdout。
进度不包含完整计划或资料正文，模型等待阶段不提供内部实时进度。
```
