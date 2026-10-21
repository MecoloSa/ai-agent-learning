# 证据驱动的本地学习智能体

> 面向本地多类型学习资料，提供有来源依据的问答与学习计划，并根据用户确认的进度、困难和时间约束，持续调整学习安排。

本项目面向研究生和自学者：根据本地资料与学习目标生成有来源依据的学习路径，并通过多轮对话解释资料、调整目标和更新计划。学习内容不限于编程；阅读理解、观点比较、技能练习和代码实现应采用不同的学习活动与完成标志。

长期目标是形成“资料依据 → 初始路径 → 用户反馈 → 必要调整 → 进度延续”的闭环。当前已实现资料分析、计划生成、会话内目标/计划更新、手动任务登记与确认反馈；含资料关联的存档可直接--resume-state继续问答，自动核对目录/指纹，--state-only仅管理记录。full/window/summary记忆已持久化并默认沿用原策略；每次运行更新自己的current存档，/save-state保留独立历史快照，恢复来源不覆盖。用户确认上一轮修改与测试通过。旧工具证据仍未恢复，不将这些能力等同于完整长期记忆。

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

已实现 `--orchestrator legacy|langgraph`，默认保留 `legacy`。`orchestration.py` 将目录校验、资料分析、计划生成组织为三个节点：目录无效经条件边直接结束，分析异常继续抛出并阻止规划；节点仅返回变化字段，沿用原三字符串返回值与审计对象。`--mode pipeline|agent` 决定节点内部分析方式，与外层编排选项独立；DeepSeek 重排继续复用原业务函数。

在线烟测 `langgraph-deepseek-smoke-v1.md` 对应日志确认三节点依次完成、四次 DeepSeek 聊天重排正常返回、两个主回答请求完成并导出三天计划。本次实际为向量召回＋聊天重排，不是混合检索对照。粗略墙钟时间220秒，主回答用量67,649 tokens、重排27,950 tokens，合计已记录95,599 tokens，不含Embedding等未记录用量，不等于账单。节点、输出展示、重排及关联入口本次复核92项离线测试通过；未重复在线调用。

默认图仍为单次运行，底层可选内存或SQLite检查点，普通CLI可显式恢复失败规划，见下文；不自动重试，未迁移交互循环。节点完成不代表答案已经语义评审；尚无同条件 legacy 在线对照，不宣称速度或回答质量提升。

2026-10-17 已实现统一计划输出检查，覆盖无检索、直接回答与预算强制收束三个出口：空文本抛出 `PlanExecutionError`，CLI停止交互与导出并非零退出；常见任务段落逐任务检查非空学习内容、行动、完成标志和引用标记，其余布局仅提示人工核对。保留原文、不自动重试，诊断单独写入 `plan_output_check`，不等同事实或引用支持度评测。用户确认离线及在线测试完成；本次检查当前代码与关联离线回归，不重复在线请求。

已实现三个图节点的统一观察包装：记录运行ID、输入/输出字符数量、耗时、返回或失败状态及程序错误码，保留原异常、不重试；调用方通过 `workflow_trace["latest_run"]` 接收内存轨迹。目录拒绝标记为 rejected，与节点异常 failed 区分；returned/completed 不代表质量通过。用户确认正常在线运行、假业务失败与关联离线验收通过。默认终端保持简洁，详细摘要用 `-v`，失败定位始终可见；观察包装本身不自动写轨迹文件或提供恢复，检查点由底层入口显式启用。

批量 `evaluation.runner` 已增加可选 `--orchestrator`，默认 legacy；编排方式及图轨迹接入现有结果JSONL，保留已捕获异常时的节点记录。用户确认实验成功，观察到分析产出与规划输入摘要一致、规划产出增长。自定义四参数 execute 仅用于 legacy，避免把假执行标成图运行。轨迹随批次正常写入，不承诺进程强杀时保存，不是失败续跑；原报告和证据审计可能含资料正文，不能因节点摘要脱敏就公开整份结果。

底层图入口已可选启用 `InMemorySaver`，用户确认同进程规划失败恢复及关联测试通过：要求同一saver和thread_id、参数一致且检查点确有失败的plan任务；拒绝覆盖已有任务，不自动重试，每次尝试独立记录。内存实现本身不支持进程重启；SQLite实现见下文。普通CLI恢复入口见下文，评测尚未增加恢复参数，不支持分析节点恢复。checkpoint保存完整图状态，与只保存长度的轨迹不同。

checkpoint路径已增加受支持资料的相对文件名＋内容指纹，绑定分析产出并在分析/规划阶段及恢复前核对；修改、增删、重命名拒绝继续混用旧分析，缺少指纹的旧检查点要求新建任务。原假目录测试使用固定指纹替身，独立测试使用临时真实文件；只有业务模型是假的，文件操作和指纹比较是真实的。默认非checkpoint流程不增加文件哈希。指纹不是资料副本，前后核对不等于并发写入下的原子快照。

底层图已接收LangGraph检查点接口，新增显式SQLite连接上下文；用户确认实现与测试通过。两个独立Python进程验证首次规划失败后退出、重开同一数据库仅重试规划；资料变化仍拒绝恢复。sqlite适配器固定3.1.1，与当前checkpoint4.2.0兼容，不顺带升级框架。模型配置/自动代码指纹和跨次完整审计仍未绑定，不能据此宣布完整生产级恢复。数据库含完整分析与资料证据，默认放在已忽略的`.agents_data/checkpoints/`，不得公开提交或加载陌生来源的检查点文件。

已实现显式工作流兼容版本保存及恢复前核对；状态结构或分析产物含义发生不兼容变化时主动升级版本，缺失/不匹配的旧检查点要求新建任务。对应版本拒绝测试已存在，名称为`test_incompatible_revision_blocks_resume_without_overwrite`，用`-k workflow_revision`会漏选，应运行整文件。该版本不是自动代码哈希，也不检查模型配置，不因日志或注释调整自动使检查点失效。

2026-10-18 普通CLI已显式接入SQLite新任务/失败规划恢复，要求`--checkpoint-db`与`--workflow-id`配对、`--orchestrator langgraph`，恢复额外指定`--resume-workflow`并重复原任务参数；不与`--resume-state`或交互模式混用。恢复不存在的数据库在打开连接前拒绝，不静默创建空库；原默认执行保持不变。入口使用当前运行配置，不承诺复原模型/检索设置或并发安全。9项CLI测试复核通过；用户在线新任务`react-20261018-01`完成三个节点并导出。只读核对数据库335,872字节，含`checkpoints`、`writes`表及该任务5条检查点；SQLite不是文本文件，编辑器预览空白不代表没有保存。该次成功运行不额外证明真实在线故障恢复。

已新增独立只读检查点摘要命令`python -m ai_agent.checkpoint_inspect --checkpoint-db 路径 --workflow-id 任务ID`，用SQLite只读连接查看指定任务根命名空间的记录数量和最新检查点ID，不反序列化正文、不执行图或调用模型。用户查询`react-20261018-01`返回5条记录及最新/父检查点ID。不把记录存在判为任务可恢复；可恢复性仍由正式入口核对。当前不扩建全文数据库编辑器，不直接修改序列化检查点或盲删单条记录。

显式检查点CLI已在图尝试结束后，将现有节点摘要写为独立JSON轨迹；以数据库路径＋任务ID关联同一工作流，以run_id区分失败与恢复尝试，不保存资料/模型正文。不改变执行结果，轨迹写入失败仅警告，不掩盖原业务异常；不承诺进程强杀时留存。源码及相关测试已接入，不新增数据库管理界面。

2026-10-18 方向调整：暂不实施独立`/suggest`命令或建议模型，复用普通问答的最新状态与现有资料工具，先验证方法建议、依据资料的补学建议和“仅讨论不修改”三种请求。`--resume-state`当前只恢复状态管理，不能用于这组普通问答验证；带检查点的启动也暂不支持`--interactive`。

计划更新确认边界已实现：复用现有模型和Markdown标记，拒绝重复/缺失/嵌套标记；纯函数校验候选后显示差异与完整候选，只接受本次明确`y`再采用。目标与计划一并确认，取消/非法/无变化不改生效修订号、不保存。聊天记忆在确认结论之后记录程序结果，避免摘要先把未采用提案当成已生效；两处既有更新会话测试已补确认输入。修正了会话返回语句误缩进至循环内导致的提前退出/隐式None，用户确认离线测试通过。该保护不等于计划语义或引用正确性校验，不支持异步确认或跨会话待办提案。

用户已提供在线预览及最新学习状态恢复记录：快照为v2，修订号2，初始计划3,142字符、当前计划3,856字符，目标不变；任务仍为completed/self_reported_unclear并保留原反馈，tasks_need_review为true。恢复主要路径符合预期；附件预览停在取消输入，未单独包含取消后的状态，不用最终恢复记录替代每个分支的证据。候选仍含“正常结束等于正确完成”等过强表述，确认保护不等于内容质量验收。

下一局部提案（尚未实施）：默认确认界面取消开发者行级diff和重复打印完整候选，只展示变更字段、修订号与保留范围；输入v按需显示程序解析后的候选，再由y确认。复用上方模型回复但不依赖模型自己生成的变更总结，不增加模型调用，不改变状态/保存规则。同时纠正状态恢复提示中“退出前必须手动保存”的旧表述；已有变更仍自动保存，失败时手动重试。

2026-10-19 路线收敛：精确字符串局部替换不再作为必做前置。主线保持“恢复后继续学习→可靠反馈调整→可查询、可纠错的长期记忆→小范围MCP→工程收敛，按收益决定多Agent”。资料关联/v3快照已落地，用户确认离线与在线通过。撤回未落地的“固定六轮”方案；新提案在ConversationMemory内导出/还原strategy、window_size、max_chars、human/ai消息、summary及归档prompt_history，v4兼容读取v1/v2/v3。恢复默认沿用原策略，显式--memory-strategy才切换；新任务仍默认window。当前system按有效状态重建，旧prompt_history仅归档，旧工具证据不重放。确认/取消确定后复用add_ai及原压缩逻辑，再将实际记忆与有效状态同存；摘要异常时finally保存已追加的完整问答，仍保留原异常传播。window目前仅限制发送视图，内部完整消息也会持久化；full及window存储增长不被偷偷截断。独立候选关联验证169项通过，业务源码未写回。快照包含对话文本及可能含资料的提示词历史，勿公开提交或加载不可信快照。

2026-10-19 当前存档接线（本轮提案）：用户确认存储层修改及离线验收通过。拟让普通问答、状态管理在每次运行中复用一个current/下的存档；恢复时资料关联变化与后续问答共享同一路径。每次重启另开写入分支，绝不覆盖被选中的恢复文件；/save-state仍生成独立历史快照，旧文件不自动删除。未变更的查询不写盘，完整问答仍保存原记忆策略。新增业务接线测试覆盖一文件多次更新、历史独立、两次运行隔离、问答消息完整、资料重新关联共用路径。业务源码未直接写回；进程内候选关联194项、业务接线5项通过，待用户应用后验收。

2026-10-19 当前存档接线验收：用户确认关联离线与--state-only实测通过；现有代码已将普通会话、状态管理和恢复时资料关联变化接到会话专属current路径。current保存本次运行最后一次成功写盘的完整状态，/save-state另存独立历史副本；两者格式与恢复入口相同。每次恢复另开写入分支，读取来源不会随本次学习更新。状态管理模式保留快照已有记忆数据，但不运行普通问答；当前“不包含聊天历史”的展示文案不应解释为删除已有记忆。

2026-10-19 存档发现入口（本轮提案，待应用）：拟新增只读python -m ai_agent.state_catalog --limit 10，按最后修改时间展示current及外层历史/旧版文件的目标、修订号、任务数、记忆策略和完整路径。仅扫描两层指定目录、不另建索引、不自动选最新文件恢复；坏文件明确标记，目录访问失败不冒充空目录。复用load_learning_state校验，既有保存/恢复逻辑不改。新增tests/test_state_catalog.py；独立候选关联45项通过，业务模块未直接写回。此小步后停止扩展存档管理，回到反馈驱动调整的固定业务验收。

2026-10-19 存档发现已验收：用户确认上一轮通过，只读入口使用python -m ai_agent.state_catalog --limit 10；不继续扩建存档管理。主线转回反馈驱动的必要调整。本轮提案仅新增FEEDBACK_ADAPTATION_RULES并接入共用交互提示词，保留完整候选与确认机制，不新增建议Agent、状态字段或重排实验。规则强调阅读完成不等于掌握、限定调整范围、区分已花费/剩余时间、针对困难选择活动、可观察的完成检查与证据边界。tests/fixtures/evaluation/feedback_adaptation提供可复现初始状态及自编演示资料，不冒充书籍原文；候选离线关联74项通过，真实建议的针对性与时间约束仍待在线人工验收，不能把提示词约束当成程序强校验。

## 主要模块

### 目录导航（2026-10-20）

本次仅按职责移动原一级 Python 文件，文件名、业务流程、参数默认值和存储格式不变；不搬动已有子目录中的文件。分组借鉴 [Deep Agents 的职责分离](https://github.com/langchain-ai/deepagents/blob/main/libs/ARCHITECTURE.md)，不引入其框架或中间件。

```text
src/ai_agent/
├── main.py / config.py                # 主入口与配置（位置不变）
├── checkpoint_inspect.py / state_catalog.py  # 原只读命令（位置不变）
├── runtime/          # agents、编排、执行保护、计划校验、提示规则、轨迹
├── retrieval/        # rag、embedding、BM25、RRF、重排
├── learning/         # 学习状态、反馈解析/提案/确认、任务创建
├── persistence/      # 会话记忆、存档、资料绑定、检查点
├── resources/        # 资料路径保护、PDF读取、工具封装
├── infrastructure/   # 网络连接与终端日志配置
└── evaluation/ 等已有目录             # 保持原有位置与内部组织
```

原命令 `python -m ai_agent.main`、`learning-agent`、`python -m ai_agent.state_catalog`、`python -m ai_agent.checkpoint_inspect` 及 `ai_agent.evaluation.*` 均保留；`.env` 仍放在本目录，资料/索引/存档/数据库位置不变。移动模块的 Python 导入使用新路径，例如 `from ai_agent.retrieval.rag import LearningRAG`、`from ai_agent.runtime import agents`；不再使用旧的扁平导入。项目源码、评测脚本、测试及跨进程测试脚本已同步调整。日志继续使用原名称，避免改变过滤配置；下文历史记录中的裸文件名按本导航查找。

迁移验收：原测试基线756通过、1跳过；迁移后连同7项目录回归共763通过、1跳过，另有聊天重排实验目录17项通过。8个既有命令入口的`--help`烟测通过，包含主命令、两个只读查询及主要评测入口。验证不请求在线模型，不把目录整理宣称为回答质量或性能提升；三项原有依赖弃用警告未在本步处理。

2026-10-19 记忆持久化落地核对：五种回答出口已统一接入record_ai_response，完成确认/取消处理后导出记忆并保存；用户确认关联离线与在线验收通过。验收要求一轮结束即保存，不能依靠下一次普通问答补保存。

2026-10-19 保存策略下一步（提案，尚未接入运行时）：区分可更新的当前存档与不可覆盖的历史快照。先在state_store复用临时文件、flush/fsync及失败清理，只为当前存档新增os.replace发布；历史save_learning_state及/save-state继续禁止覆盖。本轮只补磁盘能力，自动保存仍沿用原路径，不宣称已减少文件数量。新增tests/test_current_state_store.py，检查连续替换、首次/已有文件保存失败、发布前完整性与历史保护；内存候选加现有存储测试21项通过，业务源码由用户应用。下一步再统一接入问答、状态管理及资料重新关联入口，存档路径不得取自被选中的历史文件。

| 文件 | 职责 |
| --- | --- |
| `main.py` | 命令行入口、日志初始化、交互会话和结果保存 |
| `runtime/agents.py` | 模型调用、重试、两种资料分析模式、学习计划生成和引用汇总 |
| `runtime/orchestration.py` | 可选 LangGraph 外层状态、三节点流程和目录校验条件边 |
| `infrastructure/console_logging.py` | 统一终端日志级别与依赖降噪，不自动配置日志文件 |
| `retrieval/rag.py` | RAG 路由、资料概览、增量索引、阈值检索和邻居扩展 |
| `retrieval/embeddings.py` | DashScope Embedding 分批适配，单批最多 20 条 |
| `persistence/memory.py` | `full`、`window`、`summary` 对话记忆及目标/计划状态历史 |
| `resources/tools.py` | 文件列举、受限读取、关键词检索、文本统计和语义检索工具 |
| `resources/materials.py` | 支持类型、目录遍历和路径越界保护 |
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

当前 `configure_console_logging()` 使用 `logging.basicConfig()` 配置终端日志，没有 `FileHandler`：默认被级别过滤的 DEBUG 明细不会自动落入 `.log` 文件。生成报告保存为 Markdown，评测结果/审计记录按各入口写入 JSON/JSONL；如需留存终端调试日志，应另行配置文件日志或显式重定向。检索命中预览仍可能包含少量资料正文，`-v` 不应视为完全无敏感内容。

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

当前项目采用 [develop/product 工作流](../../docs/git-workflow-develop-product.md)：日常开发使用 `develop`，验证后从 `develop` 更新 `product`，保留原 `main`。2026-10-16 本地核对时，当前分支为 `develop`，跟踪 `origin/develop`；`develop`、`product` 及其本地远端跟踪引用均指向 `afcaf5f`。未额外联网复核远端；本机当前分支不等于 GitHub 网页默认分支。

在 `develop` 上执行不带目标的 `git push`，通常依据其上游推送到 `origin/develop`；不要理解为从任意分支都强制推到 develop。本次 README 更新未自动提交或推送。通用提交、合并和冲突处理另见 [Git 与 GitHub 分支工作流操作手册](../../docs/git-workflow.md)。

### PDF 检索评测标签

`gold_evidence` 保留原始指定页和历史 recall/MRR 口径；
`acceptable_evidence` 记录经原文核实、每页均可独立回答的替代页。
评测另报 `acceptable_mrr_at_k`，不将替代页混入旧 recall 的分母。
页级命中不等于最终回答正确；重排已接入业务检索，混合策略仍保留开关和向量基线，排序收益不自动等于最终回答质量提升。

## 完善路线

2026-10-21 当前局部提案（业务源码待应用）：针对“已登记任务仍被说成待确认”，保留每轮最新状态注入，补充确认后的结构化学习操作结果，明确操作类型、对象、实际任务状态及候选已结束；不将内存确认等同于磁盘保存。三种记忆策略不变，summary明确优先保留程序结论，历史结果不能覆盖当前任务表。不新增取消工具、不改v5格式、不增加模型调用；新增`tests/test_learning_action_history.py`检查确认/取消、下一轮接线及摘要输入，在线仍需人工核对是否减少过时描述。

当前优先级以以下步骤为准；旧路线图中的日期和“未完成”描述不代表当前状态。

2026-10-20 接口决策修正（当前方案，待用户应用）：经用户质疑，撤回将引号兼容作为必做修复的建议。程序已有真实用户来源表，无需要求模型逐字复制quote；反馈及创建任务参数仅保留source_id和操作字段，未知来源/越权字段仍拒绝。确认界面展示所选完整原话及候选变化；来源真实不证明模型的标题/状态正确，确认不可省略。既有v5的quote由程序填入所选原话，旧存档不迁移；上下文仅在quote=text时省去重复字段，旧的不同引用片段保留。source_quotes旧模块不再依赖，未删除用户文件；测试改验接口和程序附加原文。候选关联249项通过；同一真实失败输入一次在线复测仅提交title/source_id，生成PendingTaskCreation，原文保留，未确认或保存。该单例不等于稳定成功率；不增加预算、不放开来源边界。

2026-10-20 创建任务线上定位：用户完成代码与离线测试后，实际创建未形成pending。对同一存档/输入的只读在线复现发现模型把中文引号改成英文引号，严格子串拒绝后又提交同参，被重复调用保护拦截；不是预算不足或语义同名导致。候选修复共用source_quotes，仅兼容单双引号排版、要求唯一匹配并恢复真实原文，仍拒绝字词、数字、空白与否定词改写。反馈和任务候选统一使用，提交时原有严格检查保留；不改v5格式、不增加预算、不修改重复保护。候选关联248项通过；同输入一次在线复测生成PendingTaskCreation（一次工具、两次模型响应），未确认、未登记或保存，不据此宣称稳定成功率。业务源码由用户应用；同步修正“任务需手动登记”的旧启动文案。

2026-10-20 验收更新：用户确认自然反馈来源表、承接上文及执行声明纠正的修改、离线和实际测试全部通过。当前已支持受控自然反馈，不再将后文历史提案的“待接入”视作现状。本轮下一步为自然语言创建任务：独立propose_task_creation与反馈共用原话来源和一个待确认名额，ID/初始状态由程序生成，确认后复用登记、记忆和保存；不改Markdown计划、不自动开始/完成。只做归一化同名拒绝，不宣称语义去重；反馈与创建声明按实际pending类型核对。v5结构不变，/task保留，旧FeedbackInbox导入兼容。业务源码仍由用户应用；内存候选关联236项通过，含三种策略确认/取消、下一轮读取及恢复后不重复创建。此后先验证新任务可用已有反馈入口完成，再按真实需求处理取消语义；不提前建设调度或通用操作框架。

2026-10-20 自然反馈接线：用户确认上一轮修改及离线通过，在线暴露“口头称已提议但无pending”与“承接上文却只能引用本轮”的问题。本轮待应用修正区分资料工具与反馈工具触发条件；反馈来源由程序提供本轮及原记忆策略可见的最近4条用户消息，单条展示最多2000字符，拒绝助手、摘要及未知来源。source_id默认current兼容原调用，确认后仍保存所选完整用户原文，不升级v5存档。针对已观察到的正向提议声明最多纠正一次，复用原工具预算；持续无pending则返回程序失败说明。声明匹配不是完整意图识别，不保证所有漏调都自动修复。内存候选关联206项通过，需实际在线复测；业务源码未直接写回。短期与长期路线文档按用户要求，仅在明确请求时更新。

2026-10-20：用户已验收feedback_proposals契约和离线测试。本轮待应用方案将propose_learning_feedback接入主问答的有界工具循环，每轮最多一个有效待确认提议，不在工具内部修改状态。确认后保留原话、范围及版本，任务反馈复用apply_user_feedback；当前安排原话不强制分类或绑定任务，默认仅记录当天及同一计划版本主动使用，确认时明示。拟采用v5保存，兼容读取v1至v4；旧程序不保证读取v5。pending不等于写入，内存生效不等于磁盘保存。反馈与计划同轮候选暂时全部不执行，提示分开处理。业务源码未直接写回；内存候选关联241项通过，真实语义与在线验收仍待验证。其后再支持自然语言创建任务和有区分的完成/取消，不扩大本轮反馈工具职责。

1. **恢复与存档（已验收）**。关联资料的存档可继续问答，恢复原full/window/summary记忆策略；当前存档持续更新，手动快照保留历史，state_catalog只读查找。旧工具证据不重放，恢复不重新生成计划。
2. **自然反馈已验收，当前接自然语言创建任务**。共用来源校验、确认与保存边界；记录、登记和修改计划的授权分开。随后验证新任务反馈及必要的计划调整，不把反馈穷举为固定分类或固定学习策略。
3. **长期：记忆与标准化接入**。先完成历史信息的读取、更新、纠错/删除，再做小范围MCP；已有LangGraph三节点、轨迹和受限失败恢复继续复用，不整体迁移聊天。多Agent仅在上下文隔离或并行收益明确时引入。

验证时优先比较目标满足、引用支持、反馈调整是否误改已完成任务，以及延迟和调用成本。`runner` 的 `success` 仅表示生成非空计划且未抛出异常，不代表计划质量达标或用户已学会。Judge 结果仍需人工抽检；宣称提高学习效果需要另外的真实用户研究。

- [一周短期计划：打通持续学习闭环](../../docs/short-term-roadmap-one-week.md)
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
