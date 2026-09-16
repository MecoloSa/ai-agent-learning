# 本地学习资料分析与学习计划助手

这是一个面向本地学习资料的双智能体项目。系统先分析资料结构与证据，再生成按天拆分、包含可验证产出的学习计划；生成计划后还可以进入多轮对话，继续解释资料来源、调整学习目标或更新计划。

当前支持 `.txt`、`.md`、`.py` 和 `.pdf`。所有文件访问都限制在用户指定的资料目录内；PDF 通过 `pypdf` 读取文本层，扫描版 PDF 暂不支持 OCR。

## 当前能力

- **两种执行模式**：`pipeline` 使用固定流程收集资料并完成分析；`agent` 允许模型自主选择只读工具，并通过最大轮数、重复调用检测和异常保护避免失控。
- **自适应资料处理**：小型文本资料直接读取文件清单和有限摘录；存在 PDF、单个大文件或大规模语料时，自动切换到 RAG。
- **可解释 RAG**：使用 Chroma 增量索引和余弦相似度检索，结合绝对阈值、相对阈值、位置去重与相邻 chunk 扩展；上下文保留 `source`、`page`、`chunk`、命中角色和相关性分数。
- **多角度初始检索**：`pipeline` 模式下将大资料检索拆分为 `goal`、`structure`、`implementation`、`practice` 四类查询，并将确定性资料概览与正文证据同时交给资料分析智能体。
- **证据驱动计划**：学习计划智能体同时接收分析结论和带来源的证据，要求任务标注引用；资料不足时应明确说明，而不是补写无依据内容。
- **多轮交互与记忆**：`--interactive` 支持 `full`、`window`、`summary` 三种短期记忆策略，并配上”目标更新“与”计划更新“内嵌功能。大资料场景会依据每轮用户问题重新检索，计划或目标更新后同步写回会话状态，退出后保存最终版本。
- **受控调试日志**：普通日志保留阶段、查询、阈值、命中、引用、耗时和 token 等判断信息；`-v` 增加模型与检索调试信息，同时抑制 HTTP、OpenAI、Chroma 等第三方日志刷屏。

## 执行流程

1. 校验资料目录及支持的文件类型；目录无效时提前结束，不调用计划智能体。
2. 根据资料规模选择直接摘录或 RAG；RAG 索引按文件哈希增量同步。
3. 资料分析智能体基于概览、工具结果或检索证据生成知识结构分析。
4. 学习计划智能体结合分析结果与原始证据生成指定天数的计划。
5. 可选进入交互模式；退出后将最终目标、资料分析和学习计划写入 Markdown。

`pipeline` 模式通常包含资料分析和计划生成两次聊天模型请求，但 RAG 还会调用 Embedding 服务；`agent` 模式的聊天模型请求次数由工具调用轮数决定。

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
| 模型 | `CHAT_MODEL=qwen3.7-max`、`EMBEDDING_MODEL=qwen3.7-text-embedding`、`LLM_TEMPERATURE=0.2`、`REQUEST_TIMEOUT=120` |
| 重试与 Agent | `LLM_RETRY_ATTEMPTS=3`、`LLM_RETRY_MIN_WAIT=2`、`LLM_RETRY_MAX_WAIT=8`、`EMBEDDING_MAX_RETRIES=3`、`AGENT_MAX_TOOL_ROUNDS=12` |
| RAG 路由 | `RAG_LARGE_FILE_BYTES=51200`、`RAG_LARGE_CORPUS_BYTES=204800`、`RAG_DATA_DIR=.agents_data/rag` |
| RAG 检索 | `RAG_CHUNK_SIZE=900`、`RAG_CHUNK_OVERLAP=150`、`RAG_RETRIEVAL_K=8`、`RAG_MIN_RELEVANCE=0.35`、`RAG_RELATIVE_MARGIN=0.12`、`RAG_NEIGHBOR_WINDOW=2`、`RAG_MAX_CONTEXT_CHARS=16000` |
| 记忆 | `MEMORY_WINDOW_SIZE=6`、`MEMORY_SUMMARY_TRIGGER_CHARS=2000` |
| 日志 | `LOG_LEVEL=INFO`、`LOG_PAYLOADS=false`、`LOG_PREVIEW_CHARS=800` |

其余资料概览、文件工具和计划证据长度参数可在 `config.py` 的 `Settings` 与 `get_settings()` 中查看。密钥不要写入代码或提交到版本库。

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
uv run python -m ai_agent.main --materials .\book --goal "深入理解资料并生成适合新手的学习计划" --days 3 --output .\plans\study_plan.md --mode pipeline --interactive --memory-strategy summary -v
```

`--days` 允许 `1` 至 `30`；`--memory-strategy` 可选 `full`、`window`、`summary`。交互模式中输入 `exit`、`quit` 或 `q` 后，程序才会把最终状态保存到输出文件。

`--output` 和 `RAG_DATA_DIR` 的相对路径始终基于包含 `pyproject.toml` 的项目根目录解析，不受终端或 IDE 当前工作目录影响。默认计划输出到 `./plans/study_plan.md`，RAG 索引写入 `./.agents_data/rag`；传入绝对路径时仍使用指定位置。

默认情况下，即使使用 `-v` 也不会打印完整向量、资料正文或完整模型消息。仅在确需排查提示词内容时临时设置 `LOG_PAYLOADS=true`；日志只输出受 `LOG_PREVIEW_CHARS` 限制的预览，但仍可能包含资料片段，调试完成后应恢复为 `false`。
