# 主流程性能诊断（2026-10-10）

## 结论与实验边界

当前环境首先需要解决 API 连接时的 IPv6 超时回退，其次是分析/规划的长上下文和长输出。已有索引时，本地 Chroma 查询、文件哈希和邻居扩展不是主要耗时来源。不能仅凭看到多次 `sync()` 就把性能问题归因于索引。

使用用户给出的两本 PDF、相同学习目标、agent 模式、现有 `.env` 和已有索引。真实流程对照均生成 **7 天计划**；用户原命令未指定 `--days`，实际默认是 **3 天**。因此这些总耗时不是原命令的严格复现，生成长度也可能不同；资料分析链路相同，无认证传输对照不受天数影响。没有运行交互问答，也没有计入 `uv` 启动耗时。

业务代码、模型配置和系统网络设置未修改。IPv4 对照只在独立诊断 Python 进程内过滤 DNS 结果。实验只保存耗时、计数和 token 元数据，没有保存资料正文、模型回复或私有思维链。单次对照受服务负载、模型随机性和服务端缓存影响，不作为稳定加速比例或质量等价证明。

## 1. 已确认：连接时先等待两个 IPv6 地址超时

本机 DNS 对聊天域名 `token-plan.cn-beijing.maas.aliyuncs.com` 和 embedding 域名 `dashscope.aliyuncs.com` 均先返回两个 IPv6 地址，再返回 IPv4 地址。

使用与项目相同的 HTTPX / Requests 客户端类型，对两个已配置 API 地址执行无认证 GET，同时测量 DNS、TCP、TLS。401/404 是这类无认证探测的预期响应；这里仅检查连接时间，不代表模型调用成功。

| 客户端与地址 | 默认地址选择：首次 GET | 同连接第二次 GET | 仅 IPv4：首次 GET |
|---|---:|---:|---:|
| HTTPX / 聊天 API | 42.208 s | 0.015 s | 0.076 s |
| Requests / embedding API | 42.152 s | 0.012 s | 0.068 s |

默认探测中，两个 TCP connect 各超时约 21.05 秒，随后一次 TCP 连接约 0.015–0.031 秒成功；DNS 约 0.0004–0.0034 秒，TLS 约 0.018–0.024 秒。结合 DNS 地址顺序和仅 IPv4 对照，可以定位为当前网络的 IPv6 连接失败回退，而不是 DNS 慢或模型处理了 42 秒。

真实业务也出现相同模式：

- 默认首次模型调用 44.480 秒；IPv4 对照为 2.094 秒。
- 默认首次 query embedding 42.274 秒；IPv4 对照为 0.158 秒。
- 默认随后九条 embedding 各约 0.081–0.118 秒。

相关接入点：`src/ai_agent/config.py:create_llm()`（聊天客户端）；`src/ai_agent/embeddings.py:BatchedDashScopeEmbeddings`（embedding 客户端）。两条传输链路都需要处理，单改 ChatOpenAI 无法消除 embedding 的首次等待。

建议优先修复当前机器/网络的 IPv6 可达性，或提供可回退、默认关闭的应用网络策略。另应把连接超时与响应读取超时分开配置，避免连接失败长期占用生成预算。不要硬编码 API IP；实验中的进程级 DNS monkeypatch 也不应直接搬入生产代码。观察新进程首次调用和空闲后重连的 p50/p95、失败率，确认两个 API 的约 42 秒阶梯同时消失。

## 2. 已确认：长上下文、长输出仍有显著开销

默认网络真实流程总耗时 **357.418 秒**，5 次模型调用合计 **311.596 秒**（87.2%，含这些调用内部的网络等待）；检索工具合计约 **44.02 秒**，其中首次 embedding 占 **42.274 秒**。这些计时存在嵌套：`retrieve_many` 包含 `retrieve`，后者又包含 `sync` 和 embedding，不能将各层相加。

| 默认网络阶段 | 输入字符数 | 耗时 | 观察 |
|---|---:|---:|---|
| 分析第 1 轮 | 1,104 | 44.480 s | 只请求列目录 |
| 分析第 2 轮 | 1,200 | 2.642 s | 提交 4 条检索 query |
| 第一组检索 | — | 42.891 s | 首次 embedding 连接等待占主导 |
| 分析第 3 轮 | 49,543 | 46.214 s | 再提交 6 条 query；与约 42 秒重连模式一致，但此业务请求没有独立 TCP 分解 |
| 第二组检索 | — | 1.011 s | 6 条 query 串行仍仅约 1 秒 |
| 分析第 4 轮 | 97,547 | 91.920 s | 输出 9,962 字符分析 |
| 规划第 1 轮 | 108,198 | 126.340 s | 直接生成计划，没有补充检索 |

IPv4 对照中，分析第 4 轮仍用 **68.753 秒**：输入 81,034 字符，输出 6,538 字符。因此解决网络后，不能期望整个流程变成几秒。

IPv4 对照完整流程耗时 **153.811 秒**，其中最终规划调用 **73.928 秒**，输入 88,252 字符，输出 7,380 字符。它也只有 5 次模型调用，但检索 query 数量、证据规模、输出长度和 reasoning token 数与默认运行不同；规划 reasoning 为 487 tokens，而默认为 4,493。因此 357.418 → 153.811 秒只能作为两次观测，不能将这 203.607 秒差值全部归因于网络，也不能当成固定 57% 提速承诺。

具体机制与修改顺序：

1. `tools.py:build_retrieve_learning_context_tool()` 默认单次返回预算为 `rag_max_context_chars // 2`，当前是 **48,000 字符**；`agents.py:analyse_materials_agentic()` 每轮把完整 ToolMessage 累加进 `messages`。不同 query 命中相同块时，`retrieve_many()` 仅在当次工具调用内去重，跨次调用仍可能重传相同正文。优先考虑按 source/chunk 去重、保留来源与关键事实，限制整个分析阶段的累计证据预算，而不只限制单次返回。应同时记录新增唯一块数、累计输入 token、证据覆盖率和引用忠实度，防止提速以丢失关键证据为代价。
2. `agents.py:create_study_plan()` 再次发送分析结果和原始工具证据；当前 `PLAN_MAX_EVIDENCE_CHARS=120000`，本次 96,885 字符工具轨迹基本完整传入规划。应选择能支持具体任务的证据，不把所有历史观察无差别转发；先做预算对照实验，不直接把阈值砍到无法支持引用。
3. 规划调用的 usage 记录为输入 **57,635 tokens**、输出 **7,285 tokens**，其中 reasoning **4,493 tokens**。推理 token 计数并非私有思维链内容，说明可见正文长度不是全部生成量。应单独比较推理预算、模型选择和输出详略，不仅比较输入字符数；具体配置需按当前服务支持能力验证，不能只凭参数名猜测。此处没有首 token/生成阶段分解，不能精确断言这些 token 各占多少秒。
4. `config.py:create_llm()` 使用 `streaming=False`，长回复完成前不会显示正文。流式输出可改善等待体验，但不能替代上述总耗时优化。

## 3. 可以优化，但当前优先级低

两本 PDF 合计 **114,152,852 bytes**，已有 **2,063 chunks**。三次本地同步均为 changed=0、deleted=0、added_chunks=0，分别用 0.132、0.141、0.078 秒；邻居扩展为 0.032/0.027 秒。真实默认流程十次同步总计 **0.621 秒**，两次邻居扩展合计 **0.120 秒**。

- `rag.py:retrieve_many()` 串行调用 `retrieve()`，后者每条 query 都调用 `sync()`。可改为每批 query 同步一次，并保留单独 `retrieve()` 的同步语义。此处收益主要是约几百毫秒，不是几十秒。
- 查询 embedding 可以进一步批量或有限并发，但要保持 query 语义；当前适配器的 `embed_documents()` 使用 `text_type=document`，`embed_query()` 使用 `text_type=query`，不能为了批量而直接用文档编码替代查询编码。
- `rag.py:expand_neighbors()` 每次加载命中文件的全部块；规模扩大后可按邻居 ID 查询，但本次约几十毫秒，不应先重写。
- `rag.py:build_overview_text()` 先 `_load_file()` 提取完整 PDF，再截前若干页，确实浪费 PDF 解析；但用户的 agent 分析路径不调用这个方法，它主要影响 pipeline，不能据此解释当前命令慢。

## 4. full 记忆与交互行为

`main.py` 在 `run_learning_assistant()` 完成后才创建 `ConversationMemory`。因此切换 full/window/summary 不会加速初始分析和计划生成。

进入交互后，`_build_interactive_system_prompt()` 把 `material_context` 全部放入 system prompt，随后每轮都携带。window/summary 只裁剪或压缩后续聊天历史，不能缩减这段固定 system 内容。若交互也慢，应单独限制固定证据预算并按需检索。

另外，指定 `--interactive` 后，程序要等输入 `exit/quit/q` 退出会话才写入 `--output`；一直未见输出文件并不一定代表初始计划还在生成。

## 复现材料

诊断脚本在本地 `tmp/performance_probe.py` 与 `tmp/transport_probe.py`；该目录已被项目忽略，不会误提交运行产物。原始统计：

- `tmp/performance-offline.json`
- `tmp/performance-agent-online.json`（首次版本固定 7 天）
- `tmp/performance-agent-ipv4.json`（显式 `--days 7`）
- `tmp/performance-transport.json`
- `tmp/performance-transport-ipv4.json`

后续可用以下命令做默认 3 天的独立复现；两条主流程命令均会真实调用已配置模型并产生对应使用量，但不会保存计划正文：

```powershell
.venv/Scripts/python.exe -X utf8 tmp/performance_probe.py --run --days 3 --output tmp/profile-default.json
.venv/Scripts/python.exe -X utf8 tmp/performance_probe.py --run --ipv4 --days 3 --output tmp/profile-ipv4.json
```

下一轮应交替运行多个案例、重复采样，保存阶段延迟、token、引用忠实度和任务覆盖；本次没有评测生成质量，不能声称两个计划质量一致。

## 后续修复验证（同日）

用户授权小范围修复后，新增 `network.py`，通过 HTTPX `local_address` 与 Requests adapter `source_address` 绑定 IPv4 源地址。聊天通过 `http_client` 注入，Embedding 通过 DashScope 支持的 `session` 注入；不采用诊断脚本中的全局 DNS monkeypatch。开关 `API_FORCE_IPV4` 默认 false，本地 `.env` 已开启；关闭并重启可回退。代理与 NO_PROXY 的选择已覆盖测试，同步连接池复用并在进程退出时关闭。

直接使用 `create_llm()` / `create_embeddings()`，没有修改 DNS 或 mock 网络，执行极短聊天、查询编码和单条文档编码：首次聊天 **1.067 秒**，首次查询编码 **0.173 秒**，文档编码 **0.076 秒**，两个编码都返回 1024 维向量。原始结果在 `tmp/ipv4-fix-validation.json`。这验证了真实 SDK 接入，未重新运行完整计划生成，不能据此承诺总流程秒数。

验证命令覆盖网络策略、模型选择、运行路径、RAG、工具和 Agent 循环，共 **57 项通过**。`httpx` 与 `requests` 已显式列入依赖，离线同步 `uv.lock` 没有升级包。该开关当前覆盖同步调用；未来使用 `ainvoke` 时需另行配置异步传输。
