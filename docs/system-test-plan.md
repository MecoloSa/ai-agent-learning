# 学习助手系统验收方案

## 测试层次

| 层次 | 场景 | 通过条件 |
| --- | --- | --- |
| 离线回归 | Web 与 CLI 入口、工具契约、资料读取、RAG、计划审批、任务反馈、历史检索与隔离、委派失败 | 不出现新增回归。记录跳过项。 |
| 浏览器流程 | 关联本地资料、切换模式、提交聊天、实时预览、计划采用和拒绝、任务编辑、搜索、导出、切换对话、窄屏 | 界面状态与请求一致。关闭或刷新界面后，从持久化记录恢复。 |
| 状态和隔离 | 请求重放、候选过期、其他对话并发保存、越界历史回查 | 请求重放不覆盖新状态。旧候选只能按允许的规则处理。并发提交不跨对话污染。越界时不展示原文。 |
| 真模型 Web | 一般学习建议、限定两页逐字引用、跨书组会讲解、指定缺页、索要整页 | 核对本轮实际读取与最终答复。未经确认不改业务状态。显式委派要单记成功、部分交付及主模型补救。 |
| 资料检索 | 已审校 15 题和留出 3 题 | 逐题报告目标页覆盖，以及重排前后排名、耗时与失败。答案另测引用和依据。 |
| 性能与成本 | 本地工作区接口、HTTP 首个公开文本、最终提交、子图耗时、模型使用量 | 本地延迟与模型延迟分别报告。注明实际执行路径、样本数、网络错误及未知用量。 |

## 在线案例和判定

教材在线案例使用同一份资料目录。每轮新建一个临时 SQLite 工作区，预置学习目标和计划，复用同一资料索引。限定页、缺页和跨书题在快速、深入、深入关闭思考三种模式下各重复两次。一般建议和整页摘录在快速模式各重复两次。

- 一般建议应给出三项可执行办法，且不调用本地资料工具。
- 限定页题只允许读取各书指定的物理页。逐字引文与 PDF 提取文本核对，分别记录严格匹配和忽略空白匹配。
- 跨书题检查两书机制、比较、练习及判分。另核对子图报告是否实际返回，以及主模型是否需要补读。
- 缺页题要求指出缺页、解释可读页，实际读取范围不能擅自替换。
- 整页题预期给少量短摘录和实际位置。
- 普通聊天全程比较正式状态。对话记忆允许保存，计划、任务与反馈记录需保持一致。

## 性能记录

Web 验收走本地 `POST /api/action` 流式入口，逐条记录准备事件、首个非空文本事件和最终保存回执的时间。公开的工具阶段说明可以先于答案出现。比较阶段以实际执行工具和子报告为准。主模型 token 从审计取，子图用量需独立抓取。两个重复样本只用于发现稳定性问题和建立本轮范围。正式尾延迟基线需要更多重复、随机交叉顺序，并分别记录冷索引、热索引和并发写入。

## 复现

从项目根目录执行，重复运行时使用新的结果目录。在线调用使用当前项目配置。

```powershell
./.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --durations=10
./.venv/Scripts/python.exe -X utf8 -B -m tests.helpers.workspace_system_acceptance src/ai_agent/eval/ai_agent_book .evaluation-results/system-web-recheck --cases quote missing cross_book --repeats 2
./.venv/Scripts/python.exe -X utf8 -B -m tests.helpers.workspace_system_acceptance src/ai_agent/eval/ai_agent_book .evaluation-results/system-chat-recheck --cases advice long_excerpt --modes fast --repeats 2
./.venv/Scripts/python.exe -X utf8 -B -m tests.helpers.workspace_smoke --output-dir .evaluation-results/system-plan-recheck --llm
./.venv/Scripts/python.exe -X utf8 -B -m ai_agent.evaluation.pdf_rag_eval --materials src/ai_agent/eval/ai_agent_book --cases tests/fixtures/evaluation/book_retrieval_holdout_v1.jsonl --output .evaluation-results/system-rag-holdout-recheck.json --k 8 --rerank
```

逐轮 Web 结果包含用户问题、工具审计、最终回答和状态保护结果，保存在本地。检索评测输出每行是一项案例，应按 JSONL 读取。
