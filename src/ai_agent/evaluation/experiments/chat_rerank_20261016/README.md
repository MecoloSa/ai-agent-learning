# 套餐聊天模型替代专用重排实验（2026-10-16）

实验已实际调用百炼接口，业务代码与 `.env` 未改。这里测试的是 **rerank（重排）**：在已召回的候选中选择更靠前的证据；不是替代 embedding 或完整 retrieve 链路。

后续追加了 `deepseek-v4.1-flash` 与 `qwen3.6-flash` 的同时段72次套餐请求，无新增embedding或付费rerank。最新比较与推荐见 [DEEPSEEK_RESULTS.md](DEEPSEEK_RESULTS.md)，调用入口为 `deepseek_followup.py`。原始文件不覆盖；可通过 `analyze_followup.py` 离线核验同池、两轮覆盖和配对耗时。

## 后续业务接入状态

主线与PDF/交互评测已接入共用的聊天重排适配器，默认 `RAG_RERANK_BACKEND=chat` / `RAG_RERANK_CHAT_MODEL=deepseek-v4.1-flash`，支持切回 `dashscope` 或关闭重排。此后业务 `.env` 已更新，早期实验期间“未修改业务配置”的描述仅对应当时实验状态。

`integration_smoke.py` / `integration_smoke.json` 记录正式入口验证：复用首题冻结候选，主线 `hybrid` 与 `evaluate_case()` 各1次实际Token Plan调用，无新增embedding或付费rerank。两次均成功；主线首位为gold第21页，评测Recall@8与RR均为1，无补尾。该烟测验证接线而非完整索引/Agent端到端质量。新适配器同时覆盖 `dense` 和 `hybrid`，差异与开关见主线README。全项目回归为470通过、1跳过（只存在依赖弃用警告）。

## 候选选择依据

- [qwen3.8-flash 官方说明](https://help.aliyun.com/zh/model-studio/qwen3-8-flash)：强调响应速度与理解能力，支持结构化输出，作为新一代效果候选。
- [qwen3.6-flash 官方说明](https://help.aliyun.com/zh/model-studio/qwen3-6-flash)：支持结构化输出，官方报告相较前代提升，以已有 Flash 型号作速度对照。
- [思考模式文档](https://help.aliyun.com/zh/model-studio/deep-thinking)：此次显式 `enable_thinking=false`，避免额外思考生成延迟。

这些公开信息支持将 Flash 系列列入候选，但没有提供这两个托管型号在本项目中文重排任务上的可比吞吐或准确率。代码、数学榜单也不能等同检索质量，因此没有编造公开速度排名。未选 auto（模型不固定，不利于复现）、Max/Pro（当前优先速度）、图像/音频模型（任务不匹配）。没有证据证明这些排除项在所有场景都更慢。

## 实验设计

1. 读取现有 `a1b2c4e4a74b` 教材索引，不更新或重建整个资料库。
2. 使用 `book_retrieval_cases_audited_v2.jsonl` 的15题和 `book_retrieval_holdout_v1.jsonl` 的3题。每题编码一次，向量前32块与 BM25 前32块通过现有 RRF 公式融合，固定取前16块。
3. 冻结完整片段、元数据与金标准到本地 `inputs.local.json`，文件被本目录 `.gitignore` 排除。模型仅看到问题、候选编号、文件名与正文，不看到正确页或金标准。保存输入 SHA256 以验证各分支使用同一候选。
4. 每个聊天模型每题两轮，串行请求并固定随机种子打乱模型/问题顺序。`temperature=0`、`max_tokens=160`、JSON对象输出、关闭思考，SDK不重试，超时45秒。
5. 专用 `qwen3.7-text-rerank` 每题一次，共18次按量付费调用，另有RRF原顺序对照。后续诊断与修复实验没有再调用付费重排。用户已明确授权教材片段外发与18次付费基线。
6. 第一组严格要求16个编号完整排列；异常整体回退RRF。两次诊断各复查一个失败例，发现缺少单个索引；原始失败请求只保留异常类型，无法断言所有失败均属于同一原因。
7. 第二组保持提示词和模型参数不变，允许长度至少5、整数唯一且不越界的列表；漏项按RRF原顺序补尾。每个模型再测两轮，单独保存，禁止删除第一组失败记录。

这是同池排序评测，不包含回答生成、最终答案正确率、工具多轮往返、邻居扩展、运行时分数阈值和无相关证据拒答。页级指标先去重再取K。`acceptable_hit1` 接受已审计的充分替代页，严格 `hit1` 只接受原始 gold 页，避免把“任一充分证据”误解成“必须召回全部替代页”。MRR@5表示首个正确页排名倒数的平均值，越接近1越好。

## 文件与复现

- `run.py`：冻结候选、调用模型、校验排序、失败回退、汇总。
- `traces.jsonl` / `summary.json`：严格校验组的请求轨迹和汇总。
- `diagnose.py` / `diagnosis.json`：两次失败复查，仅记录候选编号。
- `traces_partial.jsonl` / `summary_partial.json`：漏项补尾组。
- `test_experiment.py`：正常排列、布尔/浮点/重复/越界/缺项拒绝、补尾、页去重和失败分母的离线测试。
- `RESULTS.md`：实测结果与使用建议。

项目根目录 PowerShell：

```powershell
.venv/Scripts/python.exe -X utf8 src/ai_agent/evaluation/experiments/chat_rerank_20261016/run.py --prepare
.venv/Scripts/python.exe -X utf8 src/ai_agent/evaluation/experiments/chat_rerank_20261016/run.py --paid-baseline
.venv/Scripts/python.exe -X utf8 src/ai_agent/evaluation/experiments/chat_rerank_20261016/run.py --allow-partial
.venv/Scripts/python.exe -X utf8 -m pytest src/ai_agent/evaluation/experiments/chat_rerank_20261016/test_experiment.py
```

`--prepare` 会编码18条问题，产生少量 embedding 按量费用；不需要复现候选时不要重复运行。`--paid-baseline` 显式启用付费重排，默认不开。已有轨迹按题目/模型/轮次跳过；需要全新实验请使用新目录，不能在旧轨迹存在时重建输入后继续混用。原始套餐Credits消耗、缓存折扣和人民币账单未由API返回，token数不是账单。

## 接入前必须注意

聊天排序没有专用重排的 `relevance_score`；不能直接复用业务里的0.5过滤阈值，也不能伪造一个分数。建议先保留实验开关，分别处理排序与相关性过滤；需要增加无答案/无关候选测试后再讨论替换运行时过滤。

本数据只有18个独立问题，包含多条相近的“第四部分”问题，留出集仅3题；两轮调用不是36个独立问题。补尾策略是在观察失败后制定，留出结果也已被查看，因此属于探索性证据，需要新的未见问题确认。这里只测同一时段与当前端点，API负载、缓存和网络都会影响延迟；RRF的0毫秒只表示冻结候选无需额外模型排序，不表示整个检索零耗时。
