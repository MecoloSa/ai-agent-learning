# 历史检索策略对照（2026-10-29）

## 范围与方法

本轮不修改业务代码。确认用户已新增retrieval/semantic_history.py，正式源码13项组件测试通过；对照脚本与组件、原RRF合计24项实际通过。

沿用history_retrieval_coverage.json和history_semantic_followup.json，共16条固定查询、5条可读合成用户消息。范围外原话和助手消息不编码。完整问题已在聊天展示，没有新增运行提示词示例或改动原有标签。coverage-005“不想动笔怎么学”的目标不够唯一，保留原始排名，但排除出下述汇总；人工改写对照仍单独标注，不视为Agent生成查询的成绩。

在线复用create_embeddings，当前qwen3.7-text-embedding-flash生成5条文档向量及16条查询向量；冻结在.evaluation-results/history-strategy-artifacts/4abfccbe933a480ba17946c098e7ef91/frozen.json。四种策略及阈值观察使用同一批向量，不重新请求模型：BM25、纯向量、现有fuse_rrf_candidates、BM25非空则使用否则查向量。RRF适配只新增实验位置元数据，不修改算法、不将history-source当文件路径。

结果history-strategy-comparison-v1.json，后续纯本地重放history-strategy-local-replay-v1.json，均位于.evaluation-results。重放本轮Embedding客户端调用为0。冻结向量记录中的编码耗时属于首次采样，不是重放耗时；策略接口调用数也不等于HTTP重试后的实际请求数。

## 原始不筛选结果

以下正例10题包括词面3题、人工改写2题、换说法5题；负例5题。小样本检索结果不等于真实Agent回答准确率或总体分布。

| 策略 | 正例目标第1名/10 | 目标前3/10 | 负例返回空/5 | 模拟16题需要查询编码的题数 |
| --- | --- | --- | --- | --- |
| BM25 | 5 | 5 | 5 | 0 |
| 纯向量 | 10 | 10 | 0 | 16 |
| RRF混合 | 9 | 10 | 0 | 16 |
| BM25空结果才查向量 | 9 | 9 | 0 | 10 |

通勤换说法题的BM25因“可以”命中“明天的统计练习最多60分钟，可以写代码”。非空兜底直接接受错误候选；原RRF把该候选的词面第1与向量第2累加，超过只有向量第1的通勤原话。这说明有结果不等于相关，两路融合也不保证优于单路。

阈值灵敏度仅用于观察：dense_min=0.4时，纯向量与RRF均10/10正例第1、负例5/5为空；0.5时纯向量丢失两道正例。不得据同一小集合调出0.4后称生产阈值已验证，也不能拿余弦阈值筛RRF分数。

## 时间与聊天重排

首次5条消息编码213毫秒，16次查询编码中位数115毫秒、最大126毫秒。只有极小合成历史与本轮网络条件，未测长历史、首次建库端到端耗时或真实费用。

在冻结向量上，仅两题调用现有chat重排，DeepSeek配置保持不变；结果history-strategy-rerank-comparison-v1.json：

- 通勤题：把RRF第2的正确原话提到第1，485毫秒，返回总token数270。
- 无关“火山温度曲线”：依然排列全部5条候选，1106毫秒，262 token。当前重排契约要求全排列，不支持全部无关返回空，不能拿排名或存在候选当成有证据。

## 建议与边界

当前Embedding是在线模型，语义组件本地缓存与计算，不是本地模型。测试人工向量不能代替语义模型；聊天重排也不生成向量。复用现有Embedding、RRF和必要时的重排客户端合理，但历史范围、位置、缓存、引用和回读必须单独适配，不能整个复用LearningRAG资料目录流水线。

建议保留BM25基线；自然语言历史查询试验可配置语义/混合路径，而非仅空结果时兜底。先过滤与范围保护，再取候选，按需重排，回读核实后回答；未找到可支持原话时说明本次未找到，不能宣称历史不存在。聊天重排默认不必每题调用。

接线前仍需独立校准负例/边界正例、模型版本绑定阈值与缓存、跨轮文档向量复用、Embedding失败时显式降级、不绕过预算与范围。未直接修改业务检索策略或设定生产阈值。

## 本地复现

在项目根目录执行以下命令，不调用在线模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_strategy_comparison.py --replay .evaluation-results/history-strategy-artifacts/4abfccbe933a480ba17946c098e7ef91/frozen.json --output .evaluation-results/history-strategy-local-replay-v2.json
```

加入--rerank会调用在线聊天重排，不是纯本地复现；--embed会重新生成向量，不保证与冻结采样逐位一致。
