# Embedding调用链与小样本用量核对（2026-10-29）

## 实际调用链

- config.create_embeddings创建BatchedDashScopeEmbeddings，当前模型qwen3.7-text-embedding-flash；SDK主机dashscope.aliyuncs.com，仅记录主机，不公开密钥。
- LearningRAG构造Chroma时指定embedding_function=create_embeddings；sync对新增/变更的非Python资料调用add_documents编码；未变资料不重复编码。查询的similarity_search_with_relevance_scores会编码问题。不是每次构造客户端就发出编码请求。
- 当前配置rag_retrieval_strategy=dense，rag_rerank_enabled=True，rag_rerank_backend=chat，rag_rerank_chat_model=deepseek-v4.1-flash。聊天模型替代专用重排模型，不替代Embedding。
- 历史回查search_conversation_turns当前仍是BM25；SemanticHistoryIndex已落地但尚未接到历史工具。

## 服务端用量实测

tests/helpers/history_embedding_usage_probe.py默认预检；--embed复用原客户端，只记录供应商status_code、用量白名单和耗时。合成输入沿用history_retrieval_coverage.json可读的5条用户消息，另查询“路上用耳朵学东西”。不记录密钥、原始响应或正式业务资料。

结果.evaluation-results/history-embedding-usage-v1.json：

| 请求 | 服务端total_tokens | 实测耗时 |
| --- | --- | --- |
| 5条合成用户消息编码 | 143 | 244毫秒 |
| 1条查询编码 | 21 | 117毫秒 |

官网北京地域qwen3.7-text-embedding-flash实时输入单价0.125元/百万Token，输出不计费。2026-10-29核对：https://help.aliyun.com/zh/model-studio/model-pricing 。不将适配器一次提交多条文本当作官网异步Batch半价服务。

按标价计算：143×0.125/1000000=0.000017875元；21×0.125/1000000=0.000002625元；合计0.0000205元。若每次查询均21Token，1万次查询约0.02625元，另计新增历史编码。仅是用量和标价推算，不是账户账单；免费额度、实际地域/渠道和折扣未审计。长历史、重试及重复编码会增加消耗。

## 决策

当前学习项目的小规模历史检索可继续采用已有在线Embedding，无需因本次费用担心而另换本地模型或测试替身。保留BM25基线；语义路径后续按固定流程接线，不直接改业务。优先保证缓存按模型和原文指纹绑定、范围过滤先于候选构建、避免每轮全量编码、记录失败降级和调用用量。当前独立组件仅缓存实例内向量，跨轮增量缓存尚未实现，不能写成已有能力。

新增计量辅助测试2项与正式语义组件13项，合计15项实际通过。测试验证透传响应、计量字段白名单以及预检不创建客户端；不宣称历史检索集成已完成。
