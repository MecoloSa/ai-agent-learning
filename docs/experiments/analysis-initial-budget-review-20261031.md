# pipeline首次分析资料预算：候选验证记录

## 当前已完成

Python全量读取的公共路径校验、UTF-8/BOM严格解码及统一失败出口已直接应用，实际源码172项通过、1项符号链接环境跳过。本轮未新增在线模型请求。

## 下一阶段必要性

`analyse_materials`先组装概览、检索证据和完整Python源码，然后创建分析模型。`PLAN_MAX_EVIDENCE_CHARS`作用于后续规划，不能保护这个更早的请求。Python全量读取仍无总文本上限。拟增加独立`ANALYSIS_INITIAL_CONTEXT_MAX_CHARS`，默认120000字符，作为可调整工程起点，不视为模型token窗口或效果保证。

候选只用于pipeline组装后的资料上下文：小于或等于上限时保留全文；超出时抛出已有AnalysisExecutionError，reason为analysis_context_budget_exceeded。既不静默截断，也不自动切换Agent；用户可以缩小目录、选择Agent按需读取或显式调高额度，保留全读基线。

不覆盖系统消息、用户目标、Agent累计消息/工具预算、文件加载内存；RAG索引、召回或重排在检查前可能已产生资源消耗。本候选仅在分析LLM调用前检查，不宣称拦截所有上游API。

## 验证可见性

新增`tests/test_analysis_initial_context_budget.py`10项，全为本地测试。以9/10/11字符资料和10字符预算分别验证正常、恰好等于和越界；另检查Python来源头与概览组合、RAG组装路径、显式放大预算保留全文、Agent循环不受该额度影响、legacy/LangGraph不创建模型或规划、不暗中换模式、环境变量及配置快照、零预算拒绝。

当前源码新增10项均失败，因为新配置和检查尚不存在。候选仅在进程内替换config、agents及配置快照，相关190项收集，189通过、1项环境跳过。首次候选有一项测试替身缺少expand_neighbors接口，补齐真实调用契约后通过，没有放宽断言。

用户随后授权直接应用。Settings默认字段、get_settings环境读取、analyse_materials边界检查、EVAL_CONFIG_KEYS允许列表及.env.example说明均已落地，实际业务源码189项通过、1项环境跳过，无在线模型请求。现有CLI异常处理和两种编排保持原接口。新配置快照新增字段，旧实验不会自动补填，比较历史条件时应区分版本。该机制是可配置的请求前边界，不是资料选择算法，也不是模型效果改进的验证。
