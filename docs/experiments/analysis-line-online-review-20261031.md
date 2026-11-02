# 分析按行工具落地、双通道示例与访问顺序探针

## 已应用代码验证

用户应用后实际源码148项通过、1项符号链接环境跳过。按行工具复用原路径校验、200行/8000正文字符限制、实际范围引用；新增收集参数不暴露给模型。已有交互、本地/MCP和检查点回归覆盖。

两个列表由`analyse_materials_agentic`每次创建，经`build_agent_tools`传给工具工厂。工厂闭包保留同一个列表对象；内部`.append`更新分析函数的列表，`return result`仍独立回传当前模型。证据收集成功取得的非空正文及来源范围；获取事实收集查询/预算限制、截断、失败等观察事实。空列表表示启用但尚无记录，None表示不使用该旁路收集。默认None使原交互/MCP调用保持兼容。

## 真实在线案例

提问：解释rules.py中target_rule返回什么，以及每个返回值对应哪个变量。依据文件原文说明，不执行代码。

输入文件`tests/fixtures/evaluation/analysis_line_online/materials/rules.py`，8201字符，函数位于301—304行，超过开头5000字符。300行开头填充只是访问边界测试，不是完整真实项目代表。函数原文：

```python
def target_rule():
    required_count = 2
    total_count = 3
    return required_count, total_count
```

入口`python -m tests.helpers.analysis_line_online --output-dir 新目录 --llm`，真实聊天模型与文件工具，不执行输入源码，不请求embedding或重排。默认无--llm只预检。实际顺序：列目录→开头读取5000→搜索target_rule→按行读取301—400。实际只返回301—304，truncated=false；模型正确解释元组(2,3)对应required_count和total_count，引用[source=rules.py, lines=301-304]。不能把请求结束400当成实际引用结束。

evidence_blocks最终包含两条读取JSON：一条5000字符开头摘录，一条完整函数的4行。acquisition_facts仍含read_learning_file的prefix截断记录，这是早期获取事实，不是当前函数仍未读全的判定。按行成功且未截断，因此不添加新的限制记录。分析结束时pack_material_context序列化两通道，规划入口unpack后分别展示，列表本身不是模型参数。

结果`.evaluation-results/analysis-line-online-20261031-v1/observation.json`包含user_input、request_messages、model_responses、tool_results、final_answer、evidence和acquisition_facts。单次观察17.751秒，5次模型响应，输入17390/输出805/总18195 token，不把该值视为所有源码任务的成本。

## 单次候选探针

业务文件未修改，仅在进程内替换一条系统说明：

- 原：目录由程序绑定，不传root_dir或绝对路径；先用list_learning_files了解资料范围。
- 候选：目录由程序绑定，不传root_dir或绝对路径；文件未知时用list_learning_files发现资料。

同文件、同提问、同模型qwen3.7-plus及温度0.2，首轮系统消息只有这一条替换，用户消息相同。实际仍列目录，但跳过开头读取，直接搜索后读取301—500，实际范围仍301—304。回答正确。4次模型响应、输入5709/输出657/总6366 token，14.549秒。输出`.evaluation-results/analysis-line-discovery-candidate-20261031-v1/observation.json`。

这不是稳定收益保证，也不是严格归因证明：每条件一次、模型存在随机性，不能断言替换必然导致跳过开头。候选的独立理由是移除已知路径时不必要的强制步骤，保留自主选择，不增加强制搜索。相关进程内候选72项回归通过；业务尚未应用。

下一步仅建议这一处规则替换。更大范围的上下文保留/裁剪需要代表性任务对照，不能因本例开头无关就自动删除其他任务的开头证据；本轮也未解决pipeline全量Python输入。
