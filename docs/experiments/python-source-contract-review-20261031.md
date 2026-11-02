# Python全量读取契约检查

## 上一轮已落地验证

条件式目录发现规则已核实应用，实际源码72项回归通过。真实单例依次列目录、搜索target_rule、按行读取301—450；实际返回301—304，正确解释(2,3)与变量对应。4次模型响应，输入5709/输出732/总6441 token，15.408秒。结果`.evaluation-results/analysis-discovery-applied-20261031-v1/observation.json`。模型仍列目录，不能宣称最短访问路径已保证，也不把单例当作稳定节省。

## 新发现及离线复现

`runtime/agents.py::_collect_python_sources`直接`path.read_text(encoding="utf-8", errors="replace")`，没有复用`resources/materials.py::safe_material_path`。

新增`tests/test_python_source_read_contract.py`八项：普通UTF-8、BOM、非法字节、公共路径校验接线、读取权限错误、legacy/langgraph失败停止、合法UTF-8中的U+FFFD字符。坏字节文件为`def rule():\n    return '\xff'\n`的字节表示，模型工厂被替换为立即报错的哨兵；旧实现确实走到哨兵，未请求在线API。路径测试注入公共校验拒绝，验证必须先经过它，不声称在当前Windows环境真实创建了越界符号链接。

首次检查源码2项通过、6项失败。候选先校验解析后的路径，再`read_text(encoding="utf-8-sig")`严格解码；读取/解码/路径错误转换为已有`AnalysisExecutionError(reason="source_read_failed")`。用户随后授权直接应用，实际业务源码173项收集，172通过、1项环境跳过；两种编排均在模型及规划之前停止。该读取修正已落地。

这是资料准确性与统一边界修正，不是Python语法检查，不执行源码，不新增调用模型、预算或提示词。合法UTF-8内容保持不变，BOM按编码标记处理；不自动猜测GBK等其他编码。遇到不可准确读取的文件，pipeline整体明确失败，不默默跳过；需用户修正编码/权限或按需选择其他资料路径。Python全量上下文未设整体预算的问题仍未解决。

主流程联动已检查：`analyse_materials`在创建模型前收集Python；legacy及LangGraph调用该分析函数；CLI已处理AnalysisExecutionError并停止计划生成和文件导出。因此无需新增CLI、配置、存档字段或调用方参数。手动修改方案直接在对话提供。
