# 第5项正式落地：规划证据展示副本

2026-11-02用户明确授权。runtime/agents.py新增stage_evidence，在_limit_evidence_context完全去重后、预算选择前整理已识别的成功读取JSON。保留正文、缩进、原始citation、实际范围、truncated和partial，去掉规划不需要重复看到的外层包装及请求范围字段。未知字段/结构、失败回执、不连续行号原样保留。

原工具返回、分析中的ToolMessage、evidence_blocks原始收集结果、material_context及获取事实均不被改写。只在规划输入生成新字符串，预算统计实际展示文本。不是LLM摘要，不新增模型调用，不合并范围，不声称原文支持所有教学建议。

更新test_acquisition_handoff和test_analysis_evidence_channel：原始JSON仍能解析且正文不变，规划副本正文/来源/完整性明确，initial_context_chars保留原输入量，失败事实继续独立。没有删除来源、授权或状态保护断言。

实际源码全量1615通过、1项符号链接环境跳过、3项依赖弃用警告，135.83秒；日志.evaluation-results/stage-evidence-adopted-20261102/pytest.log。未新增在线请求。此前短案例503→169字符、680→575输入token仍是候选对照观察，不升级为任意任务效果保证。

v0.8.1标签未移动，当前修改尚未创建新提交。后续提示词精简、请求视图、摘要窗口均仅为审计方案，没有顺带落地。
