# 多来源历史定位验收（2026-10-29）

本轮未修改业务代码。确认工具注册已落地；扩展tests/helpers/history_readback_online.py支持多个独立归档、逐来源上限及目标来源观察，保留旧单来源案例；新增两项离线辅助测试。实际关联96项通过。

案例完整输入与初始合成问答在tests/fixtures/evaluation/history_multisource_online_cases.json；两例独立初始化，近期消息为空，摘要不含答案。来源1仅1轮（20分钟），来源2数据库有2轮（第1轮更正35分钟、第2轮暗号松柏），快照对两个来源均仅允许第1轮。自动输入提问、exit，意外确认输入n并记录。

## 实际结果

- 008，run_id=63467be3ebe74118b67f69b7e9dca3be：先list，再省略source_index搜索“学习时间 更正 今天”，命中来源2第1轮为rank1、来源1第1轮为rank2。读取来源2第1轮后，额外请求读取来源1第1轮因3次预算耗尽被拒。最终完整引用35分钟原话与工具返回citation。35.263秒。
- 009，run_id=ec31cadcd90e497dadf558386603fe70：省略source_index搜索“暗号松柏”，searched_sources包含两个来源且各上限1，matches为空。未读取范围外第2轮，回答限定为本次范围未找到。6.915秒。
- 两例首次请求均未包含目标完整原话；正式状态、原快照及所有父归档字节、继承来源上限均保持；意外确认0次。
- 009 successful_read_count、scope_denial_count、returned_citation_in_answer均为0/False符合预期：合法范围内搜索返回空，不是越界读取被拒。
- assessment保留not_evaluated；此处为人工核对原始审计后的质量结论，不改原始结果标签。

原始结果：.evaluation-results/history-multisource-online-v1.jsonl。各行observation_path保存实际输入、首次消息、工具调用与结果。

## 结论与边界

两例支持跨来源定位和逐来源范围保护已接通，不能证明多次随机运行稳定性或语义召回覆盖。008发现冗余目录查询占用预算，应优先明确原话查询默认直接搜索的提示词，再用同条件复测；不据此直接提高全局预算，也不强制只读排名第一项。

## 复现

### v3与适用日期检查

- 用户008，bf23483b5a7e40be8a9c2e10b1e933f6：search→read来源2第1轮→read来源1第1轮。最终同时引用35分钟更正与20分钟原安排，无list、无预算拒绝，状态保护通过。两次read核对不同候选，不是重复执行相同参数。
- 助手009，79cc31a125544971a58b3338f94057b8：一次全范围搜索返回空，未泄露范围外原话，状态保护通过。
- 新010，7db9eb06a9c44fb79b199e74f3bac666，完整输入见tests/fixtures/evaluation/history_dated_arrangements_online_case.json：来源1原文明确10月29日最多50分钟，来源2明确10月28日更正35分钟；问10月29日安排。结果正确引用50分钟。实际list→search→read来源1；额外读取来源2被预算拒绝。因此只支持答案依据正确，不证明模型比较了两条原文或稳定省去目录查询。
- 009追加至history-multisource-online-v3.jsonl；010保存于history-dated-arrangements-online-v1.jsonl；均位于.evaluation-results。旧原始结果不修改。
- 本轮确认说明落地并执行关联96项通过；随后仅改测试辅助代码增加实际工具序列、成功读取位置、预算拒绝计数，辅助测试12项通过。后续结果包含这些观察字段，旧结果可从observation_path审计读取。
- 当前不继续堆叠提示词或放宽预算。功能链路已验证，效率仍有波动；下一阶段应积累代表性检索问题与任务成功/调用成本记录，再决定是否需要固定编排或语义历史检索。

### v2复测

用户运行008，run_id=0a8805854e5947c4aac012b74b7cb422，19.420秒。首次请求已包含main.py的新查询顺序规则，但实际仍为list→search（省略source_index）→read来源2第1轮；未尝试额外回读。原话、引用、正式状态、父文件、继承范围均通过，首次请求不含目标原话。调用效率目标未达成。源码复查发现list工具描述仍要求“来源未知时先列出”，与系统规则冲突；下一步候选仅统一list/read工具描述，不改参数和执行边界。相关96项测试复核通过。

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_readback_online.py --cases tests/fixtures/evaluation/history_multisource_online_cases.json --case-id history-multisource-online-008 --output .evaluation-results/history-multisource-online-v2.jsonl --llm
```

009替换case-id末尾编号；去掉--llm只预检，不产生在线结果。在线命令退出0只表示运行和状态保护检查通过，仍需人工检查工具路径、原话与引用。
