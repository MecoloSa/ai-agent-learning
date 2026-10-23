# 项目演示与验收

本项目是有资料依据、可跟进进度的本地学习助手，不是通用自主学习系统。本文记录当前使用路径，不改变业务逻辑。展示中的状态、反馈和时间均为演示数据，不是永久系统规则。

## 1. 准备与验证范围

从项目根目录运行。模型阶段需要可用服务配置，并可能产生费用；MCP与本地工具对照不调用模型。不要公开.env、完整存档、资料正文或原始审计轨迹。

```powershell
Set-Location "F:\LLM\projects\agent"
uv sync
uv run python -m ai_agent.main --help
```

默认使用本地工具，MCP仅切换交互中的TXT/MD/PY按行读取，不切换初始分析、PDF和检索。--mode选择内部pipeline/agent；--orchestrator选择外部编排；两者独立。

## 2. 首次准备：不计入3—5分钟展示时长

若已有同版本资料的演示存档和真实结果，无需重新请求模型。需要从头复现时：

```powershell
uv run python -m ai_agent.main `
  --materials "F:/LLM/projects/agent/tests/fixtures/evaluation/feedback_adaptation/materials" `
  --goal "理解工具调用闭环，区分两类停止及其决定来源" `
  --days 1 --mode pipeline --orchestrator langgraph `
  --interactive --memory-strategy window --read-transport local `
  --output "F:/LLM/projects/agent/.evaluation-results/demo-study-plan.md"
```

自然输入依次为：

1. “请把阅读并比较两类停止定义加入待办，先不改计划。”核对后y。
2. “这个任务已经读完，但仍不理解为什么正常结束不代表答案正确。请记录，先不改计划。”核对后y。
3. “今天后续只有20分钟，只阅读和口头复述。请记录限制，先不改计划。”核对后y。
4. “依据这些反馈讨论后续安排，只讨论，不修改记录或计划。”检查没有写入确认。
5. “现在调整必要的后续安排，保留其他内容和任务记录，给完整候选供确认。”v查看，再n否决。
6. 再次提出候选请求。先人工检查困难、方式与负荷；合理才y，不为完成演示而认可不合适计划。
7. /state核对，/save-state保留快照，复制实际路径，然后exit。

任务对象不清时允许澄清，不强求固定工具轨迹。未形成真实pending不能只凭口头承诺算成功。反馈记录不授权改计划，读取完成不表示理解或掌握。

## 3. 3—5分钟展示：讲设计选择，不等待整轮生成

- 前30秒：说明用户场景及两条边界——资料提供依据，用户确认决定写入。
- 第1—2分钟：展示已准备的反馈和计划候选；对比只讨论/n否决前后状态不变、y采用后修订号增长且任务/反馈保留。
- 第2—3分钟：使用真实快照恢复一次，展示不重新生成初始计划、沿用原记忆策略；查询计划、任务状态与安排反馈，三者分开回答。模型延迟较大时用已有真实录屏或日志，明确它不是本次实时运行。
- 第3—4分钟：展示一份MCP对照结果，解释相同业务JSON、额外协议开销、默认保留local。
- 最后约30秒：说明计划内容仍需核对、摘要可能丢失历史、旧引用需重读、MCP未开放写状态等局限。

恢复命令：

```powershell
uv run python -m ai_agent.state_catalog --limit 5
uv run python -m ai_agent.main --resume-state "实际快照完整路径"
```

需要演示MCP读取时显式附加--read-transport mcp；传输方式不保存在学习快照中。读取问题可用：“请实际调用read_material_lines读取reading_notes.md第6到10行，简要复述，不修改状态。”

## 4. MCP对照：一次小样本即可

```powershell
uv run python -m ai_agent.evaluation.mcp_compare `
  --material-dir "F:/LLM/projects/agent/tests/fixtures/evaluation/feedback_adaptation/materials" `
  --relative-path reading_notes.md --start-line 6 --end-line 10 `
  --repeat 5 --output "F:/LLM/projects/agent/.evaluation-results/demo-mcp-comparison.json"
```

核对all_results_equal及各pair，不只看耗时。已记录的一次实测为5/5等价，本地中位1.231ms、MCP中位4.381ms、建立连接与适配1637.871ms；这些是特定环境小样本，不是固定验收阈值或稳定性能结论。

## 5. 验收分层

| 层次 | 观察依据 | 不代表 |
| --- | --- | --- |
| 工具 | 实际执行、结果状态、范围、截断 | 最终回答正确 |
| 状态 | 候选、确认结果、前后对象比较 | 候选适合用户 |
| 恢复 | 独立进程加载、当前字段一致 | 全部旧证据已验证 |
| 内容 | 对照原文与用户要求人工核对 | 用户已学会 |

此前真实闭环的控制链路通过，但被否决候选存在画图与方式限制冲突，采用候选的负荷仍需人工判断。参数校验错误曾恢复成功，不能在未记录错误参数时猜具体原因。结构检查兼容了括号标题，不等于能解析所有Markdown。

## 6. 回归与停止条件

```powershell
uv run python -m pytest tests -q -p no:cacheprovider
```

报告保留实际命令、版本/日期、通过/失败/跳过、人工发现和局限；跳过不能算功能通过。没有新改动无需每次重复付费在线测试。新资料、存档格式或权限行为变化后再选相关案例复验。

本阶段完成后停止增加MCP工具和错误分类。后续先根据真实使用解决可复现的计划约束/负荷或记忆管理需求，不为展示框架增加多Agent，也不将全文计划替换机械改成字符串局部编辑。
