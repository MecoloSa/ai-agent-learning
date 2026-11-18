# 教材拒绝后漏提交：快照重放与职责边界诊断

日期：2026-11-05。仅修改测试辅助与文档，业务代码及运行提示词未改。侧栏/权限/统一操作入口设计已保存于../sidebar-and-operation-lifecycle-design.md，仍为待落实方案。

## 问题与条件

原真实教材闭环中，澄清后的首次请求调用propose_plan_change，用户n拒绝；再次相同请求只生成正文，未进入确认。不能将正文自动当作可采用候选。

本轮从clarified.json退回失败之前：移除最后一问一答；第二历史来源through_turn从2退回1，第一来源保留2。真实SQLite归档复制到各实验目录，工具仍受来源范围限制，不读取后续失败答案；正式计划修订号1、任务completed/self_reported_unclear及正式反馈相同。分析、章节证据、模型、预算和工具契约不变，产生候选一律n拒绝后退出。

完整请求：我说的是通用工具与专用工具。今天最多20分钟，只阅读和口头说明。请据此提交完整候选计划供我确认，保留学习目标与任务反馈。

## 单变量条件

- baseline：原历史候选全文与拒绝回执。
- locator：仅在请求副本将已拒绝候选正文换成source_index=2/turn_number=1/read_conversation_turn定位；用户消息、真实拒绝回执保留，原文在归档中可读。
- operation_state：保留全部历史，仅加程序事实数据pending=null、上次cancelled/user_declined，不加操作指令。

顺序baseline/locator/operation_state/operation_state/locator/baseline，各2次；共6条小对照，不反复运行到成功。成功判据为真实工具提交且实际进入确认，不是输出文本标签；普通回答结束也可process_code=0，因此不能只看退出码。

## 结果

| 条件 | 真实提交/运行 | 模型调用 | 输入token | 输出token | 正式状态不变 |
|---|---:|---:|---:|---:|---:|
| baseline | 2/2 | 3 | 27531 | 5840 | 2/2 |
| locator | 2/2 | 2 | 15732 | 3680 | 2/2 |
| operation_state | 1/2 | 2 | 17626 | 3408 | 2/2 |

首次baseline的11条序列化模型输入消息与原失败模型请求逐项相等，却正常提交。只证明消息数组相同；没有原HTTP完整schema/服务端快照，不能宣称所有上游条件字节级一致。baseline第二次自主补读PDF再提交，合理轨迹未被限制。

operation_state第二次复现只输出20分钟计划正文、无工具调用、0次确认。证明补充“当前无候选”的事实不足以消除失败，不证明这条消息导致退化。6条均无误写入，生成后用户n拒绝的5条均保持原状态。

locator减少请求体规模但未证明提交率提升：baseline也2/2。成本差还受baseline一次额外读取、生成长度和缓存影响，不能直接作费用改善结论。它仅测重新生成，没测依赖原草稿的编辑，不得推广成默认删除历史助手正文策略。

总计7次逻辑模型调用、输入60889、输出12928 token。未换算金额；本轮不是语义内容质量全面评测。42项相关离线通过，含3项新视图保全测试及计划/确认历史回归，3项既有依赖警告，未重复全量。

## 已定位的程序边界与未定位的原因

PlanProposalInbox.start_turn清空本轮pending；_propose才登记；prepare_delivery遇到pending=None直接返回None。_invoke_interactive_turn在模型无工具调用时可通过普通回答路径finish_answer结束。这使“写出了候选正文但未创建候选对象”成为允许结束的状态。权限保护正确，计划生成请求的任务完成契约尚未独立表达。

确定：不是每次都必现；不是只有缺少pending事实才能发生；没有发现状态丢失导致误采用。尚未证明模型内部为何在某次选择不调用工具，不能归因为特定旧句子或历史长度。

停止继续刷对照。下一步涉及业务/架构判断：是否将明确生成候选的操作入口与普通聊天结束路径区分，并供侧栏复用。该入口可明确返回真实候选、澄清或未完成；不能把任意计划正文直接解析为授权，也不要求所有普通问答走结构化终结工具。需先给完整调用/返回/确认/保存及兼容方案，再经用户确认落实。

## 重放与查看

入口tests/helpers/book_resubmit_diagnosis.py。项目根目录执行（新输出目录）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tests.helpers.book_resubmit_diagnosis --output-dir .evaluation-results/resubmit-preflight-new
```

默认只生成manifest，不调用模型；加--llm执行6条，调用在线模型并产生测试副本。不要反复重跑此组寻求更好的比例。

原输入、当前状态、完整request_messages、响应、工具及确认记录位于.evaluation-results/book-resubmit-diagnosis-20261105各目录；results.json为执行摘要，readable-transcript.md为完整可读回答。运行前检查源教材存档和归档仍在，不能将其他快照强套为source_index=2/turn=1。
