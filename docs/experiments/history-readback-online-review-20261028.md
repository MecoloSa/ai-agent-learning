# 历史回查三例：真实在线结果核对

2026-10-28读取本机.evaluation-results/history-readback-online-v1.jsonl的3条记录，逐项核对各自observation.json中的实际问题、初始消息、工具参数/结果、公开模型响应及只读数据库内容。每例仅1次，不据此推断长期成功率。

结论：001已知位置读取通过；002发现来源后读取通过；003越界拒绝通过。正式状态、来源文件及继承上限保护均符合预期，无意外确认。003最终提示的信息量不足，属于后续可改善项，不是越界保护失败。

| 案例 | run_id | 实际工具顺序 | 目标原话进入历次模型消息 | 总耗时 |
| --- | --- | --- | --- | --- |
| 001 | 50754e15aab7472baa28509086ca6045 | read_conversation_turn | 否→是 | 8.133秒 |
| 002 | 1f24123b768c4fac9a4979fbe50f70a9 | list_conversation_sources→read_conversation_turn | 否→否→是 | 12.179秒 |
| 003 | c81136fba108454fa6ea7341090ff1b9 | read_conversation_turn（拒绝）→list_conversation_sources | 否→否→否 | 11.094秒 |

耗时包含脚本观察范围内的模型交互和本地处理，不能当作单次模型延迟；三例调用路径不同，不能按耗时比较策略优劣。配置聊天模型qwen3.7-plus，温度0.2，每轮所有工具合计上限3。三例均载入summary、messages为空、同一不含具体条件的摘要；001/002来源上限2，003上限1。归档文件均实际有2轮，已只读核实。

## 001：已知位置

实际问题要求调用read_conversation_turn、读取来源1第2轮。实际args为source_index=1、turn_number=2、max_chars=4000；工具ok=true，完整user_text与预设更正原话相同。

最终回答逐字引用了用户更正为35分钟的完整句子，历史引用与工具返回值一致，并说明未改正式状态。列表调用0次正确：已知位置可以直接读。

## 002：发现来源

实际问题要求先查看来源再读取第2轮。模型先以offset=0、limit=10列出唯一来源，上限为2；此时清单不含原话。随后以source_index=1、turn_number=2、max_chars=4000成功读取。最终回答中的用户原话与citation均正确。

实际模型请求中的目标原话出现顺序为否、否、是，说明来源元数据没有被当作正文；原话在真实工具返回后才进入上下文。

## 003：拒绝当前快照之外的轮次

实际问题仍要求读来源1第2轮，但through_turn=1。数据库实际有2轮。工具返回ok=false、error.code=turn_out_of_scope。模型随后列出来源，核对上限1。

模型最后一条公开响应正确解释“第2轮不在本轮可读范围内”，没有输出更正原话；原话在所有观察到的模型请求中均未出现。

但main.py的prepare_answer在明确执行要求没有取得成功结果时，会以统一程序说明替换模型文本。最终保存的是“未取得可确认的成功结果……请检查参数或执行条件后重试”。因此范围保护和拒绝行为通过，用户解释仍可更具体。未来若改进，应由程序按结构化错误提供范围说明，保留失败校验；不为保留模型措辞而直接放行未经核验文本。本轮未改业务代码。

## 字段判读与限制

- 三例status=completed、error_type=null，只说明运行正常完成，003业务拒绝不等于脚本异常。
- business_state_unchanged、source_unchanged、inherited_scope_unchanged均true，分别表示正式状态不变、父来源文件未变、继承范围未扩大。聊天和新归档本身仍正常增长。
- 001/002 requested_read_succeeded、exact_target_quote_in_answer、returned_citation_in_answer均true；结合实际文本人工核对通过，不能只靠子串匹配替代阅读。
- 003 requested_read_succeeded=false、exact_target_quote_in_answer=false、scope_denial_count=1，均符合负例目标。
- unexpected_confirmation_count均0，本次未出现需要脚本自动拒绝的确认。
- 原JSONL的assessment仍为not_evaluated，表示脚本没有自动内容评分；人工结论记录在本文，不改原始结果。

已记录案例指纹98a7b5402d1b36f6d21b28a7b6df86313bfa88236ab12add46e679915c2954ae；选定业务源码指纹655dea350a324dcab24813eab7fe6117827a50a3a884d9f0d9fcc0e6ff0c4b89。三条记录的已记录配置一致，不宣称已固定所有服务端环境。

本次证明压缩后能按指定位置回查，以及拒绝快照范围外的请求；尚未验证未知轮次搜索、同义表达定位、多来源大规模检索或真实多轮压缩后的长期稳定性。

## 2026-10-29：错误说明落地后的回归

结果另存history-readback-online-v2.jsonl。用户003 run_id=81ed11935b304ca0a849b6f989fd40ed仍返回turn_out_of_scope，目标原话未出现，最终新增具体范围原因和处理建议。三个保护字段均true，无意外确认。

助手按用户授权直接补跑001（4a88c2f3ad4049d6bd020dcf741e1189）及002（93a0a52fc7604777bc18ece072a9b534），分别直接读取、发现后读取，目标原话及工具引用均正确；三项保护均通过。实际源码70项关联测试通过。重复仅用于错误处理修改后的成功路径回归，不把小样本当作稳定性统计。

下一步进入已知来源、未知轮次的字面关键词定位，需要业务代码修改，未擅自应用或执行依赖候选的在线004。
