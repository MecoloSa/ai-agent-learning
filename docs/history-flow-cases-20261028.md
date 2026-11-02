# 原文归档执行流程：四个完整案例

以下复盘当前已落地的归档代码；A、S、T仅为便于阅读的ID代称，真实ID都是32位小写十六进制字符串。

## 案例一：连续问答，然后发生摘要

独立新会话，正式计划不变。summary策略，演示触发阈值1字符，摘要返回固定文本“已讨论松针练习的安排；正式状态不变。”。阈值1只用于覆盖压缩路径，不是实际配置建议。

同一会话自动依次输入：

1. `这轮练习名为松针练习。今天最多20分钟，只能阅读与口头说明，不绘图、不编程。这些只是本轮对话条件，不要登记任务或正式反馈。`
2. `更正：今天最多35分钟，其他条件保持。这里只更正对话条件，不修改任何计划或记录。`
3. `exit`

固定助手回复依次为“了解，这是本轮对话中的练习条件，不写入正式记录。”与“了解，可用时间改为35分钟，其余条件保持，不修改正式记录。”。

main.py先保留原始输入：

```python
raw_user_input = input("你：")
user_input = raw_user_input.strip()
```

业务识别使用user_input，归档使用raw_user_input，避免删除原文两端空格。模型回复显示后调用：

```python
archived = capture.capture(raw_user_input, response_text)
```

capture()第一次分配归档文件A、会话S，随后每轮分配自己的T1/T2；在内存pending中先记住ID及原文，再尝试数据库操作。

```python
if self._active is None:
    self._active = (uuid4().hex, uuid4().hex)
pending = PendingArchiveTurn(
    archive_id=self._active[0], session_id=self._active[1], turn_id=uuid4().hex,
    user_text=user_text, assistant_text=assistant_text,
)
self.state = HistoryArchiveState(enabled=True, sources=self.state.sources, pending=pending)
return self.flush_pending()
```

flush_pending()定位A.sqlite3内的S，完整问答同一行提交。成功后清除pending，把through_turn更新为1或2。随后原链路记录助手回答：

```python
memory.add_ai(text)
```

summary策略此时才可能压缩；第二轮后，旧问答变成摘要，最近一问一答仍在messages。当前快照保存summary、最近消息、来源A/S及上限2；SQLite保留两轮逐字原文。

预期：retained_memory_messages=2，through_turn=2；数据库两轮均可逐字读取。证明原文未随摘要丢失，不证明模型已经能回查。

复现：在项目根目录运行`.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_session_demo.py`。模型和摘要均为替身，无网络。

## 案例二：助手提出候选，用户拒绝

独立新会话，初始计划“原计划”。输入顺序：`请给候选`→`n`→`exit`。固定助手正文：

```text
===NEW_PLAN_START===
阅读10分钟。
===NEW_PLAN_END===
```

显示候选后先归档这段完整正文。接下来确认入口读取n，因此不采用候选。已有记录链在记忆中附上cancelled等程序结果。

预期：正式计划仍为“原计划”；归档assistant_text保留三行候选正文；聊天记忆的助手消息另含取消结果；n不是一条普通问答，不单独生成归档轮次。

这里保存两种信息的原因：归档记录当时显示了什么，程序回执记录最后采用了什么。不能从“归档里有候选”推导“计划已经生效”。y是另一个独立测试，只有y使正式计划成为“阅读10分钟。”。

对应tests/test_history_session_integration.py的test_displayed_candidate_is_separate_from_program_confirmation。

## 案例三：归档失败，停止确认，随后恢复

独立新会话，初始计划“原计划”。预先准备输入`  请给候选  `→`y`，助手返回案例二的三行候选。注入归档异常：

1. 数据库未提交问答就失败。
2. 数据库已提交问答，但调用方收到异常、没收到成功回执。

两种情形分别独立初始化。main.py捕获已归类的存储故障：

```python
memory.history_archive = capture.state
memory.add_ai(
    response_text + "\n\n【程序结果】原文归档失败，候选未确认，学习操作未采用。",
    compress=False,
)
```

capture.state.pending仍含原始A/S/T和双方原文。compress=False使这次应急保存不再触发摘要。程序保存恢复快照后停止，预备的y不会被读取，正式计划不变。

故障解除后，从快照恢复，输入`继续解释`→`exit`；固定助手回复“只解释，不产生候选”。恢复顺序：

```python
if capture.state.pending is not None:
    capture.flush_pending()
```

这发生在新问答模型创建之前。对于第一种失败，原T尚不存在，append_turn真正补写；对于第二种失败，原T已存在且文本相同，append_turn返回原记录，不增加轮次：

```python
old = connection.execute(
    "SELECT * FROM history_turns WHERE turn_id=?", (turn_id,),
).fetchone()
```

预期：旧来源仅1轮，turn_id不变、pending清空。恢复后的新问答进入新文件B、新会话S2，而不是继续向A/S追加。上次候选不会因原文恢复而自动采用。

注意：pending先存在内存里，只有恢复快照保存成功后才落盘。SQLite和JSON不共享事务；若两处均保存失败，程序明确提示未可靠落盘。这里不能承诺任意时刻强杀进程都可恢复。

对应test_failed_archive_stops_before_confirmation_and_pending_recovers（两种故障）及test_double_storage_failure_is_not_reported_as_saved。

## 案例四：从较早快照继续，来源保留当时范围

独立新会话，输入`第一轮`→`/save-state`→`第二轮`→`exit`；两个普通问答固定回复“显示给用户的回答”。

第一轮后，历史快照H记录来源A/S、through_turn=1。随后父会话继续，数据库A实际出现第二轮，父当前快照的上限变为2。历史快照H保持不变。

从H恢复，输入`分支问题`→`exit`，得到同一固定回复。ArchiveCapture的_active从None重新开始：

```python
self._active: tuple[str, str] | None = None
```

因此新问题写入B/S2，新当前快照保存两个来源：

```text
A/S   through_turn=1   继承的旧来源
B/S2  through_turn=1   当前分支新产生的第一轮
```

两个1分别是各来源自己的轮次，不是全局序号。source_index则是回查时本轮来源表的编号，与session_id和turn_number不同。

预期：父数据库和历史快照H内容不变，继承上限仍为1。当前实现已保存该范围；下一轮候选HistoryReadAccess会在数据库读取前拒绝A/S第2轮，不因文件里确实有它就放行。

对应test_snapshot_watermark_does_not_expand_when_origin_continues；下一轮范围拒绝对应tests/test_history_readback.py。

## 本轮新增回读验收的完整背景

新回读演示独立初始化：用案例一的两轮真实文本构造归档；快照messages为空，仅摘要“此前讨论过练习安排；这里没有保留具体条件和逐字原话。”，正式计划“原计划，保持不变。”。

输入：`请实际调用 read_conversation_turn，读取来源1第2轮，逐字引用用户更正时间的完整原话；不要修改计划、任务或反馈。`，随后exit。

模型替身先发出source_index=1、turn_number=2的工具调用；真实工具校验范围并读取数据库；替身收到ToolMessage后才逐字引用其中user_text并附history引用。新回答会按现有保存链写入新分支，不改变正式计划、任务、反馈。

预期original_quote_in_first_request=false、工具executed=true且result.ok=true、最后完整引用更正原话、business_state_unchanged=true。executed只表示进入调用，必须同时看ok。演示observation.json含user_input和每次request_messages，便于核对原话何时进入上下文。

来源发现流程另用固定输入“请先列出历史来源，再读来源1第2轮并引用原话。不要修改记录。”；固定模型先list_conversation_sources、再read_conversation_turn，审计应记录两个成功结果。它验证接线，不验证真实模型自主定位能力。

边界测试将快照来源上限设为1但数据库保留两轮，读取来源1第2轮必须返回turn_out_of_scope且不接触数据库。分页测试构造11个来源，仅列元数据；前10个返回next_offset=10，下页返回第11个。其余测试覆盖错误参数、未知来源、缺失数据库、不支持格式、正文超预算、轮次缺失及全局工具预算，不由LLM判分。
