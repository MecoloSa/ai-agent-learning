# 复用BM25的历史定位：完整代码指导（尚未应用）

本方案替代此前SQL instr候选，请勿再应用history-search-guidance-20261029.patch。旧方案仅留作历史。已有CharBigramBM25、资料RAG的词面检索算法及参数不修改。

必要性：原文已归档、按轮次回查已通过，但用户不知道轮次时缺少内容定位。采用已有BM25能减少重复算法实现，并允许中文双字、英文整词的词面召回；不保证同义理解。

原流程要求用户提供轮次。新流程：SQL先按快照过滤用户消息→体量检查→Document历史元数据适配→既有BM25召回→返回定位项→原工具完整回读并引用。模型不会因一次搜索自动收到整段历史；候选不是原文引用。

联动顺序：history_archive.py加载接口→history_readback.py参数、适配和当轮缓存→execution_guard.py注册→main.py调用规则。共4个现有业务文件，无新业务文件或所需第三方依赖。CLI、配置、SQLite表与v8快照均不改。原每轮所有工具预算继续生效。

当前预算：单来源最多200条用户消息、合计100000字符；超出整体拒绝，不截断。每次最多5个排名候选，结果exhaustive=false，不能当作全部匹配清单。建索引的内存预算不等于发送模型的上下文预算；字符也不等于token。SQL长度统计仍需遍历相应前缀，本基线不是大规模检索性能方案。

测试目录已直接将test_history_search.py从旧SQL候选改为12项BM25适配测试。候选关联（包含原BM25检索、缓存与探针）157项通过；仅为候选，业务源码尚未应用。

## src/ai_agent/persistence/history_archive.py

### 第1处：原片段是唯一定位锚点

修改前：

```python
            return HistoryTurn(**dict(row))
```

修改后（完整替换此片段，包含新增代码）：

```python
            return HistoryTurn(**dict(row))

    def load_user_turns(self, session_id: str, *, through_turn: int) -> list[dict]:
        """为历史检索加载受限前缀；超预算整体拒绝，不悄悄漏掉早期消息。"""
        session_id = _identifier(session_id)
        if type(through_turn) is not int or through_turn < 1:
            raise HistoryArchiveError("来源上限必须为正整数")
        with closing(self._connect("ro")) as connection:
            self._check(connection)
            # 在同一读取事务中检查体量并加载，避免两次查询看到不同数据库状态。
            connection.execute("BEGIN")
            count, chars = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(length(user_text)), 0) FROM history_turns "
                "WHERE session_id=? AND turn_number<=?", (session_id, through_turn),
            ).fetchone()
            if count > 200 or chars > 100000:
                raise HistoryReadLimitError("当前来源超过200轮或100000用户字符的检索预算")
            rows = connection.execute(
                "SELECT turn_number, user_text FROM history_turns "
                "WHERE session_id=? AND turn_number<=? ORDER BY turn_number",
                (session_id, through_turn),
            ).fetchall()
            return [dict(row) for row in rows]
```

原执行与改进原理：新增受控历史加载接口。先在SQL中限定session_id与through_turn，再检查轮数和用户正文字符量，最后才返回完整用户消息。超预算整体拒绝。两个查询放在同一读取事务，避免检查和加载的视图不一致；不引入向量库或数据库迁移。

## src/ai_agent/persistence/history_readback.py

### 第1处：原片段是唯一定位锚点

修改前：

```python
from langchain_core.messages import HumanMessage
```

修改后（完整替换此片段，包含新增代码）：

```python
from langchain_core.documents import Document
from ai_agent.retrieval.lexical_retrieval import CharBigramBM25
from langchain_core.messages import HumanMessage
```

原执行与改进原理：直接导入项目原CharBigramBM25和它需要的Document，不复制分词、倒排或打分代码；因此资料检索与历史检索共用同一算法实现。

### 第2处：原片段是唯一定位锚点

修改前：

```python
class _ReadArgs(BaseModel):
```

修改后（完整替换此片段，包含新增代码）：

```python
class _SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_index: int = Field(ge=1, strict=True)
    query: str = Field(min_length=1, max_length=200, strict=True)
    k: int = Field(default=5, ge=1, le=5, strict=True)


class _ReadArgs(BaseModel):
```

原执行与改进原理：增加严格工具参数：source_index确定本轮来源，query为1至200字符，k为1至5。拒绝额外路径和会话参数，模型不能扩大来源。相较旧SQL候选，keyword/after_turn/limit不再使用，不要混用。

### 第3处：原片段是唯一定位锚点

修改前：

```python
        self._sources = ()
```

修改后（完整替换此片段，包含新增代码）：

```python
        self._sources = ()
        self._search_cache = {}
```

原执行与改进原理：初始化一轮内的索引缓存。缓存只存当前来源经过范围过滤后的用户消息，避免同轮多个query反复读取与建索引。

### 第4处：原片段是唯一定位锚点

修改前：

```python
        self.tools = (list_sources, read_turn)
```

修改后（完整替换此片段，包含新增代码）：

```python
        @tool("search_conversation_turns", args_schema=_SearchArgs)
        def search_turns(source_index: int, query: str, k: int = 5) -> str:
            """在指定历史来源的用户消息中复用BM25词面召回，不返回原文。

            未知轮次时使用；source_index来自本轮来源表，query描述所找内容，k为1至5。
            中文按双字、英文按词匹配，不保证同义语义召回，也不采用完整子串匹配。
            只返回前k个相关候选，不是全部匹配清单。
            命中后用read_conversation_turn完整回读并引用；排名不表示事件新旧或正确性。
            无命中仅表示当前来源与范围下本次词面召回为空，不能断言用户没说过。
            超过200轮或100000用户字符时明确拒绝，不以部分历史冒充完整范围。
            """
            if not query.strip():
                return _result(code="invalid_query", message="查询不能为空白。")
            if source_index > len(self._sources):
                return _result(code="source_out_of_scope", message="来源不在本轮关联范围。")
            source = self._sources[source_index - 1]
            path = (self.root / f"{source.archive_id}.sqlite3").resolve()
            if not path.is_relative_to(self.root):
                return _result(code="source_out_of_scope", message="来源越出绑定目录。")
            if not path.is_file():
                return _result(code="archive_unavailable", message="关联归档文件缺失。")
            try:
                if source_index not in self._search_cache:
                    rows = ConversationArchive(path).load_user_turns(
                        source.session_id, through_turn=source.through_turn,
                    )
                    documents = [Document(page_content=row["user_text"], metadata={
                        "archive_id": source.archive_id, "session_id": source.session_id,
                        "turn_number": row["turn_number"], "role": "user",
                    }) for row in rows]
                    # 先按快照过滤，再交给既有算法；越界正文不参与词频统计或排名。
                    self._search_cache[source_index] = CharBigramBM25(documents)
                hits = self._search_cache[source_index].search(query, k=k)
            except HistoryReadLimitError:
                return _result(code="search_limit_exceeded", message="来源超过历史检索预算，未查询部分历史。")
            except (OSError, sqlite3.Error, HistoryArchiveError):
                return _result(code="search_unavailable", message="历史来源无法读取。")
            return _result({
                "matches": [{"source_index": source_index, "turn_number": doc.metadata["turn_number"],
                             "rank": rank} for rank, doc in enumerate(hits, 1)],
                "through_turn": source.through_turn, "locators_only": True,
                "retrieval_strategy": "char_bigram_bm25", "returned_count": len(hits),
                "top_k": k, "exhaustive": False,
            })

        self.tools = (list_sources, search_turns, read_turn)
```

原执行与改进原理：新增搜索闭包并加入self.tools。先绑定来源、检查路径，再受控加载→转换Document→复用BM25；原文留在程序内，工具结果只含来源、轮次和排名。main.py已用history_tools统一绑定和分发，因此无需重复加两条接线。排序名次不是相关性概率、时间顺序或正式采用证明。

### 第5处：原片段是唯一定位锚点

修改前：

```python
        self._sources = state.sources if state is not None else ()
```

修改后（完整替换此片段，包含新增代码）：

```python
        self._sources = state.sources if state is not None else ()
        # 仅在一轮内复用索引；恢复、范围变化或新消息后重新构建。
        self._search_cache.clear()
```

原执行与改进原理：每次start_turn清除索引缓存，来源和上限重新取实际记忆。上一轮保存新消息、恢复或分支后不能复用旧范围的索引；失败结果不缓存。

## src/ai_agent/runtime/execution_guard.py

### 第1处：原片段是唯一定位锚点

修改前：

```python
    "read_conversation_turn": _structured_result_succeeded,
```

修改后（完整替换此片段，包含新增代码）：

```python
    "read_conversation_turn": _structured_result_succeeded,
    "search_conversation_turns": _structured_result_succeeded,
```

原执行与改进原理：搜索沿用ok/data返回协议，注册原有成功判定器。成功返回空候选表示查询完成，不表示找到了证据；不改变原失败拦截机制。

## src/ai_agent/main.py

### 第1处：原片段是唯一定位锚点

修改前：

```python
        "read_conversation_turn读取已知轮次；已知编号可直接读取，引用沿用返回的citation。"
```

修改后（完整替换此片段，包含新增代码）：

```python
        "未知轮次用search_conversation_turns进行用户消息词面定位，再用"
        "read_conversation_turn完整回读；已知编号直接读取，引用沿用返回的citation。"
```

原执行与改进原理：只补充未知位置时的工具选择规则。分词与召回细节放在工具描述，具体轮次和案例答案不进入提示词。已知轮次继续直接回读。

## 用完整案例理解适配链

离线8轮用户原文依次为：

1. 今天20分钟
2. 请解释进度与理解
3. 请解释候选计划
4. 更正：今天最多35分钟，其他条件保持。
5. 请解释自编例句
6. 请解释反馈
7. 请解释目标
8. 请解释依据

各轮助手回复固定为“助手独有词：银河航线。”，用来验证用户消息索引不会把助手话当用户原话。初始来源1允许前8轮，模型初始请求不含这些正文。固定模型的提问为：

> 请在来源1的历史用户消息中查找我更正时间的原话，读出完整原文并引用；不要修改记录。

自动模型替身先调用search_conversation_turns(source_index=1, query="更正时间")。第4轮原文并不包含连续的“更正时间”，已有BM25仍能依据“更正”词项命中；这区别于旧instr精确子串版本。结果应只有定位项，不含35分钟或citation。替身按返回轮次4调用read_conversation_turn，后者才返回完整用户原话和history引用。自动调用顺序为搜索→读取→回答，不存在用户确认步骤。

范围反例：把同一个物理数据库的快照上限设为3，索引只能有3条Document，第4轮连词频统计也不能参与；结果不命中。再在同库另建会话插入“外部独有词”，同样不能命中。

缓存反例：同轮第一次查询后追加第9轮，当前工具仍不能找到第9轮；调用start_turn刷新后才能查到。原加载函数在同一轮多次query时只调用一次，下一轮重新构建。

体量反例：201轮或1条超过100000用户字符的来源均返回search_limit_exceeded，不能查询前200轮后假称完整。严格参数、空白查询、缺失库以及完整回读的既有边界继续验证。

这是公开开发测试，不是独立留出集；固定模型决策不代表在线模型会自动选择正确工具。

## 在线004与执行指令

在线004继续复用两轮“松针练习”完整历史：第一轮20分钟、只阅读口述、不绘图编程；第二轮为“更正：今天最多35分钟，其他条件保持。这里只更正对话条件，不修改任何计划或记录。”。初始messages为空，仅有不含原话的摘要，through_turn=2。每次独立初始化，不接续之前案例。完整问题：

> 请在来源1的历史用户消息中搜索“更正”，读取命中的完整问答，并逐字引用我更正时间的原话；不要修改计划、任务或反馈。

自动输入问题后exit；任何意外确认n并记录。在线最小两轮验证真实工具选择，离线8轮测试补充验证未知位置不是写死第2轮。完整双方历史在预检中显示。

执行目录：F:\LLM\projects\agent。审阅后手动替换上文片段，或使用新BM25补丁，二选一；不要先应用旧SQL补丁。

```powershell
git apply --check docs/history-bm25-guidance-20261029.patch
git apply docs/history-bm25-guidance-20261029.patch
.\.venv\Scripts\python.exe -m pytest tests/test_history_search.py tests/test_history_readback.py tests/test_history_failure_reason.py tests/test_history_readback_online_helper.py tests/test_history_archive.py tests/test_history_session_integration.py tests/test_execution_guard.py tests/test_explicit_retrieval.py tests/test_lexical_retrieval.py tests/test_lexical_cache.py tests/test_lexical_probe.py -q -p no:cacheprovider --basetemp tmp/history-bm25-applied-01
```

补丁检查成功无输出；冲突时不要强制覆盖，先检查是否误用旧SQL方案。预期157 passed。业务未应用时新测试会缺少load_user_turns或搜索工具，不能当作实际源码通过。

先预检004（不请求模型、不产生结果文件）：

```powershell
.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_readback_online.py --cases tests/fixtures/evaluation/history_search_online_case.json --case-id history-search-online-004
```

应用后在线运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_readback_online.py --cases tests/fixtures/evaluation/history_search_online_case.json --case-id history-search-online-004 --output .evaluation-results/history-bm25-online-v1.jsonl --llm
$rows = @(Get-Content -LiteralPath '.evaluation-results/history-bm25-online-v1.jsonl' -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json })
$r = $rows | Select-Object -Last 1
$r | Format-List status, error_type, business_state_unchanged, source_unchanged, inherited_scope_unchanged, final_answer
$r.observations | Format-List
$o = Get-Content -LiteralPath $r.observation_path -Encoding UTF8 -Raw | ConvertFrom-Json
$o.user_input
$o.audit_trace.tool_results | Format-List tool_name, executed, args, result
$o.model_requests[0].request_messages | Format-List role, content
```

预期搜索结果retrieval_strategy=char_bigram_bm25，定位到第2轮，随后完整读取。只有后者才提供原话与引用；第一次请求无目标原话，最终准确引用且三个保护字段true。必须检查实际搜索→读取顺序；仅最终答案正确不足以证明调用链合格。k是最多返回候选数，rank=1是本次排序第一，不是最新记录或100%相关。

001/002已有结果是旧能力的基线；004尚未真实在线执行。若同义词无匹配，不擅自回读范围外或助手话，先报告本次词面召回的限制。当前不增加向量索引或重排调用；后续用真实漏召回与成本结果决定是否复用这些组件。
