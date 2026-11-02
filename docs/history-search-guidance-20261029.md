# 下一步：历史用户消息的字面定位（业务代码尚未应用）

2026-10-29方案修正：用户指出应复用既有retrieve组件。核查发现retrieval/lexical_retrieval.py的CharBigramBM25已是纯内存、无文件/模型依赖的通用组件，能够保留历史位置metadata。优先改为“按快照过滤SQLite用户消息→Document→复用BM25→定位→完整回读”。下文SQL instr版本仅保留作旧候选记录，暂停应用其补丁；135项是旧候选成绩，不代表BM25历史适配已实现。已有LearningRAG仍与资料目录、Chroma、文件切分及source/chunk引用耦合，不整类搬入历史会话。

当前进度：用户的新003输出符合预期，完整结果已核对。2026-10-29助手直接补跑001、002在线成功回归，结果另存history-readback-online-v2.jsonl；70项实际源码测试通过。下一缺口是无法按内容找到未知轮次，不需继续重复上述同题。

原流程要求已知来源和轮次。新流程为来源已知、轮次未知时，用关键词获取定位项，再按原工具完整回读。先做字面、区分大小写、用户消息限定的基线，不新增向量模型或数据库表。来源未知仍先列来源。

联动：history_archive.py新增底座方法；history_readback.py新增schema及工具并加入self.tools；execution_guard.py注册成功判定；main.py补充调用顺序。配置、CLI、快照v8、SQLite格式均不变，复用原全局工具预算。12项测试已直接新增，候选关联135项通过；实际业务未应用。

## src/ai_agent/persistence/history_archive.py

### 第1处：原片段为唯一定位锚点

修改前：

```python
            return HistoryTurn(**dict(row))
```

修改后（完整替换片段，新增代码已在其中）：

```python
            return HistoryTurn(**dict(row))

    def find_user_turns(self, session_id: str, keyword: str, *, through_turn: int,
                        after_turn: int = 0, limit: int = 5) -> dict:
        """在会话的快照前缀内做区分大小写的字面子串匹配，只返回定位信息。"""
        session_id = _identifier(session_id)
        if not isinstance(keyword, str) or not keyword.strip() or len(keyword) > 80:
            raise HistoryArchiveError("关键词必须为1至80字符的非空文本")
        if type(through_turn) is not int or through_turn < 1:
            raise HistoryArchiveError("来源上限必须为正整数")
        if type(after_turn) is not int or not 0 <= after_turn <= through_turn:
            raise HistoryArchiveError("分页位置不在来源范围内")
        if type(limit) is not int or not 1 <= limit <= 5:
            raise HistoryArchiveError("每页最多5个定位结果")
        with closing(self._connect("ro")) as connection:
            self._check(connection)
            # instr按字面匹配；参数绑定避免把引号、%或SQL片段解释为查询指令。
            # 上限直接进入SQL，范围外文本不会成为匹配候选。
            rows = connection.execute(
                "SELECT turn_number FROM history_turns "
                "WHERE session_id=? AND turn_number>? AND turn_number<=? "
                "AND instr(user_text, ?) > 0 ORDER BY turn_number LIMIT ?",
                (session_id, after_turn, through_turn, keyword, limit + 1),
            ).fetchall()
        numbers = [row["turn_number"] for row in rows[:limit]]
        return {"turn_numbers": numbers,
                "next_after_turn": numbers[-1] if len(rows) > limit else None}
```

原理、原因与效果：原来只能按已知轮次读取。新增只读SQL字面查询，只匹配指定session_id、快照上限内的user_text。instr不把%当通配符，参数绑定防止SQL注入。多取1项判断是否还有下一页，不返回用户或助手正文。LIMIT限制返回数量，不限制数据库扫描工作；当前为小规模基线，不宣称大库高效。

## src/ai_agent/persistence/history_readback.py

### 第1处：原片段为唯一定位锚点

修改前：

```python
class _ReadArgs(BaseModel):
```

修改后（完整替换片段，新增代码已在其中）：

```python
class _SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_index: int = Field(ge=1, strict=True)
    keyword: str = Field(min_length=1, max_length=80, strict=True)
    after_turn: int = Field(default=0, ge=0, strict=True)
    limit: int = Field(default=5, ge=1, le=5, strict=True)


class _ReadArgs(BaseModel):
```

原理、原因与效果：新增严格参数契约。编号和分页必须为整数，关键词1至80字符，额外目录或会话参数被拒绝。

### 第2处：原片段为唯一定位锚点

修改前：

```python
        self.tools = (list_sources, read_turn)
```

修改后（完整替换片段，新增代码已在其中）：

```python
        @tool("search_conversation_turns", args_schema=_SearchArgs)
        def search_turns(source_index: int, keyword: str, after_turn: int = 0, limit: int = 5) -> str:
            """在指定历史来源的用户消息中按字面关键词定位轮次，不返回原文。

            source_index来自本轮来源表；未知轮次时使用。区分大小写，不支持语义检索。
            after_turn初始为0，按next_after_turn继续；limit为1至5。
            命中后用read_conversation_turn读取完整问答，清单不能代替原话或引用。
            无命中仅表示本来源、快照范围、该字面关键词未匹配，不证明用户未说过相关内容。
            """
            if source_index > len(self._sources):
                return _result(code="source_out_of_scope", message="来源不在本轮关联范围。")
            source = self._sources[source_index - 1]
            if after_turn > source.through_turn:
                return _result(code="turn_out_of_scope", message="分页位置超过快照范围。")
            path = (self.root / f"{source.archive_id}.sqlite3").resolve()
            if not path.is_relative_to(self.root):
                return _result(code="source_out_of_scope", message="来源越出绑定目录。")
            if not path.is_file():
                return _result(code="archive_unavailable", message="关联归档文件缺失。")
            try:
                found = ConversationArchive(path).find_user_turns(
                    source.session_id, keyword, through_turn=source.through_turn,
                    after_turn=after_turn, limit=limit,
                )
            except (OSError, sqlite3.Error, HistoryArchiveError):
                return _result(code="search_unavailable", message="历史字面查询失败，请检查关键词及来源。")
            return _result({
                "matches": [{"source_index": source_index, "turn_number": n}
                            for n in found["turn_numbers"]],
                "next_after_turn": found["next_after_turn"],
                "through_turn": source.through_turn, "locators_only": True,
                "match_mode": "literal_case_sensitive_user_text",
            })

        self.tools = (list_sources, search_turns, read_turn)
```

原理、原因与效果：新工具先检查本轮来源、分页与路径，再调用只读底座。只返回source_index和turn_number定位项，不能当作完整原话；加入self.tools后，main.py已有的统一绑定与extra_tools分发自动接入，不需重复改两条接线。

## src/ai_agent/runtime/execution_guard.py

### 第1处：原片段为唯一定位锚点

修改前：

```python
    "read_conversation_turn": _structured_result_succeeded,
```

修改后（完整替换片段，新增代码已在其中）：

```python
    "read_conversation_turn": _structured_result_succeeded,
    "search_conversation_turns": _structured_result_succeeded,
```

原理、原因与效果：注册同一ok/data协议，让明确点名搜索工具的请求能够进行原成功校验。搜索成功返回空清单属于查询成功，不代表找到了原话。

## src/ai_agent/main.py

### 第1处：原片段为唯一定位锚点

修改前：

```python
        "read_conversation_turn读取已知轮次；已知编号可直接读取，引用沿用返回的citation。"
```

修改后（完整替换片段，新增代码已在其中）：

```python
        "未知轮次时用search_conversation_turns按用户消息的字面关键词定位，再用"
        "read_conversation_turn读取完整问答；已知编号直接读取，引用沿用返回的citation。"
```

原理、原因与效果：原规则只支持发现来源和读取已知轮次。补充未知轮次时的字面搜索顺序；匹配规则写在工具描述中，提示词不写案例数值或答案。

## 新案例的输入与判据

离线8轮独立归档，用户原文依次为：

1. 今天20分钟
2. 请解释进度与理解
3. 请解释候选计划
4. 更正：今天最多35分钟，其他条件保持。
5. 请解释自编例句
6. 请解释反馈
7. 请解释目标
8. 请解释依据

每轮助手固定回复“仅讨论，不修改记录。”。本轮工具可读来源1的前8轮。请求原文：

> 请在来源1的历史用户消息中搜索“更正”，读取命中的完整问答，并逐字引用我更正时间的原话；不要修改计划、任务或反馈。

模型替身先搜索关键词“更正”，真实工具应返回轮次4但不含35分钟原话；随后替身根据返回定位项调用read_conversation_turn，取得并引用原话。这里不给工具写死轮次2。反例将快照上限改为3，虽数据库有第4轮，搜索仍应无命中；另有同库其他会话记录也不能命中。还验证分页、字面%和SQL字符串、只搜索用户而非助手消息、下一轮刷新前范围不扩大。全部为独立测试，不继承线上状态。

在线最小案例004保存在tests/fixtures/evaluation/history_search_online_case.json，与旧案例分文件。归档复用20→35分钟的两轮完整松针练习对话；messages为空，只有不含原话的旧摘要；through_turn=2。问题与上面一致，不提供轮次。自动输入问题后exit，意外确认仍n。预检会完整展示双方历史。该在线小例验证工具选择，不证明大规模定位，8轮离线例补充防止写死第2轮的程序验证。

定位结果参数：after_turn=0从开头搜索；limit最多5条命中；next_after_turn有值则继续传回，无值表示当前范围查询已结束。source_index仅本轮有效，分页不得跨快照范围。空清单只表示该范围、该字面关键词没匹配；不表示相关内容从未出现。

## 应用与验证

执行目录F:\LLM\projects\agent。审阅上文后手动替换，或使用补丁，二选一：

```powershell
git apply --check docs/history-search-guidance-20261029.patch
git apply docs/history-search-guidance-20261029.patch
.\.venv\Scripts\python.exe -m pytest tests/test_history_search.py tests/test_history_readback.py tests/test_history_failure_reason.py tests/test_history_readback_online_helper.py tests/test_history_archive.py tests/test_history_session_integration.py tests/test_execution_guard.py tests/test_explicit_retrieval.py -q -p no:cacheprovider --basetemp tmp/history-search-applied-01
```

预期135 passed。补丁检查成功无输出；冲突时不强行覆盖。源码未应用时新增测试因缺少搜索方法/工具而失败，不算实际源码已验收。

再执行在线预检（不请求模型，不创建结果文件）：

```powershell
.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_readback_online.py --cases tests/fixtures/evaluation/history_search_online_case.json --case-id history-search-online-004
```

应用后真实在线验证：

```powershell
.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_readback_online.py --cases tests/fixtures/evaluation/history_search_online_case.json --case-id history-search-online-004 --output .evaluation-results/history-search-online-v1.jsonl --llm
$rows = @(Get-Content -LiteralPath '.evaluation-results/history-search-online-v1.jsonl' -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json })
$r = $rows | Select-Object -Last 1
$r | Format-List status, business_state_unchanged, source_unchanged, inherited_scope_unchanged, final_answer
$o = Get-Content -LiteralPath $r.observation_path -Encoding UTF8 -Raw | ConvertFrom-Json
$o.user_input
$o.audit_trace.tool_results | Format-List tool_name, executed, args, result
$o.model_requests[0].request_messages | Format-List role, content
$r.observations | Format-List
```

预期先search_conversation_turns成功、再read_conversation_turn读取命中的第2轮；前者仅定位不含原文或citation，后者才提供完整原话与引用。首次请求不含原话、回答逐字引用、正式状态和父来源保持；assessment仍不自动评分。只看最终答案对，不足以证明实际搜索过，必须检查工具顺序。

本轮未执行004真实在线模型，因为对应业务代码尚未应用。关键词换成同义词时可能无命中，这是字面基线的边界；后续根据真实检索缺口再评估分词、FTS或向量检索。
