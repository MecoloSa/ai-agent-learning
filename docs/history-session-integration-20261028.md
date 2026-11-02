# 真实会话原文归档接线：完整代码与验收记录

本方案涉及6个业务文件：main.py、memory.py、memory_state.py、state_store.py、history_archive.py及新增history_session.py。以下保留应用前的唯一定位片段与完整替换内容，新增文件含所有导入。2026-10-28已核实运行源码应用，补齐state_store.py遗漏的`state: LearningState`后，实际源码核心测试53 passed、完整回归1290 passed/1 skipped，离线演示通过。下文补丁命令只用于历史记录，不应重复应用。

## 执行流程与边界

显式--archive-history开启，--no-archive-history关闭；恢复默认沿用快照设置。只记录有非空最终回复的普通问答，不记录状态命令、确认按键、隐藏推理或未完成的半轮。历史已被压缩的内容不补造。

用户原文在strip前捕获，助手正文在清理并显示后、用户确认及摘要之前归档。处理后的确认回执仍由旧记忆链保存，不当作原始助手回复。

归档成功后更新来源水位，正常确认后沿原链保存快照。SQLite和JSON之间没有跨文件原子提交；可能存在已提交但尚未被快照引用的归档行，不能宣称崩溃时两者必定同步。

归档失败时保存pending原文、固定ID及完整本轮聊天，停止确认和摘要并非零退出。恢复先幂等补写pending，再启动模型；如果归档和恢复快照都失败，则明确不能确认可靠落盘。

每次恢复新问答写入新分支，继承来源只允许读到through_turn水位，不能自动看到父会话后来的消息。来源ID不是权限，模型回查工具下一步仍须绑定并校验范围。

快照新写v8，兼容读v1至v7；旧版程序不保证读v8。不批量迁移或改写恢复来源，回退运行旧代码时应使用保留的旧快照。

## src/ai_agent/persistence/history_archive.py

### 第1处：以修改前片段作为唯一定位锚点

修改前：

```python
"""原文归档底座：完整问答原样保存；尚未接入Agent主流程或模型工具。"""
```

修改后：

```python
"""原文归档底座：完整问答原样保存，支持稳定定位与精确读取。"""
```

### 第2处：以修改前片段作为唯一定位锚点

修改前：

```python
    def create_session(self) -> str:
        session_id = uuid4().hex
        with closing(self._connect("rw")) as connection, connection:
            self._check(connection)
            connection.execute(
                "INSERT INTO history_sessions VALUES (?, ?)", (session_id, _now()),
            )
        return session_id
```

修改后：

```python
    def create_session(self) -> str:
        return self.ensure_session(uuid4().hex)

    def ensure_session(self, session_id: str) -> str:
        """支持待归档恢复：会话ID由程序预先分配，重复确保不改变既有记录。"""
        session_id = _identifier(session_id)
        with closing(self._connect("rw")) as connection, connection:
            self._check(connection)
            connection.execute(
                "INSERT OR IGNORE INTO history_sessions VALUES (?, ?)", (session_id, _now()),
            )
        return session_id
```

## src/ai_agent/persistence/memory_state.py

### 第1处：以修改前片段作为唯一定位锚点

修改前：

```python
class StoredMessage(MemoryData):
```

修改后：

```python
class ArchiveSource(MemoryData):
    archive_id: str = Field(pattern=r"^[0-9a-f]{32}$", strict=True)
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$", strict=True)
    # 快照只授权读取当时已提交的前缀，不能自动看到后续追加。
    through_turn: int = Field(gt=0, strict=True)


class PendingArchiveTurn(MemoryData):
    archive_id: str = Field(pattern=r"^[0-9a-f]{32}$", strict=True)
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$", strict=True)
    turn_id: str = Field(pattern=r"^[0-9a-f]{32}$", strict=True)
    user_text: str = Field(min_length=1, strict=True)
    assistant_text: str = Field(min_length=1, strict=True)


class HistoryArchiveState(MemoryData):
    enabled: bool = Field(default=True, strict=True)
    sources: tuple[ArchiveSource, ...] = ()
    pending: PendingArchiveTurn | None = None

    @model_validator(mode="after")
    def unique_sources(self):
        keys = [(s.archive_id, s.session_id) for s in self.sources]
        if len(keys) != len(set(keys)):
            raise ValueError("原文归档来源重复")
        return self


class StoredMessage(MemoryData):
```

### 第2处：以修改前片段作为唯一定位锚点

修改前：

```python
    prompt_history: tuple[str, ...] = ()
```

修改后：

```python
    prompt_history: tuple[str, ...] = ()
    history_archive: HistoryArchiveState | None = None
```

## src/ai_agent/persistence/memory.py

### 第1处：以修改前片段作为唯一定位锚点

修改前：

```python
from ai_agent.persistence.memory_state import MemorySnapshot
```

修改后：

```python
from ai_agent.persistence.memory_state import MemorySnapshot, HistoryArchiveState
```

### 第2处：以修改前片段作为唯一定位锚点

修改前：

```python
    prompt_history: list[str] = field(default_factory=list)
```

修改后：

```python
    prompt_history: list[str] = field(default_factory=list)
    history_archive: HistoryArchiveState | None = None
```

### 第3处：以修改前片段作为唯一定位锚点

修改前：

```python
            "prompt_history": list(self.prompt_history),
```

修改后：

```python
            "prompt_history": list(self.prompt_history),
            "history_archive": self.history_archive,
```

### 第4处：以修改前片段作为唯一定位锚点

修改前：

```python
            prompt_history=list(snapshot.prompt_history),
```

修改后：

```python
            prompt_history=list(snapshot.prompt_history),
            history_archive=snapshot.history_archive,
```

### 第5处：以修改前片段作为唯一定位锚点

修改前：

```python
    def add_ai(self, text: str) -> None:
```

修改后：

```python
    def add_ai(self, text: str, *, compress: bool = True) -> None:
```

### 第6处：以修改前片段作为唯一定位锚点

修改前：

```python
        if self.strategy == "summary":
            try:
```

修改后：

```python
        if self.strategy == "summary" and compress:
            try:
```

## src/ai_agent/persistence/state_store.py

### 第1处：以修改前片段作为唯一定位锚点

修改前：

```python
    # v7增加明确的反馈更正关系；旧记录缺失时默认没有更正关系。
    # 兼容读取v1至v6；新写入使用v7，旧版程序不保证能读取v7。
```

修改后：

```python
    # v8增加原文归档来源和待归档记录；旧存档默认没有归档元数据。
    # 兼容读取v1至v7；新写入v8，旧版程序不保证能读取v8。
```

### 第2处：以修改前片段作为唯一定位锚点

修改前：

```python
        "learning-state-v7",
    ] = "learning-state-v7"
```

修改后：

```python
        "learning-state-v7",
        "learning-state-v8",
    ] = "learning-state-v8"
```

## src/ai_agent/main.py

### 第1处：以修改前片段作为唯一定位锚点

修改前：

```python
from ai_agent.persistence.state_store import load_learning_state, save_current_learning_state
```

修改后：

```python
from ai_agent.persistence.state_store import load_learning_state, save_current_learning_state
from ai_agent.persistence.history_session import (
    ArchiveCapture, HistoryPersistenceError, ARCHIVE_STORAGE_ERRORS,
)
```

### 第2处：以修改前片段作为唯一定位锚点

修改前：

```python
        read_transport: str = "local",
) -> tuple[str, str]:
```

修改后：

```python
        read_transport: str = "local",
        archive_history: bool | None = None,
        history_archive_root: Path | None = None,
) -> tuple[str, str]:
```

### 第3处：以修改前片段作为唯一定位锚点

修改前：

```python
            read_tool_override=reader,
```

修改后：

```python
            read_tool_override=reader,
            archive_history=archive_history,
            history_archive_root=history_archive_root,
```

### 第4处：以修改前片段作为唯一定位锚点

修改前：

```python
        read_tool_override=None,
) -> tuple[str, str]:
```

修改后：

```python
        read_tool_override=None,
        archive_history: bool | None = None,
        history_archive_root: Path | None = None,
) -> tuple[str, str]:
```

### 第5处：以修改前片段作为唯一定位锚点

修改前：

```python
    strategy = memory.strategy

    # 刚进入会话
```

修改后：

```python
    strategy = memory.strategy

    capture = None
    recovered_pending = False
    state_before_archive_recovery = learning_state
    if archive_history is not None or memory.history_archive is not None:
        capture = ArchiveCapture(
            history_archive_root if history_archive_root is not None
            else resolve_project_path(".agents_data/history_archives"),
            memory.history_archive, archive_history,
        )
        if capture.state.pending is not None:
            try:
                capture.flush_pending()
            except ARCHIVE_STORAGE_ERRORS:
                raise HistoryPersistenceError(
                    "待归档原文补写失败，未启动新问答；原文仍在恢复来源快照中。"
                ) from None
            recovered_pending = True
        memory.history_archive = capture.state
        print("[原文归档] " + ("已开启；仅记录完整普通问答。" if capture.state.enabled else "本次关闭。"))

    # 刚进入会话
```

### 第6处：以修改前片段作为唯一定位锚点

修改前：

```python
    def record_ai_response(text: str) -> None:
```

修改后：

```python
    if recovered_pending:
        _autosave_learning_change(
            state_before_archive_recovery, learning_state, current_path=current_state_path,
        )

    def record_ai_response(text: str) -> None:
```

### 第7处：以修改前片段作为唯一定位锚点

修改前：

```python
        "--state-only仅管理记录。完整问答结束后自动保存当前策略与记忆。"
    )

    while True:
        user_input = input("你：").strip()
```

修改后：

```python
        "--state-only仅管理记录。完整问答结束后自动保存当前策略与记忆。"
    )

    while True:
        raw_user_input = input("你：")
        user_input = raw_user_input.strip()
```

### 第8处：以修改前片段作为唯一定位锚点

修改前：

```python
        print(f"{label}：{response_text}\n")

        if pending_feedback is not None:
```

修改后：

```python
        print(f"{label}：{response_text}\n")

        if capture is not None and capture.state.enabled and response_text.strip():
            try:
                archived = capture.capture(raw_user_input, response_text)
            except ARCHIVE_STORAGE_ERRORS:
                # 在确认之前暂停：不采用学习操作或候选，不触发摘要。
                memory.history_archive = capture.state
                memory.add_ai(
                    response_text + "\n\n【程序结果】原文归档失败，候选未确认，学习操作未采用。",
                    compress=False,
                )
                learning_state = LearningState.model_validate({
                    **learning_state.model_dump(),
                    "conversation_memory": memory.export_state().model_dump(),
                })
                try:
                    saved = save_current_learning_state(learning_state, current_state_path)
                except (OSError, ValueError):
                    raise HistoryPersistenceError(
                        "原文归档与恢复快照均保存失败；本轮原文尚未可靠落盘，请保留终端记录。"
                    ) from None
                print(f"[待归档原文已暂存] 恢复快照：{saved}")
                raise HistoryPersistenceError(
                    "归档失败，本轮原文已暂存在快照；会话停止，未继续确认或摘要。"
                ) from None
            memory.history_archive = capture.state
            print(f"[原文归档成功] 第{archived.turn_number}轮，用户消息ID={archived.message_id('user')}")
        elif capture is not None and capture.state.enabled:
            print("[原文归档] 本轮无助手正文，不计归档轮次。")

        if pending_feedback is not None:
```

### 第9处：以修改前片段作为唯一定位锚点

修改前：

```python
    args = parser.parse_args()

    if args.read_transport
```

修改后：

```python
    parser.add_argument(
        "--archive-history", action=argparse.BooleanOptionalAction, default=None,
        help="开启普通问答原文归档；恢复默认沿用保存的设置，--no-archive-history关闭",
    )
    args = parser.parse_args()

    if args.archive_history is not None and (args.state_only or not (args.interactive or args.resume_state)):
        parser.error("原文归档开关只用于交互问答，不能用于仅状态管理或仅生成计划。")

    if args.read_transport
```

### 第10处：以修改前片段作为唯一定位锚点

修改前：

```python
                read_transport=args.read_transport,
            )
```

修改后：

```python
                read_transport=args.read_transport,
                archive_history=args.archive_history,
            )
```

### 第11处：以修改前片段作为唯一定位锚点

修改前：

```python
            read_transport=args.read_transport,
        )
```

修改后：

```python
            read_transport=args.read_transport,
            archive_history=args.archive_history,
        )
```

### 第12处：以修改前片段作为唯一定位锚点

修改前：

```python
    except MCPStartupError:
        print(
```

修改后：

```python
    except HistoryPersistenceError as exc:
        print(f"[原文归档未完成] {exc}")
        raise SystemExit(1) from None
    except MCPStartupError:
        print(
```

### 第13处：以修改前片段作为唯一定位锚点

修改前：

```python
    """CLI边界只处理已明确归类的MCP启动错误。"""
```

修改后：

```python
    """CLI边界处理已归类的MCP启动及原文归档错误。"""
```

## 新建 src/ai_agent/persistence/history_session.py

无修改前代码；完整新文件如下：

```python
"""普通问答原文归档接线；无模型调用，不接受模型指定数据库路径。"""
from __future__ import annotations

from pathlib import Path
import sqlite3
from uuid import uuid4

from ai_agent.persistence.history_archive import ConversationArchive, HistoryArchiveError
from ai_agent.persistence.memory_state import ArchiveSource, HistoryArchiveState, PendingArchiveTurn

ARCHIVE_STORAGE_ERRORS = (OSError, sqlite3.Error, HistoryArchiveError)


class HistoryPersistenceError(RuntimeError):
    """原文归档不能继续；CLI显示明确回执并非零退出。"""


class ArchiveCapture:
    def __init__(self, root: Path, state: HistoryArchiveState | None, enabled: bool | None):
        self.root = Path(root).resolve()
        requested = enabled if enabled is not None else (state.enabled if state is not None else False)
        if state is not None and state.pending is not None and not requested:
            raise HistoryPersistenceError("存在待归档原文，请先开启归档完成恢复，再关闭归档。")
        self.state = HistoryArchiveState(
            enabled=requested,
            sources=state.sources if state is not None else (),
            pending=state.pending if state is not None else None,
        )
        # 每次进入交互都新建分支，不继续向来源会话追加新的问答。
        self._active: tuple[str, str] | None = None

    def flush_pending(self):
        pending = self.state.pending
        if pending is None:
            return None
        previous = next((s for s in self.state.sources
                         if (s.archive_id, s.session_id) == (pending.archive_id, pending.session_id)), None)
        archive_path = (self.root / f"{pending.archive_id}.sqlite3").resolve()
        if not archive_path.is_relative_to(self.root):
            raise HistoryArchiveError("归档文件越出绑定目录")
        archive = ConversationArchive(archive_path)
        if previous is None:
            archive.initialize()
            archive.ensure_session(pending.session_id)
        elif not archive.path.is_file():
            raise HistoryArchiveError("已提交的原文来源缺失，拒绝重建空库")
        turn = archive.append_turn(
            pending.session_id, pending.turn_id, pending.user_text, pending.assistant_text,
        )
        source = ArchiveSource(
            archive_id=pending.archive_id, session_id=pending.session_id,
            through_turn=max(turn.turn_number, previous.through_turn if previous is not None else 0),
        )
        sources = tuple(s for s in self.state.sources
                        if (s.archive_id, s.session_id) != (source.archive_id, source.session_id))
        self.state = HistoryArchiveState(enabled=self.state.enabled, sources=(*sources, source))
        return turn

    def capture(self, user_text: str, assistant_text: str):
        if not self.state.enabled:
            return None
        if self.state.pending is not None:
            raise HistoryArchiveError("待归档消息尚未处理，拒绝覆盖")
        if self._active is None:
            self._active = (uuid4().hex, uuid4().hex)
        pending = PendingArchiveTurn(
            archive_id=self._active[0], session_id=self._active[1], turn_id=uuid4().hex,
            user_text=user_text, assistant_text=assistant_text,
        )
        # 在数据库操作之前记录稳定ID；异常后仍可导出原文并幂等补写。
        self.state = HistoryArchiveState(enabled=True, sources=self.state.sources, pending=pending)
        return self.flush_pending()
```

## 测试与演示

新增tests/test_history_session_integration.py（24项），包括两个独立CLI进程、v1—v7读取、v8写入、失败暂停及幂等恢复。已有写入版本断言同步为v8，旧输入兼容案例不删除。

测试与演示均使用固定模型回复，不访问网络；运行源码已核实完整回归1290 passed、1 skipped。

离线演示tests/helpers/history_session_demo.py使用原开发案例前两轮用户与助手文本，强制在第2轮后摘要。预期记忆仅保留2条近期消息，归档仍可逐字读取第1、2轮；快照v8、来源水位2。演示数据库位于tmp独立目录，不修改个人存档。

### 演示的完整输入与判据

初始状态：独立新会话，目标“离线原文归档演示”，计划“仅测试保存链路，不登记任务。”；summary策略，压缩触发阈值1字符，窗口6。这个极低阈值只用于强制覆盖压缩路径，不是推荐产品配置。模型与摘要均用固定替身，不产生在线费用。

同一会话自动依次输入：

1. `这轮练习名为松针练习。今天最多20分钟，只能阅读与口头说明，不绘图、不编程。这些只是本轮对话条件，不要登记任务或正式反馈。`
2. `更正：今天最多35分钟，其他条件保持。这里只更正对话条件，不修改任何计划或记录。`
3. `exit`

前两轮固定助手回复依次为：

1. `了解，这是本轮对话中的练习条件，不写入正式记录。`
2. `了解，可用时间改为35分钟，其余条件保持，不修改正式记录。`

摘要替身返回`已讨论松针练习的安排；正式状态不变。`，故意不包含具体分钟数。验收的是归档能否逐字保留两轮问答，即使摘要已经丢失数字。不是检验模型能否推断35分钟，也不是验证模型主动检索。

运行源码已执行：输出`schema_version=learning-state-v8`、`retained_memory_messages=2`、`through_turn=2`，第1、2轮原文均一致。终端可能出现“回答未包含资料引用”，本例只讨论对话条件，不要求资料引用，该提示不构成此保存测试失败。

### 关键故障和隔离案例

- **原文保真**：输入`  请解释区别。  `，两端各两个空格；归档必须保留，不能把主流程strip后的文本当原文。
- **确认分离**：初始为“原计划”；依次输入`请给候选`、`y`或`n`、`exit`（两个独立案例）。固定助手正文为下列三行。归档均保存正文，记忆另含程序确认结果；只有y更改正式计划。

```text
===NEW_PLAN_START===
阅读10分钟。
===NEW_PLAN_END===
```

- **失败恢复**：仍为原计划，输入`  请给候选  `，助手返回上述三行；注入数据库写入失败，分为“提交前失败”与“已提交但回执丢失”两个独立案例。预先准备的`y`不得被消费，正式计划不变，快照保存待归档原文。修复故障后恢复，输入`继续解释`、`exit`，固定回复`只解释，不产生候选`。旧问答补写或识别为已写入，轮次不重复；新问答写入新分支，原快照不变。这里恢复的是原文，不自动采用上次候选计划。
- **来源上限**：同一会话输入`第一轮`、`/save-state`、`第二轮`、`exit`；从中间快照恢复并输入`分支问题`、`exit`。中间快照只继承父会话第1轮，新分支单独保存，父数据库与来源快照不改。当前验证元数据上限；模型工具的读取限制在下一步实现。
- **双重失败**：同时注入归档写入和恢复快照保存失败，必须明确提示未可靠落盘，不能输出保存成功。待归档修复失败时也不能先启动模型继续对话。

### 复核记录

执行目录：`F:\LLM\projects\agent`。上文“修改前／修改后”用于复盘；当前源码已经应用，不再执行历史补丁。此次检查发现手动应用时遗漏了`StateSnapshot.state: LearningState`，导致保存时`ValidationError`，已补齐并重新验证。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_history_session_integration.py tests/test_history_archive.py -q -p no:cacheprovider --basetemp tmp/history-session-applied-01
.\.venv\Scripts\python.exe -X utf8 tests/helpers/history_session_demo.py
```

实际结果53 passed；演示显示两轮原文及来源上限2。项目原有日志还会提示“回答未包含资料引用”；本演示无资料问答，不影响保存判据。

完整回归命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp tmp/history-session-full-applied-01
```

实际运行源码完整回归1290 passed、1 skipped，3项第三方依赖弃用警告。测试中有固定替身模型及两个独立CLI进程；未进行真实模型在线质量评估。

