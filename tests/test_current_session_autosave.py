"""当前存档的业务接线测试。

使用临时目录、假终端输入和假问答；保留真实状态转换与磁盘写入，
验证少生成文件不是靠漏保存实现的，也不覆盖恢复来源和手动快照。
"""

import builtins
import sys

import ai_agent.learning.learning_commands as commands
import ai_agent.main as main
from ai_agent.learning.learning_state import LearningState, LearningTask
from ai_agent.persistence.state_store import load_learning_state, save_learning_state


def make_state():
    return LearningState(
        learning_goal="理解ReAct", plan_markdown="比较正常结束和保护性停止。",
        tasks=(LearningTask(task_id="t1", title="阅读ReAct"),),
    )


def inputs(monkeypatch, values):
    values = iter(values)
    monkeypatch.setattr(builtins, "input", lambda _: next(values))


def test_feedback_reuses_current_but_manual_history_is_independent(monkeypatch, tmp_path):
    current = tmp_path / "current" / "session.json"
    source = save_learning_state(make_state(), tmp_path / "source.json")
    source_bytes = source.read_bytes()
    monkeypatch.setattr(main, "new_current_state_path", lambda: current)
    monkeypatch.setattr(commands, "resolve_project_path", lambda name: tmp_path / "history" / name.split("/")[-1])
    inputs(monkeypatch, [
        '/feedback {"task_id":"t1","text":"读一半","activity_status":"in_progress"}',
        "/save-state",
        '/feedback {"task_id":"t1","text":"读完","activity_status":"completed"}',
        "exit",
    ])
    result = main.run_state_recovery_session(load_learning_state(source))
    assert load_learning_state(current) == result
    assert result.tasks[0].activity_status == "completed"
    assert list(current.parent.glob("*.json")) == [current]
    history = list((tmp_path / "history").glob("*.json"))
    assert len(history) == 1
    assert load_learning_state(history[0]).tasks[0].activity_status == "in_progress"
    assert source.read_bytes() == source_bytes


def test_query_does_not_create_current_file(monkeypatch, tmp_path):
    path = tmp_path / "current.json"
    monkeypatch.setattr(main, "new_current_state_path", lambda: path)
    inputs(monkeypatch, ["/state", "exit"])
    main.run_state_recovery_session(make_state())
    assert not path.exists()


def test_two_runs_do_not_share_writable_target(monkeypatch, tmp_path):
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    generated = iter(paths)
    monkeypatch.setattr(main, "new_current_state_path", lambda: next(generated))
    first = None
    for index in range(2):
        inputs(monkeypatch, [
            f'/feedback {{"task_id":"t1","text":"反馈{index}"}}', "exit",
        ])
        main.run_state_recovery_session(make_state())
        if index == 0:
            first = paths[0].read_bytes()
    assert paths[0].read_bytes() == first
    assert load_learning_state(paths[1]).tasks[0].latest_feedback == "反馈1"


def test_two_answers_keep_four_messages_in_one_current_file(monkeypatch, tmp_path):
    path = tmp_path / "current.json"
    monkeypatch.setattr(main, "new_current_state_path", lambda: path)
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(main, "_invoke_interactive_turn", lambda **kw: "解释，但不修改计划。")
    inputs(monkeypatch, ["解释正常结束", "再解释保护性停止", "exit"])
    main.run_interactive_session(
        analysis="", plan="原计划", material_context="", learning_goal="理解ReAct",
        strategy="full", material_dir="unused", initial_state=make_state(),
    )
    stored = load_learning_state(path)
    assert len(stored.conversation_memory.messages) == 4
    assert stored.conversation_memory.messages[2].content == "再解释保护性停止"
    assert list(tmp_path.glob("*.json")) == [path]


def test_material_rebinding_and_chat_share_target(monkeypatch, tmp_path):
    source = save_learning_state(make_state(), tmp_path / "source.json")
    original = source.read_bytes()
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "note.md").write_text("ReAct资料", encoding="utf-8")
    current = tmp_path / "current.json"
    monkeypatch.setattr(main, "new_current_state_path", lambda: current)
    received = []
    monkeypatch.setattr(main, "run_interactive_session", lambda **kw: received.append(kw))
    monkeypatch.setattr(sys, "argv", [
        "agent", "--resume-state", str(source), "--interactive", "--materials", str(materials),
    ])
    main.main()
    assert received[0]["current_state_path"] == current
    assert load_learning_state(current) == received[0]["initial_state"]
    assert source.read_bytes() == original
