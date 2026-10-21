"""状态命令及交互分流测试；不调用在线模型。"""

import builtins

import pytest

import ai_agent.main as main_module
from ai_agent.learning.learning_commands import handle_learning_command
from ai_agent.learning.learning_state import LearningState


def make_state():
    return LearningState(
        learning_goal="理解文章观点",
        plan_markdown="第1天阅读文章。",
    )


def test_register_then_record_feedback():
    state = make_state()
    state, message = handle_learning_command(
        state,
        '/task {"task_id":"task-001","title":"阅读文章"}',
    )
    assert "已登记任务" in message

    updated, message = handle_learning_command(
        state,
        '/feedback {"task_id":"task-001","text":"读完但没理解",'
        '"activity_status":"completed",'
        '"understanding_status":"self_reported_unclear"}',
    )

    assert "已记录用户反馈" in message
    assert updated.tasks[0].activity_status == "completed"
    assert updated.tasks[0].understanding_status == "self_reported_unclear"
    assert state.tasks[0].activity_status == "not_started"


@pytest.mark.parametrize("command", [
    "/task {bad-json}",
    '/task {"task_id":"task-001","title":"重复登记"}',
    '/feedback {"task_id":"missing","text":"完成"}',
    '/feedback {"task_id":"task-001","text":"完成",'
    '"activity_status":"finished"}',
    "/state extra",
])
def test_invalid_command_preserves_original_state(command):
    state, _ = handle_learning_command(
        make_state(),
        '/task {"task_id":"task-001","title":"阅读"}',
    )

    updated, message = handle_learning_command(state, command)

    assert updated is state
    assert "未执行" in message


def test_normal_question_returns_to_chat_flow():
    state = make_state()
    updated, message = handle_learning_command(
        state, "为什么第一天先阅读？",
    )

    assert updated is state
    assert message is None


def test_interactive_commands_do_not_invoke_model(monkeypatch, capsys):
    inputs = iter([
        '/task {"task_id":"task-001","title":"阅读文章"}',
        '/feedback {"task_id":"task-001","text":"已读完",'
        '"activity_status":"completed"}',
        "/state",
        "exit",
    ])

    def unexpected_model_call(**kwargs):
        raise AssertionError("状态命令不应该进入模型问答流程")

    monkeypatch.setattr(main_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(main_module, "create_llm", lambda: object())
    monkeypatch.setattr(
        main_module, "_invoke_interactive_turn", unexpected_model_call,
    )
    monkeypatch.setattr(
        builtins, "input", lambda _: next(inputs),
    )

    result = main_module.run_interactive_session(
        analysis="资料分析",
        plan="第1天阅读文章。",
        material_context="资料证据",
        learning_goal="理解文章观点",
        strategy="window",
        material_dir="materials",
    )

    output = capsys.readouterr().out
    assert "已登记任务" in output
    assert "已记录用户反馈" in output
    assert '"activity_status": "completed"' in output
    assert result == ("第1天阅读文章。", "理解文章观点")
