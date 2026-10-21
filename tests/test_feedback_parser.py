"""自然语言反馈解析与确认门控测试。

使用假模型返回值，不调用在线服务。
通过这些测试不代表真实模型一定能正确理解自然语言。
"""

import builtins

import pytest
from langchain_core.messages import AIMessage

import ai_agent.learning.feedback_parser as parser_module
import ai_agent.main as main_module
from ai_agent.runtime.agents import LLMInvocationError
from ai_agent.learning.learning_state import LearningState, LearningTask


def make_state():
    return LearningState(
        learning_goal="理解文章观点",
        plan_markdown="阅读文章并比较观点。",
        tasks=(
            LearningTask(task_id="task-001", title="阅读文章"),
        ),
    )


def mock_model(monkeypatch, content):
    monkeypatch.setattr(
        parser_module,
        "invoke_llm_response",
        lambda llm, messages, *, stage: AIMessage(content=content),
    )


def test_parser_returns_suggestion_without_updating_state(monkeypatch):
    mock_model(
        monkeypatch,
        '{"activity_status":"in_progress",'
        '"understanding_status":"self_reported_unclear"}',
    )
    state = make_state()
    text = "文章只读了一半，两个观点的区别还是不理解。"

    suggestion = parser_module.propose_user_feedback(
        state, "task-001", text, object(),
    )

    assert suggestion.task_id == "task-001"
    assert suggestion.text == text
    assert suggestion.activity_status == "in_progress"
    assert suggestion.understanding_status == "self_reported_unclear"
    assert state.tasks[0].activity_status == "not_started"


@pytest.mark.parametrize("content", [
    "这不是 JSON",
    '{"activity_status":"finished","understanding_status":null}',
    '{"activity_status":null,"understanding_status":null}',
    '{"activity_status":"completed","understanding_status":null,'
    '"task_id":"task-002"}',
])
def test_invalid_model_output_is_rejected(monkeypatch, content):
    mock_model(monkeypatch, content)

    with pytest.raises(ValueError):
        parser_module.propose_user_feedback(
            make_state(), "task-001", "读完了", object(),
        )


def test_unknown_task_does_not_call_model(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("任务不存在时不应调用模型")

    monkeypatch.setattr(
        parser_module, "invoke_llm_response", unexpected_call,
    )

    with pytest.raises(ValueError, match="任务不存在"):
        parser_module.propose_user_feedback(
            make_state(), "missing", "读完了", object(),
        )


@pytest.mark.parametrize("answer", ["n", "", "其他内容"])
def test_unconfirmed_suggestion_never_calls_update(monkeypatch, answer):
    mock_model(
        monkeypatch,
        '{"activity_status":"completed","understanding_status":null}',
    )
    monkeypatch.setattr(builtins, "input", lambda _: answer)

    def unexpected_update(*args, **kwargs):
        raise AssertionError("未确认时不能调用状态更新函数")

    monkeypatch.setattr(
        main_module, "apply_user_feedback", unexpected_update,
    )
    state = make_state()

    updated = main_module._handle_report_command(
        state, "task-001 文章读完了。", object(),
    )

    assert updated is state


def test_confirmed_suggestion_updates_only_explicit_fields(monkeypatch):
    mock_model(
        monkeypatch,
        '{"activity_status":"completed","understanding_status":null}',
    )
    monkeypatch.setattr(builtins, "input", lambda _: "y")
    state = make_state()

    updated = main_module._handle_report_command(
        state, "task-001 文章读完了。", object(),
    )

    assert updated.tasks[0].activity_status == "completed"
    assert updated.tasks[0].understanding_status == "unknown"
    assert updated.tasks[0].latest_feedback == "文章读完了。"
    assert state.tasks[0].activity_status == "not_started"


def test_model_failure_preserves_state_without_confirmation(monkeypatch):
    def failed_call(*args, **kwargs):
        raise LLMInvocationError("模拟调用失败")

    def unexpected_confirmation(*args):
        raise AssertionError("解析失败不应进入确认")

    monkeypatch.setattr(
        parser_module, "invoke_llm_response", failed_call,
    )
    monkeypatch.setattr(builtins, "input", unexpected_confirmation)
    state = make_state()

    updated = main_module._handle_report_command(
        state, "task-001 文章读完了。", object(),
    )

    assert updated is state


def test_interactive_report_uses_new_branch(monkeypatch, capsys):
    # 除了测试辅助函数，还验证实际交互循环确实进入 /report 分支。
    inputs = iter([
        '/task {"task_id":"task-001","title":"阅读文章"}',
        "/report task-001 文章读完了。",
        "y",
        "/state",
        "exit",
    ])
    mock_model(
        monkeypatch,
        '{"activity_status":"completed","understanding_status":null}',
    )

    def unexpected_chat(**kwargs):
        raise AssertionError("/report 不应进入普通问答")

    created_roles = []

    def fake_create_llm(*, role="standard"):
        # 不创建真实客户端；同时记录程序究竟选择了哪一档模型。
        created_roles.append(role)
        return object()

    monkeypatch.setattr(main_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(main_module, "create_llm", fake_create_llm)
    monkeypatch.setattr(
        main_module, "_invoke_interactive_turn", unexpected_chat,
    )
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    result = main_module.run_interactive_session(
        analysis="资料分析",
        plan="阅读文章。",
        material_context="资料证据",
        learning_goal="理解文章观点",
        strategy="window",
        material_dir="materials",
    )

    output = capsys.readouterr().out
    assert "以下只是建议，尚未写入" in output
    assert "已写入本次会话状态" in output
    assert '"activity_status": "completed"' in output
    assert result == ("阅读文章。", "理解文章观点")
    # 初始化普通问答用 Plus，进入 /report 后才创建 Flash。
    # 本次流程不应创建 enhanced 模型。
    assert created_roles == ["standard", "feedback"]
