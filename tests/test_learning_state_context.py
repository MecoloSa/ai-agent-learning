"""学习状态进入普通问答的测试。

使用假模型捕获真实请求，检查最新状态是否传入。
不访问在线服务、不写入真实快照。
测试验证上下文接线，不直接证明模型回答质量提高。
"""

import builtins
import json

import pytest

import ai_agent.main as main_module
from ai_agent.learning.learning_context import build_learning_state_message
from ai_agent.learning.learning_state import LearningState, LearningTask, UserFeedback


def test_state_message_contains_tasks_without_full_plan():
    state = LearningState(
        learning_goal="理解ReAct",
        plan_markdown="一份完整计划",
        tasks=(
            LearningTask(
                task_id="task-001",
                title="理解ReAct循环机制",
                understanding_status="self_reported_unclear",
                latest_feedback="不理解如何结束。",
                feedback_source="user_report",
            ),
        ),
    )

    data = json.loads(build_learning_state_message(state).content)

    assert data["context_type"] == "current_learning_state"
    assert data["tasks"][0]["latest_feedback"] == "不理解如何结束。"
    assert data["tasks"][0]["feedback_source"] == "user_report"

    # 任务状态之外，不重复附带初始或当前的完整计划。
    assert "plan_markdown" not in data
    assert "initial_plan_markdown" not in data


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
def test_chat_receives_current_state_after_update_and_cancel(
    monkeypatch, tmp_path, strategy,
):
    inputs = iter([
        '/task {"task_id":"task-001","title":"理解ReAct循环机制"}',
        '/feedback {"task_id":"task-001","text":"还不理解如何结束。",'
        '"understanding_status":"self_reported_unclear"}',
        "针对我的困难，建议下一步做什么？",

        "/report task-001 我已经理解了。",
        "n",
        "现在记录的理解情况是什么？",

        '/feedback {"task_id":"missing","text":"理解了。",'
        '"understanding_status":"self_reported_clear"}',
        "再次确认我的理解情况。",

        '/feedback {"task_id":"task-001","text":"现在理解了。",'
        '"understanding_status":"self_reported_clear"}',
        "现在建议怎么学？",
        "exit",
    ])
    observed_states = []

    def fake_turn(**kwargs):
        messages = kwargs["request_messages"]
        state_messages = []

        for message in messages:
            if message.type != "human":
                continue
            try:
                data = json.loads(str(message.content))
            except ValueError:
                continue
            if (
                isinstance(data, dict)
                and data.get("context_type") == "current_learning_state"
            ):
                state_messages.append(data)

        # 每轮只有当前状态，不能把旧状态不断写进聊天历史。
        assert len(state_messages) == 1
        observed_states.append(state_messages[0])
        return "测试回答，不修改目标或计划。"

    def fake_proposal(state, task_id, text, llm):
        return UserFeedback(
            task_id=task_id,
            text=text,
            understanding_status="self_reported_clear",
        )

    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(main_module, "create_llm", lambda **kwargs: object())
    monkeypatch.setattr(main_module, "_invoke_interactive_turn", fake_turn)
    monkeypatch.setattr(main_module, "propose_user_feedback", fake_proposal)

    # 本测试只验证上下文传递；磁盘可靠性由保存测试单独验证。
    monkeypatch.setattr(
        main_module,
        "save_current_learning_state",
        lambda state, _path: (lambda state: tmp_path / "unused.json")(state),
    )

    main_module.run_interactive_session(
        analysis="资料分析",
        plan="阅读并解释ReAct循环。",
        material_context="测试资料",
        learning_goal="理解ReAct",
        strategy=strategy,
        material_dir="materials",
    )

    tasks = [data["tasks"][0] for data in observed_states]

    assert len(tasks) == 4
    assert tasks[0]["understanding_status"] == "self_reported_unclear"

    # 取消建议和非法任务反馈，都不能改变下一轮看到的状态。
    assert tasks[1] == tasks[0]
    assert tasks[2] == tasks[0]

    assert tasks[3]["understanding_status"] == "self_reported_clear"
    assert tasks[3]["latest_feedback"] == "现在理解了。"

    print(f"\n记忆策略：{strategy}")
    print("问答读取的理解情况：")
    for index, task in enumerate(tasks, start=1):
        print(
            f"第{index}轮：{task['understanding_status']}，"
            f"最新反馈={task['latest_feedback']}"
        )
