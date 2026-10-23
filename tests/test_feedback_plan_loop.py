"""反馈到计划调整的组合回归。

假模型只测状态/确认/保存接线；候选是否适合学习、是否满足时间预算，
仍需真实模型和人工核对，不能用本测试替代语义验收。
"""
import builtins
import json

import pytest

import ai_agent.main as main
from ai_agent.learning.learning_state import LearningState, LearningTask
from ai_agent.learning.feedback_proposals import FeedbackProposalArgs, prepare_feedback_proposal
from ai_agent.learning.feedback_runtime import commit_feedback_proposal
from ai_agent.learning.task_cancellation import TaskCancellation, cancel_learning_task
from ai_agent.persistence.memory import ConversationMemory
from ai_agent.persistence.state_store import load_learning_state


def initial_state():
    original = LearningState(
        learning_goal="区分两类停止",
        plan_markdown="# 原计划\n阅读10分钟；复述15分钟；画图10分钟。",
        tasks=(
            LearningTask(task_id="read-001", title="阅读两类停止的定义", activity_status="completed",
                         understanding_status="self_reported_unclear", latest_feedback="读完但仍不理解",
                         feedback_source="user_report"),
            LearningTask(task_id="explain-002", title="复述区别"),
            LearningTask(task_id="draw-003", title="画图", activity_status="in_progress"),
        ),
    )
    original = cancel_learning_task(original, TaskCancellation(task_id="draw-003", text="取消画图，先不改计划"))
    proposal = prepare_feedback_proposal(original, FeedbackProposalArgs(
        scope="current_context", task_id=None, activity_status=None, understanding_status=None,
    ), user_message="今天剩20分钟，请记为当前安排限制，先不改计划")
    # 显式模拟已确认的测试初始数据，不伪称真实用户曾作出这些反馈。
    return commit_feedback_proposal(original, proposal)


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
@pytest.mark.parametrize("answer", ["y", "n"])
def test_adjustment_receives_feedback_and_preserves_learning_records(monkeypatch, tmp_path, strategy, answer):
    original = initial_state()
    before_data = original.model_dump()
    candidate = "# 候选计划\n保留已完成阅读记录；先比较定义8分钟，再复述12分钟；不再安排画图。"
    path = tmp_path / "current.json"
    inputs = iter(["依据已确认反馈调整后续安排，不修改任务记录", answer, "只查看当前计划", "exit"])
    requests = []

    def read(_):
        value = next(inputs)
        if value == answer:
            assert not path.exists(), "确认之前不应保存候选计划"
        return value

    def invoke(**kwargs):
        requests.append(list(kwargs["request_messages"]))
        if len(requests) == 1:
            assert kwargs["feedback_inbox"].pending is None
            return f"===NEW_PLAN_START===\n{candidate}\n===NEW_PLAN_END==="
        return "以当前生效计划为准。"

    monkeypatch.setattr(builtins, "input", read)
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(main, "_invoke_interactive_turn", invoke)
    monkeypatch.setattr(ConversationMemory, "_maybe_compress", lambda self: None)
    main.run_interactive_session("", original.plan_markdown, "", original.learning_goal, strategy, str(tmp_path),
                                 initial_state=original, current_state_path=path)

    received = next(json.loads(message.content) for message in requests[0]
                    if message.type == "human" and str(message.content).startswith('{"context_type": "current_learning_state"'))
    assert received["tasks"][0]["activity_status"] == "completed"
    assert received["tasks"][0]["understanding_status"] == "self_reported_unclear"
    assert received["tasks"][2]["is_cancelled"] is True
    assert received["current_context_feedback"][0]["text"] == original.feedback_records[0].text

    saved = load_learning_state(path)
    adopted = answer == "y"
    assert saved.plan_markdown == (candidate if adopted else original.plan_markdown)
    assert saved.learning_goal == original.learning_goal
    assert saved.initial_plan_markdown == original.initial_plan_markdown
    assert saved.context_revision == (2 if adopted else 1)
    assert saved.tasks_need_review is adopted
    assert saved.tasks == original.tasks
    assert saved.feedback_records == original.feedback_records
    assert saved.conversation_memory.strategy == strategy
    assert saved.plan_markdown in requests[1][0].content
    following = next(json.loads(message.content) for message in requests[1]
                     if message.type == "human" and str(message.content).startswith('{"context_type": "current_learning_state"'))
    visible_key = "previous_context_feedback" if adopted else "current_context_feedback"
    assert len(following[visible_key]) == 1
    assert following[visible_key][0]["context_revision"] == 1
    assert ("confirmed" if adopted else "cancelled") in saved.conversation_memory.messages[1].content
    assert original.model_dump() == before_data

    # 另起一次交互，从刚保存的对象恢复；不复用上一会话的memory。
    # 测的是恢复后实际发给模型的数据，不是模型对这些数据的理解能力。
    source_bytes = path.read_bytes()
    resumed_requests = []
    resumed_inputs = iter(["只复述当前安排和已确认记录，不修改状态", "exit"])

    def resumed_invoke(**kwargs):
        resumed_requests.append(list(kwargs["request_messages"]))
        assert kwargs["feedback_inbox"].pending is None
        return "这里只读取系统提供的当前记录，不产生修改候选。"

    monkeypatch.setattr(builtins, "input", lambda _: next(resumed_inputs))
    monkeypatch.setattr(main, "_invoke_interactive_turn", resumed_invoke)
    resumed_path = tmp_path / "resumed.json"
    main.run_interactive_session(
        "", saved.plan_markdown, "", saved.learning_goal, None, str(tmp_path),
        initial_state=saved, current_state_path=resumed_path,
    )

    assert len(resumed_requests) == 1
    assert saved.plan_markdown in resumed_requests[0][0].content
    restored_view = next(
        json.loads(message.content) for message in resumed_requests[0]
        if message.type == "human"
        and str(message.content).startswith('{"context_type": "current_learning_state"')
    )
    assert restored_view["tasks"][0]["activity_status"] == "completed"
    assert restored_view["tasks"][0]["understanding_status"] == "self_reported_unclear"
    assert restored_view["tasks"][2]["is_cancelled"] is True
    assert restored_view[visible_key][0]["text"] == original.feedback_records[0].text

    resumed = load_learning_state(resumed_path)
    # 新问答会更新聊天记忆，但不能顺便改变业务记录或覆盖恢复来源。
    for field in ("learning_goal", "plan_markdown", "context_revision", "tasks_need_review",
                  "tasks", "feedback_records", "initial_plan_markdown"):
        assert getattr(resumed, field) == getattr(saved, field)
    assert resumed.conversation_memory.strategy == strategy
    assert path.read_bytes() == source_bytes


def test_discussion_only_does_not_adopt_plan_or_mutate_tasks(monkeypatch, tmp_path):
    original = initial_state()
    inputs = iter(["只讨论剩20分钟怎么安排，不要修改计划或记录", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(main, "_invoke_interactive_turn", lambda **kw: "可以优先比较定义，再做复述；这里只讨论。")
    monkeypatch.setattr(ConversationMemory, "_maybe_compress", lambda self: None)
    path = tmp_path / "current.json"
    main.run_interactive_session("", original.plan_markdown, "", original.learning_goal, "window", str(tmp_path),
                                 initial_state=original, current_state_path=path)
    saved = load_learning_state(path)
    assert saved.plan_markdown == original.plan_markdown
    assert saved.context_revision == original.context_revision
    assert saved.tasks == original.tasks
    assert saved.feedback_records == original.feedback_records
    assert len(saved.conversation_memory.messages) == 2
