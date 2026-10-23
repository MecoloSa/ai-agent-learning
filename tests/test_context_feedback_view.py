"""反馈读取视图：来源版本不等于事实是否仍适用。

保留v6存档和原反馈提议协议，只测试按日、版本分组和恢复后的上下文接线。
假模型不证明语义回答正确；在线仍需核对是否混淆“历史已确认”与“当前适用”。
"""
import builtins
import json
from datetime import timedelta

import pytest

import ai_agent.main as main
from ai_agent.learning import learning_context
from ai_agent.learning.learning_state import (
    FeedbackRecord, LearningState, LearningTask, learning_today, update_learning_context,
)
from ai_agent.persistence.memory import ConversationMemory
from ai_agent.persistence.state_store import load_learning_state, save_learning_state


def make_record(record_id="limit-1", revision=1, days_ago=0, scope="current_context"):
    return FeedbackRecord(
        record_id=record_id, scope=scope,
        task_id="read-001" if scope == "task" else None,
        quote="今天还有20分钟。", text="今天还有20分钟。",
        context_revision=revision, recorded_on=learning_today() - timedelta(days=days_ago),
        activity_status=None, understanding_status=None,
    )


def make_state(records=()):
    return LearningState(
        learning_goal="区分两类停止", plan_markdown="# 原计划\n阅读后复述。",
        tasks=(LearningTask(task_id="read-001", title="阅读定义", activity_status="completed"),),
        feedback_records=records,
    )


def data(state):
    return json.loads(learning_context.build_learning_state_message(state).content)


def test_plan_revision_moves_record_to_reference_without_rewriting_it():
    original = make_state((make_record(),))
    before = original.model_dump()
    revised = update_learning_context(original, plan_markdown="# 新计划\n比较10分钟、复述10分钟。")
    payload = data(revised)
    assert payload["current_context_feedback"] == []
    assert payload["previous_context_feedback"][0]["context_revision"] == 1
    assert revised.context_revision == 2
    assert revised.feedback_records == original.feedback_records
    assert original.model_dump() == before
    assert revised.tasks == original.tasks
    assert revised.tasks_need_review is True


def test_current_and_previous_records_have_separate_groups():
    state = make_state((make_record("old", 1), make_record("new", 2)))
    revised = update_learning_context(state, plan_markdown="新计划")
    payload = data(revised)
    assert [r["record_id"] for r in payload["current_context_feedback"]] == ["new"]
    assert [r["record_id"] for r in payload["previous_context_feedback"]] == ["old"]


@pytest.mark.parametrize("revision", [1, 2])
def test_other_day_is_outside_this_view_not_deleted_or_declared_false(revision):
    original = make_state((make_record(revision=revision, days_ago=1),))
    revised = update_learning_context(original, plan_markdown="新计划")
    payload = data(revised)
    assert payload["current_context_feedback"] == []
    assert payload["previous_context_feedback"] == []
    assert revised.feedback_records == original.feedback_records


@pytest.mark.parametrize("scope,revision,days_ago", [
    ("task", 1, 0), ("current_context", 3, 0), ("current_context", 1, -1),
])
def test_task_future_revision_and_future_day_not_in_context_view(scope, revision, days_ago):
    original = make_state((make_record(scope=scope, revision=revision, days_ago=days_ago),))
    revised = update_learning_context(original, plan_markdown="新计划")
    assert data(revised)["current_context_feedback"] == []
    assert data(revised)["previous_context_feedback"] == []


def test_older_revision_is_neither_automatically_active_nor_overwritten():
    original = make_state((make_record("a"), make_record("b")))
    revised = update_learning_context(original, plan_markdown="新计划")
    payload = data(revised)
    assert len(payload["previous_context_feedback"]) == 2
    # 约束来源版本与当前适用性分开，不绑定一种历史措辞。
    note = payload["feedback_usage_note"]
    assert "分组只表示记录时的计划版本" in note or "分组只记录来源版本" in note
    assert "不代表当前仍适用" in note or "当前适用性需结合用户请求核对" in note
    assert len(revised.feedback_records) == 2


def test_next_day_view_does_not_change_state(monkeypatch):
    original = make_state((make_record(),))
    tomorrow = original.feedback_records[0].recorded_on + timedelta(days=1)
    monkeypatch.setattr(learning_context, "learning_today", lambda: tomorrow)
    assert data(original)["current_context_feedback"] == []
    assert len(original.feedback_records) == 1


def test_registered_tasks_are_not_presented_as_today_schedule():
    original = make_state((make_record(),))
    revised = update_learning_context(original, plan_markdown="今日比较10分钟、复述10分钟。")
    payload = data(revised)
    assert "不等于当前计划的活动或今日安排" in payload["tasks_role"]
    assert "未取消也不代表今天要执行" in payload["tasks_role"]
    assert payload["tasks"][0]["activity_status"] == "completed"
    assert revised.tasks == original.tasks


def test_current_roundtrip_does_not_need_validity_field(tmp_path):
    state = update_learning_context(make_state((make_record(),)), plan_markdown="新计划")
    path = save_learning_state(state, tmp_path / "v6.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "learning-state-v7"
    assert "validity" not in payload["state"]["feedback_records"][0]
    restored = load_learning_state(path)
    assert restored == state
    assert len(data(restored)["previous_context_feedback"]) == 1


@pytest.mark.parametrize("strategy", ["full", "window", "summary"])
def test_restored_session_gets_plan_and_previous_confirmed_feedback(monkeypatch, tmp_path, strategy):
    revised = update_learning_context(make_state((make_record(),)), plan_markdown="# 有效计划\n比较10分钟；复述10分钟。")
    source = save_learning_state(revised, tmp_path / "source.json")
    original_bytes = source.read_bytes()
    restored = load_learning_state(source)
    prompts = []
    inputs = iter(["只概括当前计划和已确认反馈，不修改。", "exit"])

    def invoke(**kwargs):
        prompts.append(kwargs["request_messages"])
        return "测试替身：只检查接线，不证明模型能正确理解。"

    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **kw: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda *a, **kw: object())
    monkeypatch.setattr(main, "_invoke_interactive_turn", invoke)
    monkeypatch.setattr(ConversationMemory, "_maybe_compress", lambda self: None)
    destination = tmp_path / "branch.json"
    main.run_interactive_session("", restored.plan_markdown, "", restored.learning_goal, strategy,
                                 str(tmp_path), initial_state=restored, current_state_path=destination)
    assert revised.plan_markdown in prompts[0][0].content
    assert "不能因此声称计划缺失或拒绝复述" in prompts[0][0].content
    assert "不能用登记任务表代替当前计划概括" in prompts[0][0].content
    assert "current_context记录默认只在当天、当前计划版本内主动使用" not in prompts[0][0].content
    payload = next(json.loads(m.content) for m in prompts[0] if m.type == "human"
                   and str(m.content).startswith('{"context_type": "current_learning_state"'))
    assert payload["tasks_need_review"] is True
    assert "不等于当前计划的活动或今日安排" in payload["tasks_role"]
    assert payload["previous_context_feedback"][0]["context_revision"] == 1
    saved = load_learning_state(destination)
    assert saved.context_revision == 2
    assert saved.tasks_need_review is True
    assert saved.plan_markdown == revised.plan_markdown
    assert saved.feedback_records == revised.feedback_records
    assert source.read_bytes() == original_bytes
