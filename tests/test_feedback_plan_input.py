"""输入适配器契约与真实确认/保存接线；不验证模型计划质量。"""
import builtins
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def api():
    # 候选未应用时报错，不跳过；不能将临时验证当作源码已落地。
    return importlib.import_module("ai_agent.evaluation.feedback_plan_input")


def test_reads_chat_preview_decline_and_exit(api):
    inputs = ["给候选", "v", "n", "exit"]
    reader = api.ReplayInput(inputs)
    assert reader.read_chat() == "给候选"
    assert reader.read_plan_confirmation() == "v"
    assert reader.read_plan_confirmation() == "n"
    assert reader.read_chat() == "exit"
    reader.assert_finished()
    assert inputs == ["给候选", "v", "n", "exit"]
    assert [e["stage"] for e in reader.events] == [
        "chat", "plan_confirmation", "plan_confirmation", "chat",
    ]


@pytest.mark.parametrize("value", ["v", "y", "n", " V ", "Y"])
def test_confirmation_token_cannot_be_chat(api, value):
    reader = api.ReplayInput([value])
    with pytest.raises(api.ReplayMismatch, match="index=0 stage=chat"):
        reader.read_chat()
    assert reader.events == []
    assert reader.read_plan_confirmation() == value


@pytest.mark.parametrize("value", ["exit", "只讨论", "", "确认"])
def test_chat_cannot_be_consumed_by_confirmation(api, value):
    reader = api.ReplayInput([value])
    with pytest.raises(api.ReplayMismatch, match="stage=plan_confirmation"):
        reader.read_plan_confirmation()
    assert reader.events == []
    assert reader.read_chat() == value


def test_exhaustion_is_not_synthetic_user_decline(api):
    reader = api.ReplayInput([])
    with pytest.raises(api.ReplayMismatch, match="输入耗尽"):
        reader.read_plan_confirmation()


def test_early_return_is_not_finished(api):
    reader = api.ReplayInput(["只讨论", "exit"])
    reader.read_chat()
    with pytest.raises(api.ReplayMismatch, match="remaining=1"):
        reader.assert_finished()


def test_reader_copies_script_and_does_not_log_text(api):
    inputs = ["不要把私人文字写进事件"]
    reader = api.ReplayInput(inputs)
    inputs[0] = "调用者修改了原列表"
    assert reader.read_chat() == "不要把私人文字写进事件"
    assert reader.events == [{"index": 0, "stage": "chat"}]


@pytest.fixture
def business(monkeypatch, tmp_path):
    import ai_agent.main as main
    import ai_agent.learning.confirmation as confirmation

    # 未接线的输入入口必须失败，不能等待真实终端或吞掉脚本输入。
    def unexpected_input(prompt=""):
        raise AssertionError("本步尚未支持这个输入入口")
    monkeypatch.setattr(builtins, "input", unexpected_input)
    monkeypatch.setattr(main, "needs_rag", lambda _: False)
    monkeypatch.setattr(main, "create_llm", lambda **_: object())
    monkeypatch.setattr(main, "create_llm_with_tools", lambda *a, **k: object())
    return main, confirmation


def connect_reader(monkeypatch, business, reader):
    main, confirmation = business
    monkeypatch.setattr(main, "input", lambda _: reader.read_chat(), raising=False)
    original = main.read_confirmation

    def plan_confirmation(*args, **kwargs):
        # 只有真实计划确认入口才切换阶段；原预览、trace和确认语义不替换。
        with monkeypatch.context() as scoped:
            scoped.setattr(confirmation, "input", lambda _: reader.read_plan_confirmation(), raising=False)
            return original(*args, **kwargs)
    monkeypatch.setattr(main, "read_confirmation", plan_confirmation)


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_seed_script_drives_real_session_and_preserves_declined_state(
    api, business, monkeypatch, tmp_path, case_index,
):
    from ai_agent.evaluation.feedback_plan_cases import load_feedback_plan_cases
    from ai_agent.persistence.state_store import load_learning_state
    main, _ = business
    cases = load_feedback_plan_cases(Path(__file__).parent / "fixtures/evaluation/feedback_plan_cases.jsonl")
    case = cases[case_index]
    state = load_learning_state(case.initial_state_file)
    source_bytes = case.initial_state_file.read_bytes()
    reader = api.ReplayInput(case.inputs)
    connect_reader(monkeypatch, business, reader)
    candidate = state.plan_markdown + "\n候选附注：控制流程测试，不是模型真实生成。"
    response = "只讨论，不修改。" if case_index == 0 else (
        f"===NEW_PLAN_START===\n{candidate}\n===NEW_PLAN_END==="
    )
    monkeypatch.setattr(main, "_invoke_interactive_turn", lambda **_: response)
    current = tmp_path / "current.json"
    result = main.run_interactive_session(
        "", state.plan_markdown, "", state.learning_goal, "window", str(case.material_dir),
        initial_state=state, current_state_path=current,
    )
    reader.assert_finished()
    saved = load_learning_state(current)
    assert result == (state.plan_markdown, state.learning_goal)
    # 聊天记忆允许变化；不能用整个存档相等判断“拒绝没有改业务状态”。
    assert saved.model_dump(exclude={"conversation_memory"}) == state.model_dump(exclude={"conversation_memory"})
    assert case.initial_state_file.read_bytes() == source_bytes
    assert sum(e["stage"] == "plan_confirmation" for e in reader.events) == (0 if case_index == 0 else 2)


def test_missing_candidate_stops_instead_of_sending_v_to_model(api, business, monkeypatch, tmp_path):
    from ai_agent.learning.learning_state import LearningState
    main, _ = business
    state = LearningState(learning_goal="阅读", plan_markdown="原计划")
    reader = api.ReplayInput(["请给候选", "v", "n", "exit"])
    connect_reader(monkeypatch, business, reader)
    calls = []
    def answer(**kwargs):
        calls.append(kwargs["user_input"])
        return "模型没有给出候选。"
    monkeypatch.setattr(main, "_invoke_interactive_turn", answer)
    with pytest.raises(api.ReplayMismatch, match="stage=chat"):
        main.run_interactive_session(
            "", state.plan_markdown, "", state.learning_goal, "window", str(tmp_path),
            initial_state=state, current_state_path=tmp_path / "current.json",
        )
    assert calls == ["请给候选"]
    assert reader.events == [{"index": 0, "stage": "chat"}]
