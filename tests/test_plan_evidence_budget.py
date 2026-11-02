"""初始规划证据选择：整块、保序、预算与实际输入一致；无在线调用。"""
import pytest
import ai_agent.runtime.agents as agents


SEP = "\n\n---\n\n"


def test_oversized_first_block_does_not_hide_later_usable_evidence():
    large = "[source=large.md]\n" + "甲" * 60
    small = "[source=small.md]\n可用证据"
    trace = {}
    actual = agents._limit_evidence_context(large + SEP + small, len(small), selection_trace=trace)
    assert actual == small
    assert trace["input_blocks"] == 2 and trace["selected_blocks"] == 1
    assert trace["omitted_blocks"] == 1 and trace["oversized_blocks"] == 1
    assert trace["selected_chars"] == len(actual)


def test_skipping_a_block_preserves_remaining_evidence_order_and_headers():
    first, large, last = "[source=a]\n甲", "[source=b]\n" + "乙" * 100, "[source=c]\n丙"
    expected = first + SEP + last
    trace = {}
    assert agents._limit_evidence_context(SEP.join([first, large, last]), len(expected), selection_trace=trace) == expected
    assert trace["selected_blocks"] == 2 and trace["omitted_blocks"] == 1


@pytest.mark.parametrize("blocks", [["甲乙丙"], ["甲", "乙丙"], [" 甲 ", "\n乙\n", "丙"]])
def test_exact_budget_counts_only_actual_separators(blocks):
    expected = SEP.join(blocks)
    trace = {}
    assert agents._limit_evidence_context(expected, len(expected), selection_trace=trace) == expected
    assert trace["omitted_blocks"] == 0
    assert trace["selected_chars"] == trace["max_chars"]


def test_all_oversized_is_observable_and_does_not_return_a_truncated_fragment():
    trace = {}
    result = agents._limit_evidence_context("甲" * 20 + SEP + "乙" * 30, 10, selection_trace=trace)
    assert result == ""
    assert trace["input_blocks"] == trace["omitted_blocks"] == trace["oversized_blocks"] == 2
    assert trace["selected_blocks"] == trace["selected_chars"] == 0


@pytest.mark.parametrize("budget", [1, 8, 15, 25, 100])
def test_selection_never_exceeds_budget_or_slices_a_block(budget):
    blocks = ["[a]甲", "[b]" + "乙" * 12, "[c]丙"]
    actual = agents._limit_evidence_context(SEP.join(blocks), budget)
    assert len(actual) <= budget
    chosen = actual.split(SEP) if actual else []
    assert all(block in blocks for block in chosen)
    assert [blocks.index(block) for block in chosen] == sorted(blocks.index(block) for block in chosen)


def test_empty_blocks_are_not_fabricated_as_evidence():
    trace = {}
    assert agents._limit_evidence_context(" " + SEP + "\n", 20, selection_trace=trace) == ""
    assert trace["input_blocks"] == trace["selected_blocks"] == trace["omitted_blocks"] == 0


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_budget_requires_positive_integer(budget):
    with pytest.raises(ValueError):
        agents._limit_evidence_context("证据", budget)


@pytest.mark.parametrize("mode", ["pipeline", "agent"])
def test_model_receives_selected_text_and_actual_omission_count(monkeypatch, mode):
    selected = "[source=b.md]\n可用正文"
    context = "[source=a.md]\n" + "甲" * 100 + SEP + selected
    original = agents._limit_evidence_context
    def limited(text, *, selection_trace=None):
        return original(text, len(selected), selection_trace=selection_trace)
    monkeypatch.setattr(agents, "_limit_evidence_context", limited)
    monkeypatch.setattr(agents, "needs_rag", lambda _: False)
    monkeypatch.setattr(agents, "create_llm", lambda: object())
    captured = []
    def invoke(llm, messages, *, stage):
        captured.extend(messages)
        return "计划正文"
    monkeypatch.setattr(agents, "_invoke_llm", invoke)
    trace = {}
    agents.create_study_plan("分析", context, "学习目标", 1,
        material_dir="unused", mode=mode, plan_evidence_trace=trace)
    assert trace["initial_evidence"] == selected
    assert trace["initial_evidence_selection"]["omitted_blocks"] == 1
    assert "预算保留1块，省略1块" in captured[1].content
    assert selected in captured[1].content and "甲" * 100 not in captured[1].content
    assert "预算保留1块，省略1块" not in captured[0].content
