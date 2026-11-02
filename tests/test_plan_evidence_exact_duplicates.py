"""规划前完全重复块去重候选；不调用模型，不改变原文。"""
from ai_agent.runtime.agents import _limit_evidence_context

SEP = "\n\n---\n\n"


def test_duplicate_does_not_displace_unique_evidence():
    first = "[source=a.md]\n定义A"
    last = "[source=b.md]\n定义B"
    expected = SEP.join([first, last])
    trace = {}
    actual = _limit_evidence_context(
        SEP.join([first, first, last]), len(expected), selection_trace=trace,
    )
    assert actual == expected
    assert trace["input_blocks"] == 3
    assert trace["unique_blocks"] == 2
    assert trace["duplicate_blocks"] == 1
    assert trace["omitted_blocks"] == 0
    assert trace["selected_blocks"] == 2


def test_same_text_from_different_sources_is_not_a_duplicate():
    context = SEP.join(["[source=a.md]\n相同正文", "[source=b.md]\n相同正文"])
    assert _limit_evidence_context(context, len(context)) == context


def test_whitespace_is_preserved_not_normalized_for_deduplication():
    context = SEP.join(["[source=a.md]\n定义", "[source=a.md]\n定义 "])
    assert _limit_evidence_context(context, len(context)) == context


def test_duplicate_and_budget_omissions_are_separate():
    first, large = "[source=a.md]\n甲", "[source=b.md]\n" + "乙" * 100
    trace = {}
    actual = _limit_evidence_context(
        SEP.join([first, first, large, large]), len(first), selection_trace=trace,
    )
    assert actual == first
    assert trace["input_blocks"] == 4
    assert trace["unique_blocks"] == 2
    assert trace["duplicate_blocks"] == 2
    assert trace["omitted_blocks"] == trace["oversized_blocks"] == 1
