import pytest

from ai_agent.evaluation.metrics import recall_at_k, reciprocal_rank

def test_recall_at_k():
    ranked = ["a.md#0", "b.md#2", "c.md#1"]
    gold = ["b.md#2", "c.md#1"]

    assert recall_at_k(ranked, gold, k=2) == 0.5
    assert recall_at_k(ranked, gold, k=3) == 1.0

def test_reciprocal_rank():
    ranked = ["a.md#0", "b.md#2", "c.md#1"]

    assert reciprocal_rank(ranked, {"b.md#2"}) == pytest.approx(0.5)
    assert reciprocal_rank(ranked, {"missing.md#0"}) == 0