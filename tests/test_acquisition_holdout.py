"""故障留出实验接线：真实规划循环、固定模型动作，无在线请求。"""
import json
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from ai_agent.evaluation import acquisition_pair as experiment
from tests.helpers.acquisition_holdout import fault_factory


def test_timeout_is_returned_as_observation_and_partial_plan_survives(monkeypatch):
    root = Path("tests/fixtures/evaluation/acquisition_holdout/materials")
    definition = json.loads((root.parent / "cases.json").read_text(encoding="utf-8"))
    case = experiment.prepare_case(definition["cases"][0], root)

    class Model:
        def __init__(self, query):
            self.query = query

        def invoke(self, messages):
            if self.query:
                return AIMessage(content="", tool_calls=[{"id": "read", "name": "retrieve_learning_context", "args": {"queries": ["橙签判定条件"]}, "type": "tool_call"}])
            return AIMessage(content="今天30分钟内，先阅读并口头解释蓝签；橙签规则待核实。[source=blue.md, chunk=0]")

    monkeypatch.setattr(experiment.agents, "create_llm_with_tools", lambda _: Model(True))
    monkeypatch.setattr(experiment.agents, "create_llm", lambda: Model(False))
    with patch.object(experiment, "frozen_rag", fault_factory(experiment.frozen_rag)):
        row = experiment.run_one(case, root, "with_facts", 1)
    assert row["status"] == "completed"
    assert row["retrieval_calls"][0]["outcome"] == "timeout"
    assert "补充检索执行失败" in row["plan_trace"]["tool_observations"][0]["content"]
    assert "至少正确判断两个" not in row["plan_trace"]["initial_evidence"]
    assert row["plan_trace"]["initial_acquisition_facts"][0]["omitted_blocks"] == 1


def test_no_hit_fixture_has_no_supplement_and_rules_are_not_sent():
    root = Path("tests/fixtures/evaluation/acquisition_holdout/materials")
    cases = json.loads((root.parent / "cases.json").read_text(encoding="utf-8"))["cases"]
    case = experiment.prepare_case(cases[1], root)
    assert case["supplement_materials"] == []
    assert case["acquisition_facts"][0]["executed_queries"] == 1
    assert case["acquisition_facts"][0]["requested_queries"] == 2
    assert "三个陌生例子" not in case["evidence"]
