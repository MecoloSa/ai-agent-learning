import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tests.helpers import delegated_reading as graph
from tests.helpers.deep_role_experiment import ROLES, ROLE_OPENERS, forward_named_artifacts, project_delegation


def test_every_plannable_role_has_a_short_identity_message():
    assert ROLE_OPENERS.keys() == ROLES.keys()
    assert all(prompt.startswith("你是") and "\n" not in prompt for prompt in ROLE_OPENERS.values())


def test_delegation_comparison_changes_only_worker_request_view():
    background = HumanMessage(content="上游会话记录")
    payload = {
        "user_request": "核对两本书，最后只出一道口头练习",
        "assignment": {"instruction": "核对两本书的指定页，交付各自出处"},
        "authorized_scope": {"a.pdf": [1], "b.pdf": [2]},
        "upstream_evidence": [{"evidence_id": "p1"}],
    }
    messages = [SystemMessage(content="你是证据分析智能体。"), background,
                HumanMessage(content=json.dumps(payload, ensure_ascii=False))]
    a, b, c = (project_delegation(messages, mode) for mode in ("A", "B", "C"))

    assert a is messages
    assert json.loads(b[-1].content) == payload
    assert json.loads(c[-1].content) == {key: value for key, value in payload.items() if key != "user_request"}
    assert "按当前指派" in b[0].content and "按当前指派" in c[0].content
    assert b[1] is c[1] is background
    assert json.loads(messages[-1].content) == payload


@pytest.mark.parametrize("enabled", [True, False])
def test_role_follows_assignment_and_independent_tasks_can_omit_dependencies(tmp_path, enabled):
    book = "book.pdf"
    # ID 故意叫 teach，但这个任务负责取证。角色与 ID 的字面含义无关。
    cases = [
        ("teach", "reader", "找出教材的定义"),
        ("case-2", "analyst", "比较教材中的两个例子"),
        ("case-3", "practice_designer", "依据教材出一道口头练习"),
        ("case-4", "generalist", "把教材中的术语按出现顺序整理"),
        ("case-5", None, "重排教材中的讨论问题"),
        ("case-6", "unknown", "归纳教材中适合课堂讨论的内容"),
    ]
    seen_tools = {}

    async def ask(model_role, stage, messages, tools=()):
        system = messages[0].content
        if stage == "planner":
            assert (graph.PLANNER_ROLES in system) is enabled
            tasks = [{"id": name, "instruction": instruction, "sources": [book],
                      **({"role": role} if role else {})} for name, role, instruction in cases]
            return AIMessage(content=json.dumps({"tasks": tasks}, ensure_ascii=False))
        if stage == "deliver":
            return AIMessage(content="阅读辅助任务已结束")
        role = next(role for name, role, _ in cases if name == stage)
        opener = ROLE_OPENERS.get(role, ROLE_OPENERS["generalist"])
        assert system.splitlines()[0] == (opener if enabled else graph.WORKER.splitlines()[0])
        payload = json.loads(messages[1].content)
        assert payload["assignment"]["id"] == stage
        assert not payload["assignment"].get("depends_on")
        assert "requested_artifacts" not in payload
        assert payload["authorized_scope"] == {book: [1]}
        seen_tools[stage] = {tool.name for tool in tools}
        return AIMessage(content='{"findings": [], "gaps": []}')

    workflow = graph.build_graph(tmp_path, {book: {1}}, ask, [], roomy=True, worker_roles=enabled)
    asyncio.run(workflow.ainvoke({"question": "用这页教材辅助课堂复习"}))
    assert set(seen_tools) == {name for name, _, _ in cases}
    assert all(names == seen_tools["teach"] for names in seen_tools.values())
    assert {"read_pdf", "submit_report"} <= seen_tools["teach"]


def test_handoff_filters_default_evidence_without_losing_state_or_changing_question():
    original = {
        "user_request": "解释错填 ID 的影响，不补读其它页",
        "authorized_scope": {"book.pdf": [41, 42]},
        "assignment": {"required_artifacts": {"extract": ["call_mapping"]}},
        "upstream_reports": {
            "extract": {"status": "done", "gaps": ["资料未说明是否报错"], "findings": [
                {"artifact": "call_mapping", "evidence_ids": ["p41"]},
                {"artifact": "background", "evidence_ids": ["p42"]},
            ]},
            "other": {"status": "unavailable", "reason": "读取失败", "read_failures": [{"page": 99}]},
        },
        "upstream_evidence": [{"evidence_id": "p41"}, {"evidence_id": "p42"}],
    }
    projected = forward_named_artifacts(original)
    assert projected["user_request"] == original["user_request"]
    assert projected["authorized_scope"] == original["authorized_scope"]
    assert projected["upstream_reports"]["extract"]["gaps"] == ["资料未说明是否报错"]
    assert projected["upstream_reports"]["other"]["read_failures"] == [{"page": 99}]
    assert projected["upstream_evidence"] == [{"evidence_id": "p41"}]
    assert len(original["upstream_reports"]["extract"]["findings"]) == 2


def test_unnamed_handoff_keeps_cited_evidence():
    context = {
        "assignment": {}, "upstream_reports": {"A": {"findings": [{"evidence_ids": ["p41"]}]}},
        "upstream_evidence": [{"evidence_id": "p41"}],
    }
    assert forward_named_artifacts(context) == context
