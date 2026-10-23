"""通用提示词契约回归：协议保留，业务案例不进入固定规则。

仅验证实际消息组装、字段说明和输入隔离，不声称能自动证明自然语言清晰度。
规则候选尚未应用时显式跳过；应用后启用本模块，不请求在线模型。
"""
from types import SimpleNamespace

import pytest

import ai_agent.main as main
import ai_agent.runtime.prompt_rules as rules

if not hasattr(rules, "EVIDENCE_RULES"):
    pytest.skip("通用提示词整理方案尚未应用", allow_module_level=True)


@pytest.mark.parametrize("retrieval", [True, False])
@pytest.mark.parametrize("goal", ["理解文章中的两种论证方法", "阅读建筑史并比较两种风格", "学习函数参数与返回值"])
def test_static_rules_do_not_depend_on_goal_or_fixture_values(monkeypatch, retrieval, goal):
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(
        interactive_preload_materials=False, interactive_max_retrieval_calls=4,
        interactive_max_queries_per_tool_call=2,
    ))
    prompt = main._build_interactive_system_prompt("分析占位", "计划占位", "正文占位", goal,
        retrieval_available=retrieval, on_demand_eligible=True)
    static = prompt.split("当前学习目标：", 1)[0]
    assert goal not in static
    assert "40分钟" not in static and "60分钟" not in static
    assert "task-001" not in static and "book.pdf" not in static
    assert "所有工具合计最多调用4次" in static
    assert rules.MATERIAL_ACCESS_RULES in static and rules.PDF_CITATION_RULES in static
    assert "不能用登记任务表代替当前计划概括" in static
    assert "只有用户明确要求修改时才提交完整候选" in static
    assert "查不到不能推断事件从未发生" in static
    assert "pending只表示提议已生成" in static
    for marker in (main.PLAN_START_TAG, main.PLAN_END_TAG, main.GOAL_START_TAG, main.GOAL_END_TAG):
        assert static.count(marker) == 1


def test_source_schema_uses_general_fact_and_authorization_rule():
    from ai_agent.learning.feedback_proposals import FeedbackProposalArgs
    text = FeedbackProposalArgs.model_json_schema()["properties"]["source_id"]["description"]
    assert "事实在上文且本轮仅授权记录" in text
    assert "previous_user_*" in text and "current" in text
    assert "请记录刚才那条" not in text


def test_tool_descriptions_distinguish_confirmed_feedback_and_decision_history():
    from ai_agent.learning.feedback_runtime import LearningActionInbox
    inbox = LearningActionInbox()
    assert "正式安排反馈" in inbox.history_tool.description
    assert "程序确认历史" in inbox.confirmation_history_tool.description
    assert "不直接写入" in inbox.tool.description
    assert "完整replacement_text" in inbox.tool.description
    assert "不修改计划" in inbox.task_tool.description
    assert "否决某次提议不等于取消任务" in inbox.cancellation_tool.description


def test_shared_evidence_rules_do_not_hardcode_document_positions():
    assert "同一条证据" in rules.EVIDENCE_RULES
    assert "引用附近的具体论断必须" in rules.EVIDENCE_RULES
    assert "不增加、拼接或合并page/chunk" in rules.EVIDENCE_RULES
    assert "未检索到不等于资料本身未包含" in rules.EVIDENCE_RULES
    assert "book.pdf" not in rules.PDF_CITATION_RULES
