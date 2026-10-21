"""计划输出检查测试。

使用假模型与假工具，覆盖结构边界、三个返回出口和CLI失败停止。
不调用聊天、Embedding或重排服务。
"""
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from ai_agent.runtime import agents
from ai_agent.runtime.plan_validation import (
    PlanExecutionError,
    finalize_plan_output,
    inspect_plan_format,
)


VALID_PLAN = """### Day 1
**任务1：理解ReAct**
- **学习内容**：工具结果回传
- **具体行动**：阅读并画出消息流
- **完成标志**：能解释tool_call_id
- **证据位置**：[source=book.pdf, page=41]
"""


def test_common_format_and_aliases_are_recognized():
    for text in (
        VALID_PLAN,
        VALID_PLAN.replace("完成标志", "验收标准"),
    ):
        report = inspect_plan_format(text)
        assert report["status"] == "no_structural_warnings"
        assert report["task_count"] == 1
        assert report["warnings"] == []


def test_empty_field_is_not_filled_by_following_citation():
    text = VALID_PLAN.replace("能解释tool_call_id", "")
    report = inspect_plan_format(text)
    assert any("完成标志" in item for item in report["warnings"])


def test_each_task_is_checked_independently():
    text = VALID_PLAN + """
**任务2：比较两种实现**
- **学习内容**：提示词解析与Function Calling
- **具体行动**：列出区别
- **证据位置**：[source=book.pdf, page=42]
"""
    report = inspect_plan_format(text)
    assert report["task_count"] == 2
    assert any(
        "第2个任务" in item and "完成标志" in item
        for item in report["warnings"]
    )


def test_task_examples_in_code_fence_are_not_real_tasks():
    report = inspect_plan_format(
        "```markdown\n" + VALID_PLAN + "\n```"
    )
    assert report["status"] == "unrecognized"


def test_table_is_unrecognized_but_original_text_is_preserved():
    text = "| 任务 | 行动 |\n|---|---|\n| 阅读 | 阅读第一章 |"
    trace = {}
    assert finalize_plan_output(text, trace=trace) == text
    assert trace["plan_output_check"]["status"] == "unrecognized"


@pytest.mark.parametrize("value", [None, "", "  \n", {}])
def test_invalid_text_fails_and_records_reason(value):
    trace = {}
    with pytest.raises(PlanExecutionError) as caught:
        finalize_plan_output(value, trace=trace)
    assert caught.value.reason == "invalid_plan_output"
    assert trace["plan_output_check"]["status"] == "invalid_text"


@pytest.mark.parametrize(
    "path",
    ["without_retrieval", "direct_answer", "forced_finish"],
)
@pytest.mark.parametrize("text", ["", VALID_PLAN])
def test_all_plan_return_paths_run_validation(
    monkeypatch, path, text
):
    calls = []

    def invoke(llm, messages, *, stage):
        calls.append("model")
        return text

    def invoke_raw(llm, messages, *, stage):
        calls.append("model")
        if path == "forced_finish":
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "retrieve_learning_context",
                    "args": {"queries": ["ReAct"]},
                    "id": "call-1",
                    "type": "tool_call",
                }],
            )
        return AIMessage(content=text)

    tool = SimpleNamespace(
        name="retrieve_learning_context",
        invoke=lambda args: calls.append("tool") or "证据",
    )

    monkeypatch.setattr(
        agents, "needs_rag",
        lambda _: path != "without_retrieval",
    )
    monkeypatch.setattr(agents, "create_llm", lambda: object())
    monkeypatch.setattr(
        agents, "create_llm_with_tools", lambda tools: object()
    )
    monkeypatch.setattr(
        agents, "build_retrieve_learning_context_tool",
        lambda _: tool,
    )
    monkeypatch.setattr(agents, "_invoke_llm", invoke)
    monkeypatch.setattr(agents, "_invoke_llm_raw", invoke_raw)

    trace = {}

    def run():
        return agents.create_study_plan(
            "分析",
            "[source=book.pdf, page=41]\n证据",
            "理解ReAct",
            1,
            material_dir="materials",
            mode="pipeline",
            plan_evidence_trace=trace,
        )

    if text:
        assert run() == text
        assert trace["plan_output_check"]["status"] == (
            "no_structural_warnings"
        )
    else:
        with pytest.raises(PlanExecutionError):
            run()

    # pipeline补充检索上限为1；检查器不得增加模型或工具调用。
    assert calls == (
        ["model", "tool", "model"]
        if path == "forced_finish"
        else ["model"]
    )


def test_cli_plan_failure_exits_without_export(
    monkeypatch, tmp_path, capsys
):
    import ai_agent.main as main_module

    output = tmp_path / "must-not-exist.md"

    def fail(*args, **kwargs):
        raise PlanExecutionError(
            "invalid_plan_output", "模拟空计划"
        )

    monkeypatch.setattr(
        main_module, "run_learning_assistant", fail
    )
    monkeypatch.setattr(sys, "argv", [
        "ai_agent",
        "--goal", "理解ReAct",
        "--output", str(output),
    ])

    with pytest.raises(SystemExit) as caught:
        main_module.main()

    assert caught.value.code == 1
    assert not output.exists()
    assert "学习计划未生成有效文本" in capsys.readouterr().out
