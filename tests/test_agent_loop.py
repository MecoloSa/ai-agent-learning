from __future__ import annotations

from dataclasses import replace
import json

from langchain_core.messages import AIMessage, ToolMessage

import pytest
import ai_agent.agents as agents_module


def make_tool_call(
    call_id: str,
    *,
    name: str = "list_learning_files",
    args: dict | None = None,
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "id": call_id,
            "name": name,
            "args": args or {},
            "type": "tool_call",
        }],
    )


class FakeTool:
    name = "list_learning_files"

    def __init__(self, result: str = "a.md", error: Exception | None = None):
        self.result = result
        self.error = error
        self.call_count = 0

    def invoke(self, args):
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.result


def prepare_agent(monkeypatch, tool: FakeTool, responses: list[AIMessage]):
    response_iterator = iter(responses)

    monkeypatch.setattr(
        agents_module,
        "build_agent_tools",
        lambda _: [tool],
    )
    monkeypatch.setattr(
        agents_module,
        "create_llm_with_tools",
        lambda tools: object(),
    )
    monkeypatch.setattr(
        agents_module,
        "_invoke_llm_raw",
        lambda llm, messages, *, stage: next(response_iterator),
    )

@pytest.fixture
def analysis_limits(monkeypatch):
    """固定预算，避免开发者本机环境变量影响离线测试。"""
    monkeypatch.setattr(
        agents_module,
        "settings",
        replace(
            agents_module.settings,
            agent_max_tool_calls=12,
            agent_max_no_progress_rounds=2,
        ),
    )
    monkeypatch.setattr(agents_module, "MAX_TOOL_ROUNDS", 12)
    monkeypatch.setattr(
        agents_module,
        "ANALYSIS_TOOL_LIMITS",
        {
            "list_learning_files": 2,
            "read_learning_file": 6,
            "search_learning_files": 4,
            "retrieve_learning_context": 3,
            "summarize_text_statistics": 2,
        },
    )

def test_agent_can_finish_without_tool_call(monkeypatch):
    tool = FakeTool()
    prepare_agent(
        monkeypatch,
        tool,
        [
            AIMessage(
                content="分析完成。[source=a.md, chunk=0]"
            )
        ],
    )

    analysis, trace = agents_module.analyse_materials_agentic(
        "materials",
        "学习 Agent",
    )

    assert analysis.startswith("分析完成")
    assert trace == ""
    assert tool.call_count == 0


def test_agent_returns_tool_result_to_next_round(monkeypatch):
    tool = FakeTool(result="a.md\nb.md")
    prepare_agent(
        monkeypatch,
        tool,
        [
            make_tool_call("call-1"),
            AIMessage(
                content="根据文件清单完成分析。[source=a.md, chunk=0]"
            ),
        ],
    )

    analysis, trace = agents_module.analyse_materials_agentic(
        "materials",
        "学习 Agent",
    )

    assert analysis.startswith("根据文件清单")
    assert "a.md" in trace
    assert tool.call_count == 1


def test_agent_converts_tool_error_into_observation(monkeypatch):
    tool = FakeTool(error=OSError("读取失败"))
    prepare_agent(
        monkeypatch,
        tool,
        [
            make_tool_call("call-1"),
            AIMessage(content="资料读取失败，本轮证据不足。"),
        ],
    )

    analysis, trace = agents_module.analyse_materials_agentic(
        "materials",
        "学习 Agent",
    )

    assert "证据不足" in analysis
    assert "工具执行出错：读取失败" in trace


def test_agent_reports_repeated_call_failure(
    monkeypatch, analysis_limits
):
    tool = FakeTool(result="a.md")
    prepare_agent(
        monkeypatch,
        tool,
        [
            make_tool_call("call-1"),
            make_tool_call("call-2"),
            make_tool_call("call-3"),
        ],
    )
    monkeypatch.setattr(agents_module, "MAX_TOOL_ROUNDS", 3)

    with pytest.raises(agents_module.AnalysisExecutionError) as caught:
        agents_module.analyse_materials_agentic(
            "materials",
            "学习 Agent",
        )

    assert caught.value.reason == "repeated_tool_call"

    # 三次请求只实际执行第一次；后两次在执行前被拒绝。
    assert tool.call_count == 1

def test_pipeline_plan_forces_final_answer_after_one_retrieval(
    monkeypatch,
):
    """pipeline 只允许一次补充检索，达到上限后必须强制生成计划。"""
    tool = FakeTool(result=(
        "[source=a.md, chunk=0]\n"
        "Agent 通过工具调用获得外部信息。"
    ))

    monkeypatch.setattr(agents_module, "needs_rag", lambda _: True)
    monkeypatch.setattr(
        agents_module,
        "build_retrieve_learning_context_tool",
        lambda _: tool,
    )
    monkeypatch.setattr(
        agents_module,
        "create_llm_with_tools",
        lambda tools: object(),
    )
    monkeypatch.setattr(agents_module, "create_llm", lambda: object())
    monkeypatch.setattr(
        agents_module,
        "_invoke_llm_raw",
        lambda llm, messages, *, stage: make_tool_call(
            "plan-call-1",
            name="list_learning_files",
        ),
    )

    final_messages = []

    def fake_final(llm, messages, *, stage):
        final_messages.extend(messages)
        return "第1天：学习工具调用。[source=a.md, chunk=0]"

    monkeypatch.setattr(agents_module, "_invoke_llm", fake_final)

    trace = {}
    plan = agents_module.create_study_plan(
        analysis="Agent 可以调用外部工具。",
        material_context="[source=a.md, chunk=0]\n资料证据",
        learning_goal="学习 Function Calling",
        days=1,
        material_dir="materials",
        mode="pipeline",
        plan_evidence_trace=trace,
    )

    assert plan.startswith("第1天")
    assert trace["initial_evidence"] == (
        "[source=a.md, chunk=0]\n资料证据"
    )
    assert trace["tool_observations"] == [{
        "tool_call_id": "plan-call-1",
        "content": tool.result,
    }]
    assert [
        request["tool_observation_count"]
        for request in trace["requests"]
    ] == [0, 1]

    # 比较记录与实际模型输入，不能仅证明字典里有数据。
    observations = [
        message for message in final_messages
        if isinstance(message, agents_module.ToolMessage)
    ]
    assert observations[0].content == trace["tool_observations"][0]["content"]
    assert tool.result in observations[0].content
    assert "引用附近的具体论断必须" in final_messages[0].content

def test_plan_trace_records_limited_initial_evidence_without_rag(monkeypatch):
    trace = {}
    captured = []

    monkeypatch.setattr(agents_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(
        agents_module, "_limit_evidence_context",
        lambda _: "[source=a.md, chunk=0]\n保留正文",
    )
    monkeypatch.setattr(agents_module, "create_llm", lambda: object())

    def invoke(llm, messages, *, stage):
        captured.extend(messages)
        return "计划"

    monkeypatch.setattr(agents_module, "_invoke_llm", invoke)
    agents_module.create_study_plan(
        "分析", "截断前完整正文", "学习Agent", 1,
        material_dir="materials", mode="pipeline",
        plan_evidence_trace=trace,
    )

    assert trace["initial_evidence"] == "[source=a.md, chunk=0]\n保留正文"
    assert trace["initial_evidence"] in captured[1].content
    assert trace["tool_observations"] == []
    assert trace["requests"][0]["tool_observation_count"] == 0


def test_plan_trace_keeps_failed_tool_observation(monkeypatch):
    tool = FakeTool(error=OSError("读取失败"))
    trace = {}

    monkeypatch.setattr(agents_module, "needs_rag", lambda _: True)
    monkeypatch.setattr(
        agents_module, "build_retrieve_learning_context_tool", lambda _: tool
    )
    monkeypatch.setattr(
        agents_module, "create_llm_with_tools", lambda _: object()
    )
    monkeypatch.setattr(agents_module, "create_llm", lambda: object())
    monkeypatch.setattr(
        agents_module, "_invoke_llm_raw",
        lambda *a, **kw: make_tool_call("failed-call"),
    )
    monkeypatch.setattr(
        agents_module, "_invoke_llm", lambda *a, **kw: "证据不足的计划"
    )

    agents_module.create_study_plan(
        "分析", "初始证据", "学习Agent", 1,
        material_dir="materials", mode="pipeline",
        plan_evidence_trace=trace,
    )

    assert "补充检索执行失败" in trace["tool_observations"][0]["content"]
    assert trace["requests"][-1]["tool_observation_count"] == 1


def test_plan_generation_still_works_without_trace(monkeypatch):
    monkeypatch.setattr(agents_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(agents_module, "create_llm", lambda: object())
    monkeypatch.setattr(
        agents_module, "_invoke_llm", lambda *a, **kw: "计划"
    )

    assert agents_module.create_study_plan(
        "分析", "证据", "学习Agent", 1,
        material_dir="materials", mode="pipeline",
    ) == "计划"

def test_agent_reports_model_round_limit(monkeypatch):
    tool = FakeTool()
    prepare_agent(monkeypatch, tool, [make_tool_call("call-1")])
    monkeypatch.setattr(agents_module, "MAX_TOOL_ROUNDS", 1)

    with pytest.raises(agents_module.AnalysisExecutionError) as caught:
        agents_module.analyse_materials_agentic("materials", "学习 Agent")

    assert caught.value.reason == "model_round_limit"


@pytest.mark.parametrize("mode", ["pipeline", "agent"])
def test_empty_analysis_never_reaches_planner(monkeypatch, mode):
    monkeypatch.setattr(
        agents_module, "_validate_material_dir", lambda _: None
    )
    analysis_name = (
        "analyse_materials_agentic" if mode == "agent"
        else "analyse_materials"
    )
    monkeypatch.setattr(
        agents_module, analysis_name, lambda *args: ("  \n", "证据")
    )

    plan_calls = []

    def unexpected_plan(*args, **kwargs):
        plan_calls.append(True)
        return "不应生成的计划"

    monkeypatch.setattr(
        agents_module, "create_study_plan", unexpected_plan
    )

    with pytest.raises(agents_module.AnalysisExecutionError) as caught:
        agents_module.run_learning_assistant(
            "materials", "学习 Agent", 1, mode
        )

    assert caught.value.reason == "invalid_analysis_output"
    assert plan_calls == []


def test_round_limit_is_recorded_as_failed_evaluation(monkeypatch):
    """真实业务控制流失败后，评测不能因为生成了计划而误记成功。"""
    from ai_agent.evaluation.runner import EvalCase, run_case

    tool = FakeTool()
    prepare_agent(monkeypatch, tool, [make_tool_call("call-1")])
    monkeypatch.setattr(agents_module, "MAX_TOOL_ROUNDS", 1)
    monkeypatch.setattr(
        agents_module, "_validate_material_dir", lambda _: None
    )

    plan_calls = []

    def unexpected_plan(*args, **kwargs):
        plan_calls.append(True)
        return "不应生成的计划"

    monkeypatch.setattr(
        agents_module, "create_study_plan", unexpected_plan
    )

    record = run_case(
        EvalCase("round-limit", "materials", "学习 Agent", 1),
        "agent",
    )

    assert plan_calls == []
    assert record.success is False
    assert record.plan == ""
    assert "AnalysisExecutionError" in record.error
    assert "model_round_limit" in record.error


def test_cli_analysis_failure_exits_without_export(monkeypatch, tmp_path):
    """终端以非零状态退出，且没有导出看起来成功的计划文件。"""
    import sys
    import ai_agent.main as main_module

    output = tmp_path / "should-not-exist.md"
    monkeypatch.setattr(sys, "argv", [
        "ai_agent",
        "--goal", "学习 Agent",
        "--materials", "materials",
        "--output", str(output),
    ])

    def fail_analysis(*args, **kwargs):
        raise agents_module.AnalysisExecutionError(
            "model_round_limit", "模拟分析未完成"
        )

    monkeypatch.setattr(
        main_module, "run_learning_assistant", fail_analysis
    )

    with pytest.raises(SystemExit) as caught:
        main_module.main()

    assert caught.value.code == 1
    assert not output.exists()

@pytest.mark.parametrize("pattern", ["AAA", "ABABAB", "ABCABC"])
@pytest.mark.parametrize("error", [None, OSError("读取失败")])
def test_analysis_blocks_repeated_signatures_before_execution(
    monkeypatch, analysis_limits, pattern, error
):
    """正常或失败调用都不应因交替重复而再次执行。"""
    tool = FakeTool(result="资料证据", error=error)
    tool.name = "read_learning_file"

    prepare_agent(
        monkeypatch,
        tool,
        [
            make_tool_call(
                f"call-{index}",
                name=tool.name,
                args={"relative_path": f"{letter}.md"},
            )
            for index, letter in enumerate(pattern)
        ],
    )

    with pytest.raises(agents_module.AnalysisExecutionError) as caught:
        agents_module.analyse_materials_agentic("materials", "学习 Agent")

    assert caught.value.reason == "repeated_tool_call"
    assert tool.call_count == len(set(pattern))


def test_analysis_allows_same_tool_with_different_files(
    monkeypatch, analysis_limits
):
    """同一工具读取不同文件是正常行为，不能按工具名称误拦截。"""
    tool = FakeTool(result="资料证据")
    tool.name = "read_learning_file"

    prepare_agent(
        monkeypatch,
        tool,
        [
            make_tool_call(
                f"call-{index}",
                name=tool.name,
                args={"relative_path": path},
            )
            for index, path in enumerate(["a.md", "b.md", "c.md"])
        ] + [AIMessage(content="资料分析完成。")],
    )

    analysis, _ = agents_module.analyse_materials_agentic(
        "materials", "学习 Agent"
    )

    assert analysis == "资料分析完成。"
    assert tool.call_count == 3


def test_analysis_total_limit_applies_inside_one_batch(
    monkeypatch, analysis_limits
):
    """一轮请求五次工具，总预算为二时，只能执行两次。"""
    tool = FakeTool(result="资料证据")
    tool.name = "read_learning_file"

    batch = AIMessage(
        content="",
        tool_calls=[
            {
                "id": f"call-{index}",
                "name": tool.name,
                "args": {"relative_path": f"{index}.md"},
                "type": "tool_call",
            }
            for index in range(5)
        ],
    )
    prepare_agent(monkeypatch, tool, [batch])

    monkeypatch.setattr(
        agents_module,
        "settings",
        replace(agents_module.settings, agent_max_tool_calls=2),
    )

    captured = []

    def fake_invoke(llm, messages, *, stage):
        # 保留列表引用：函数后续追加的工具结果也能被检查。
        captured.append(messages)
        return batch

    monkeypatch.setattr(agents_module, "_invoke_llm_raw", fake_invoke)

    with pytest.raises(agents_module.AnalysisExecutionError) as caught:
        agents_module.analyse_materials_agentic("materials", "学习 Agent")

    assert caught.value.reason == "tool_budget_exceeded"
    assert tool.call_count == 2
    assert len(captured) == 1

    observations = [
        message
        for message in captured[0]
        if isinstance(message, ToolMessage)
    ]
    # 被拒绝的三次请求也必须有对应结果。
    assert {message.tool_call_id for message in observations} == {
        f"call-{index}" for index in range(5)
    }


def test_analysis_per_tool_limit_blocks_new_arguments(
    monkeypatch, analysis_limits
):
    """不断更换参数也不能突破单工具额度。"""
    tool = FakeTool(result="资料证据")
    tool.name = "read_learning_file"

    monkeypatch.setitem(
        agents_module.ANALYSIS_TOOL_LIMITS, tool.name, 2
    )
    prepare_agent(
        monkeypatch,
        tool,
        [
            make_tool_call(
                f"call-{index}",
                name=tool.name,
                args={"relative_path": f"{index}.md"},
            )
            for index in range(4)
        ],
    )

    with pytest.raises(agents_module.AnalysisExecutionError) as caught:
        agents_module.analyse_materials_agentic("materials", "学习 Agent")

    assert caught.value.reason == "tool_no_progress"
    assert tool.call_count == 2


def test_analysis_can_finish_after_using_exact_total_budget(
    monkeypatch, analysis_limits
):
    """额度用满仍允许直接输出结论，不能误判为预算失败。"""
    tool = FakeTool()
    monkeypatch.setattr(
        agents_module,
        "settings",
        replace(agents_module.settings, agent_max_tool_calls=1),
    )
    prepare_agent(
        monkeypatch,
        tool,
        [
            make_tool_call("call-1"),
            AIMessage(content="资料分析完成。"),
        ],
    )

    analysis, _ = agents_module.analyse_materials_agentic(
        "materials", "学习 Agent"
    )

    assert analysis == "资料分析完成。"
    assert tool.call_count == 1

def test_analysis_budget_uses_global_and_per_tool_limits(
    monkeypatch, analysis_limits
):
    """工具自身剩余额度不能突破全局剩余额度。"""
    status = agents_module._build_analysis_budget_status(
        tool_names=[
            "read_learning_file",
            "retrieve_learning_context",
        ],
        tool_counts={"read_learning_file": 5},
        executed_calls=10,
        total_limit=12,
        remaining_model_rounds=3,
    )

    assert status["remaining_total"] == 2
    assert status["remaining_by_tool"] == {
        "read_learning_file": 1,
        "retrieve_learning_context": 2,
    }
    assert "read_learning_file" in status["notice"]


@pytest.mark.parametrize(
    "executed_calls, expected_remaining",
    [(11, 1), (12, 0), (13, 0)],
)
def test_analysis_budget_handles_last_and_zero_calls(
    analysis_limits, executed_calls, expected_remaining
):
    """临近耗尽时提示准确；异常计数也不能产生负数。"""
    status = agents_module._build_analysis_budget_status(
        tool_names=["read_learning_file"],
        tool_counts={},
        executed_calls=executed_calls,
        total_limit=12,
        remaining_model_rounds=2,
    )

    assert status["remaining_total"] == expected_remaining
    assert status["remaining_by_tool"]["read_learning_file"] == (
        expected_remaining
    )

    if expected_remaining == 1:
        assert "合计最多还可执行1次" in status["notice"]
    else:
        assert "不要再请求工具" in status["notice"]


def test_analysis_budget_warns_about_last_model_round(analysis_limits):
    """工具额度尚有剩余，也要提醒模型轮数即将耗尽。"""
    status = agents_module._build_analysis_budget_status(
        tool_names=["read_learning_file"],
        tool_counts={},
        executed_calls=0,
        total_limit=12,
        remaining_model_rounds=1,
    )

    assert status["remaining_model_rounds_including_current"] == 1
    assert "本轮是最后一次模型决策" in status["notice"]


@pytest.mark.parametrize("error", [None, OSError("读取失败")])
def test_analysis_budget_updates_without_polluting_evidence(
    monkeypatch, analysis_limits, error
):
    """确认预算实际传给模型，失败执行也扣减，且提示不会不断累积。"""
    tool = FakeTool(result="资料证据", error=error)
    tool.name = "read_learning_file"

    responses = [
        make_tool_call(
            "call-1",
            name=tool.name,
            args={"relative_path": "a.md"},
        ),
        make_tool_call(
            "call-2",
            name=tool.name,
            args={"relative_path": "b.md"},
        ),
        AIMessage(content="资料分析完成。"),
    ]
    prepare_agent(monkeypatch, tool, responses)

    monkeypatch.setattr(
        agents_module,
        "settings",
        replace(agents_module.settings, agent_max_tool_calls=2),
    )

    response_iterator = iter(responses)
    captured_system_contents = []

    def fake_invoke(llm, messages, *, stage):
        # 保存当次内容，而非列表引用，避免后续替换影响检查。
        captured_system_contents.append(messages[0].content)
        return next(response_iterator)

    monkeypatch.setattr(
        agents_module, "_invoke_llm_raw", fake_invoke
    )

    _, trace = agents_module.analyse_materials_agentic(
        "materials", "学习 Agent"
    )

    marker = "\n[程序实时预算；不是资料证据]\n"
    statuses = [
        json.loads(content.split(marker, 1)[1])
        for content in captured_system_contents
    ]

    assert [status["remaining_total"] for status in statuses] == [
        2, 1, 0
    ]
    assert [
        status["remaining_by_tool"]["read_learning_file"]
        for status in statuses
    ] == [2, 1, 0]

    assert all(
        content.count(marker) == 1
        for content in captured_system_contents
    )
    assert "程序实时预算" not in trace
    assert tool.call_count == 2