"""失败属于本轮工具观察，不能作为下游原始资料。所有模型响应均为本地替身。"""
import json

from langchain_core.messages import AIMessage, ToolMessage

import ai_agent.runtime.agents as agents
from ai_agent.runtime.context_handoff import unpack_material_context


def test_missing_file_then_success_keeps_error_only_in_analysis_messages(
    tmp_path, monkeypatch,
):
    (tmp_path / "notes.md").write_text("叙述视角决定读者能接触的信息。", encoding="utf-8")
    monkeypatch.setattr(agents, "create_llm_with_tools", lambda tools: object())
    # 使用真实绑定目录的读取工具；只替换模型，不替换文件读取及执行循环。
    responses = iter([
        AIMessage(content="", tool_calls=[{
            "id": "missing", "name": "read_learning_file",
            "args": {"relative_path": "missing.md"}, "type": "tool_call",
        }]),
        AIMessage(content="", tool_calls=[{
            "id": "success", "name": "read_learning_file",
            "args": {"relative_path": "notes.md"}, "type": "tool_call",
        }]),
        AIMessage(content="资料说明了叙述视角的信息范围。[source=notes.md]"),
    ])
    observations = []

    def invoke(llm, messages, *, stage):
        observations[:] = [m for m in messages if isinstance(m, ToolMessage)]
        return next(responses)

    monkeypatch.setattr(agents, "_invoke_llm_raw", invoke)
    analysis, context = agents.analyse_materials_agentic(str(tmp_path), "理解叙述视角")
    evidence, facts = unpack_material_context(context)

    assert "叙述视角" in analysis
    assert len(observations) == 2
    assert observations[0].tool_call_id == "missing"
    error = json.loads(observations[0].content)
    assert error["ok"] is False
    assert error["error"]["code"] == "FileNotFoundError"
    assert error["error"]["retryable"] is False
    assert observations[1].tool_call_id == "success"
    assert observations[1].content == (
        "[source=notes.md]\n读取范围：文件开头；截断=False\n"
        "叙述视角决定读者能接触的信息。"
    )
    # 成功结果只在请求副本中整理；下游证据仍是原始成功回执。
    assert json.loads(evidence)["ok"] is True
    assert json.loads(evidence)["data"]["text"] == "叙述视角决定读者能接触的信息。"
    assert "missing.md" not in evidence
    assert "工具执行出错" not in evidence
    assert "notes.md" in evidence
    assert "叙述视角决定读者能接触的信息。" in evidence
    assert facts == [{"tool": "read_learning_file", "status": "failed", "reason": "FileNotFoundError", "source": "missing.md"}]
