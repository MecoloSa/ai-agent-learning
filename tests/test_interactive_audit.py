"""交互评测审计轨迹测试。

使用可控的工具和模型响应，验证完整证据、原始回答与最终回答的区分。
同时比较启用和关闭审计时的执行次数与结果，不访问在线服务。
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
import ai_agent.main as main_module

from ai_agent.evaluation.interactive_checks import check_pdf_answer


@pytest.mark.parametrize("audit_enabled", [False, True])
@pytest.mark.parametrize(
    "scenario",
    ["normal", "failed", "recovered", "duplicate"],
)
def test_audit_does_not_change_execution(
    monkeypatch, audit_enabled, scenario,
):
    settings = replace(
        main_module.get_settings(),
        interactive_max_retrieval_calls=2,
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    tool_count = 0

    def fake_tool(args):
        nonlocal tool_count
        tool_count += 1

        if scenario == "failed":
            raise FileNotFoundError("模拟文件不存在")

        return json.dumps({
            "ok": True,
            "data": {
                "source": "book.pdf",
                "pages": [{
                    "page": 5,
                    "text": "审计测试正文",
                    "citation": "[source=book.pdf, page=5]",
                }],
                "truncated": False,
            },
        }, ensure_ascii=False)

    tool = SimpleNamespace(
        name="read_pdf_pages",
        invoke=fake_tool,
    )
    model_count = 0

    def tool_request(call_id):
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "read_pdf_pages",
                "args": {
                    "relative_path": "book.pdf",
                    "start_page": 5,
                    "end_page": 5,
                },
                "id": call_id,
                "type": "tool_call",
            }],
        )

    def fake_model(llm, messages, *, stage):
        nonlocal model_count
        model_count += 1

        if model_count == 1:
            return tool_request("audit-001")

        if scenario == "duplicate" and model_count == 2:
            return tool_request("audit-002")

        if scenario == "recovered" and model_count == 2:
            return AIMessage(content="")

        # failed故意模拟：工具失败，模型仍然声称成功。
        return AIMessage(content="已读取测试正文。[source=book.pdf, page=5]")

    monkeypatch.setattr(main_module, "invoke_llm_response", fake_model)

    trace = {} if audit_enabled else None
    result = main_module._invoke_interactive_turn(
        request_messages=[],
        user_input="请实际调用 read_pdf_pages，读取book.pdf物理第5页。",
        plain_llm=object(),
        tool_llm=object(),
        retrieval_tool=None,
        extra_tools=(tool,),
        audit_trace=trace,
    )

    assert tool_count == 1
    assert model_count == (
        3 if scenario in {"recovered", "duplicate"} else 2
    )

    if scenario == "failed":
        assert "不能确认" in result
    else:
        assert result == "已读取测试正文。[source=book.pdf, page=5]"

    if not audit_enabled:
        return

    assert trace["finished"] is True
    assert trace["final_answer"] == result
    assert len(trace["model_responses"]) == model_count
    assert trace["tool_results"][0]["executed"] is True
    assert trace["tool_results"][0]["args"]["start_page"] == 5

    payload = json.loads(trace["tool_results"][0]["result"])

    if scenario == "failed":
        assert payload["ok"] is False
        assert "已读取" in trace["model_responses"][-1]["content"]
        assert "已读取" not in trace["final_answer"]

        raw_report = check_pdf_answer(
            trace,
            answer=trace["model_responses"][-1]["content"],
        )
        final_report = check_pdf_answer(trace)

        # 原始模型错引了没有成功读取的页面。
        assert raw_report["pdf_read_succeeded"] is False
        assert raw_report["page_citation_supported"] is False

        # 程序最终输出没有引用；None表示不适用/无法检查，不是判为正确。
        assert final_report["pdf_read_succeeded"] is False
        assert final_report["page_citation_supported"] is None

        print(json.dumps({
            "raw_model": raw_report,
            "final_output": final_report,
        }, ensure_ascii=False, indent=2))

    else:
        assert payload["data"]["pages"][0]["text"] == "审计测试正文"

    if scenario == "duplicate":
        rejected = trace["tool_results"][1]
        assert rejected["executed"] is False
        assert json.loads(rejected["result"])["error"]["code"] == (
            "duplicate_call"
        )

    if scenario == "recovered":
        assert trace["model_responses"][1]["content"] == ""
        assert trace["model_responses"][-1]["stage"] == (
            "交互回答-空响应恢复"
        )

    # 只打印合成测试资料；真实轨迹可能包含私有正文，不能随意公开。
    print(json.dumps(trace, ensure_ascii=False, indent=2))


def test_audit_records_early_rejection(monkeypatch):
    def unexpected_model(*args, **kwargs):
        pytest.fail("未绑定工具时不应调用模型")

    monkeypatch.setattr(main_module, "invoke_llm_response", unexpected_model)
    trace = {}

    result = main_module._invoke_interactive_turn(
        request_messages=[],
        user_input="请实际调用 unknown_probe，执行请求。",
        plain_llm=object(),
        tool_llm=object(),
        retrieval_tool=None,
        audit_trace=trace,
    )

    assert "未绑定工具" in result
    assert trace["finished"] is True
    assert trace["final_answer"] == result
    assert trace["model_responses"] == []
    assert trace["tool_results"] == []