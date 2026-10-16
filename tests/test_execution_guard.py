"""通用执行校验测试。

验证明确请求识别、不同返回结果的成功判定，以及未绑定工具的拒绝。
不访问在线模型，也不读写真实学习状态。
"""

import pytest

import ai_agent.main as main_module
from ai_agent.execution_guard import (
    resolve_execution_requirement,
    supports_execution_check,
    tool_result_succeeded,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "请实际调用 read_material_lines，读取第2到3行。",
            "read_material_lines",
        ),
        (
            "重新调用 read_pdf_pages，读取第5页。",
            "read_pdf_pages",
        ),
        ("请解释 read_material_lines 的作用。", None),
        ("不要实际调用 read_material_lines。", None),
        ("只复述当前任务状态。", None),
    ],
)
def test_resolve_explicit_execution_requirement(text, expected):
    requirement = resolve_execution_requirement(text)
    actual = requirement.tool_name if requirement else None
    assert actual == expected


@pytest.mark.parametrize(
    "result, expected",
    [
        ('{"ok": true, "data": {}}', True),
        ('{"ok": false, "error": {"code": "missing"}}', False),
        ('{"ok": "true", "data": {}}', False),
        ('{"ok": true, "data": null}', False),
        ('["not", "an", "object"]', False),
        ("工具已成功读取", False),
    ],
)
def test_line_read_success_requires_structured_result(result, expected):
    assert tool_result_succeeded(
        "read_material_lines", result,
    ) is expected


def test_unregistered_checker_does_not_guess_success():
    # 非空文本或伪造的 JSON，都不能自动成为其他工具的成功证明。
    assert not supports_execution_check("retrieve_learning_context")
    assert not tool_result_succeeded(
        "retrieve_learning_context",
        '{"ok": true, "data": {}}',
    )


def test_unbound_required_tool_is_rejected_before_model(monkeypatch):
    def unexpected_model_call(*args, **kwargs):
        pytest.fail("未绑定的指定工具不应交给模型模拟执行")

    monkeypatch.setattr(
        main_module, "invoke_llm_response", unexpected_model_call,
    )

    result = main_module._invoke_interactive_turn(
        request_messages=[],
        user_input="请实际调用 read_pdf_pages，读取第5页。",
        plain_llm=object(),
        tool_llm=object(),
        retrieval_tool=None,
        read_tool=None,
    )

    assert "未绑定工具 read_pdf_pages" in result