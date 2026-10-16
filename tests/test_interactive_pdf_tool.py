"""PDF读取工具与交互接入测试。

通过假PDF解析器和假模型验证参数、结果回传及执行校验。
底层路径和分页边界继续由test_pdf_material_reader.py覆盖。
不访问在线服务，不验证真实PDF的提取质量。
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import ai_agent.main as main_module
import ai_agent.pdf_materials as pdf_module
from ai_agent.execution_guard import tool_result_succeeded
from ai_agent.tools import build_read_pdf_pages_tool


def prepare_pdf(monkeypatch, tmp_path, text):
    (tmp_path / "book.pdf").write_bytes(b"mock PDF")
    reader = SimpleNamespace(
        is_encrypted=False,
        pages=[SimpleNamespace(extract_text=lambda: text)],
    )
    monkeypatch.setattr(
        pdf_module,
        "PdfReader",
        lambda *args, **kwargs: reader,
    )


@pytest.mark.parametrize("invalid_page", [True, "1", 1.5])
def test_pdf_tool_rejects_non_integer_page(tmp_path, invalid_page):
    tool = build_read_pdf_pages_tool(str(tmp_path))

    with pytest.raises(ValueError):
        tool.invoke({
            "relative_path": "book.pdf",
            "start_page": invalid_page,
            "end_page": 1,
        })


def test_pdf_tool_returns_structured_page_citation(monkeypatch, tmp_path):
    prepare_pdf(monkeypatch, tmp_path, "PDF测试正文")
    tool = build_read_pdf_pages_tool(str(tmp_path))

    # 根目录不出现在模型可生成的参数中。
    assert "root_dir" not in tool.args

    result = tool.invoke({
        "relative_path": "book.pdf",
        "start_page": 1,
        "end_page": 1,
    })
    payload = json.loads(result)

    assert tool_result_succeeded("read_pdf_pages", result)
    assert payload["data"]["page_numbering"] == "physical_1_based"
    assert payload["data"]["pages"][0]["citation"] == (
        "[source=book.pdf, page=1]"
    )


@pytest.mark.parametrize(
    "scenario",
    ["text", "no_text", "partial", "missing"],
)
def test_pdf_result_reaches_interactive_model(
    monkeypatch, tmp_path, scenario,
):
    prepare_pdf(
        monkeypatch,
        tmp_path,
        "" if scenario == "no_text" else "ABCDEFG",
    )
    if scenario == "partial":
        monkeypatch.setattr(pdf_module, "MAX_READ_PDF_TEXT_CHARS", 4)

    settings = replace(
        main_module.get_settings(),
        interactive_max_retrieval_calls=2,
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    tool = build_read_pdf_pages_tool(str(tmp_path))
    calls = 0

    def fake_invoke(llm, messages, *, stage):
        nonlocal calls
        calls += 1

        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_pdf_pages",
                    "args": {
                        "relative_path": (
                            "missing.pdf"
                            if scenario == "missing"
                            else "book.pdf"
                        ),
                        "start_page": 1,
                        "end_page": 1,
                    },
                    "id": "pdf-001",
                    "type": "tool_call",
                }],
            )

        results = [
            message for message in messages
            if isinstance(message, ToolMessage)
        ]
        assert len(results) == 1
        assert results[0].tool_call_id == "pdf-001"
        payload = json.loads(results[0].content)

        if scenario == "missing":
            assert payload["ok"] is False
            assert payload["error"]["code"] == "FileNotFoundError"
            # 故意模拟失败后谎报成功，验证最终执行校验会拦截。
            return AIMessage(content="PDF已经成功读取。")

        assert payload["ok"] is True
        page = payload["data"]["pages"][0]

        if scenario == "no_text":
            assert page["text_status"] == "no_text"
            assert page["text"] == ""
            return AIMessage(content="未提取到文字，不能判断页面空白。")

        if scenario == "partial":
            assert page["text"] == "ABCD"
            assert page["partial"] is True
            assert payload["data"]["truncated"] is True
            return AIMessage(content="仅返回部分文字：ABCD。")

        assert page["text"] == "ABCDEFG"
        return AIMessage(content="实际文字：ABCDEFG。[source=book.pdf, page=1]")

    monkeypatch.setattr(main_module, "invoke_llm_response", fake_invoke)

    result = main_module._invoke_interactive_turn(
        request_messages=[],
        user_input="请实际调用 read_pdf_pages，读取book.pdf物理第1页。",
        plain_llm=object(),
        tool_llm=object(),
        retrieval_tool=None,
        extra_tools=(tool,),
    )

    assert calls == 2

    if scenario == "missing":
        assert "不能确认" in result
        assert result != "PDF已经成功读取。"
    elif scenario == "no_text":
        assert "不能判断页面空白" in result
    elif scenario == "partial":
        assert "部分文字" in result
    else:
        assert "[source=book.pdf, page=1]" in result