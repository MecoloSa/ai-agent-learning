"""在线交互评测入口的离线测试。

使用假执行器验证结果配对与案例配置，不创建真实模型客户端。
通过不代表在线模型行为已经验证。
"""

import json
from types import SimpleNamespace

import pytest

from ai_agent.evaluation.interactive_runner import (
    OnlineCase,
    build_question,
    load_cases,
    run_case,
    validate_materials,
)


def test_run_case_separates_raw_and_final_answers():
    case = OnlineCase(
        case_id="missing",
        relative_path="missing.pdf",
        page=1,
        expected_pdf_success=False,
    )

    def fake_execute(**kwargs):
        kwargs["audit_trace"].update({
            "schema_version": 1,
            "finished": True,
            "final_answer": "读取失败，不能确认正文。",
            "model_responses": [{
                "content": "已经读取。[source=missing.pdf, page=1]",
                "tool_calls": [],
                "usage": None,
            }],
            "tool_results": [{
                "tool_name": "read_pdf_pages",
                "executed": True,
                # 模拟轨迹必须包含目标位置，才能验证“指定读取”的结果。
                "args": {
                    "relative_path": "missing.pdf",
                    "start_page": 1,
                    "end_page": 1,
                },
                "result": json.dumps({
                    "ok": False,
                    "error": {"code": "FileNotFoundError"},
                }),
            }],
        })
        return "读取失败，不能确认正文。"

    record = run_case(
        case,
        plain_llm=object(),
        tool_llm=object(),
        pdf_tool=SimpleNamespace(name="read_pdf_pages"),
        execute=fake_execute,
    )

    assert record["run_status"] == "returned"
    assert record["requested_pdf_attempted"] is True
    assert record["pdf_success_matches_expected"] is True
    assert record["raw_model_checks"]["page_citation_supported"] is False
    assert record["final_output_checks"]["page_citation_supported"] is None
    # 已记录一条响应，但提供商用量缺失：
    # 响应数已知，累计token仍然未知。
    assert record["chat_usage"]["recorded_response_count"] == 1
    assert record["chat_usage"]["usage_complete"] is False
    assert record["chat_usage"]["total_tokens"] is None
    assert "interactive_max_retrieval_calls" in (
        record["eval_config"]["settings"]
    )


def test_run_case_preserves_partial_trace_on_error():
    case = OnlineCase(
        case_id="error",
        relative_path="book.pdf",
        page=5,
        expected_pdf_success=True,
    )

    def fake_execute(**kwargs):
        kwargs["audit_trace"]["finished"] = False
        raise RuntimeError("模拟调用失败")

    record = run_case(
        case,
        plain_llm=object(),
        tool_llm=object(),
        pdf_tool=object(),
        execute=fake_execute,
    )

    assert record["run_status"] == "error"
    assert record["error_type"] == "RuntimeError"
    assert record["raw_model_checks"] is None
    assert record["final_output_checks"] is None
    assert record["pdf_success_matches_expected"] is None


def test_preflight_rejects_wrong_file_expectation(tmp_path):
    case = OnlineCase(
        case_id="missing_but_expected_present",
        relative_path="book.pdf",
        page=5,
        expected_pdf_success=True,
    )

    with pytest.raises(ValueError, match="存在情况"):
        validate_materials(tmp_path, [case])


def test_load_and_build_question(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({
        "case_id": "case-001",
        "relative_path": "book.pdf",
        "page": 5,
        "expected_pdf_success": True,
        "request_style": "autonomous",
    }), encoding="utf-8")

    case = load_cases(path)[0]
    assert "物理第5页" in build_question(case)
    assert not build_question(case).startswith("请实际调用")

def test_pdf_prompt_requires_verbatim_tool_citation():
    """评测入口必须实际包含公共规则，而不只是导入变量。"""
    from ai_agent.evaluation.interactive_runner import (
        PROMPT_VERSION,
        SYSTEM_PROMPT,
    )
    from ai_agent.runtime.prompt_rules import PDF_CITATION_RULES

    assert PROMPT_VERSION == "isolated-pdf-v3"
    assert PDF_CITATION_RULES in SYSTEM_PROMPT

    # 保留关键约定的检查，避免公共字符串被误删后测试仍通过。
    assert "逐字复制对应页面返回的citation字段值" in SYSTEM_PROMPT
    assert "[source=book.pdf, page=5]" in SYSTEM_PROMPT
    assert "不要添加citation:前缀" in SYSTEM_PROMPT
    assert "不要把page改成physical_page" in SYSTEM_PROMPT

@pytest.mark.parametrize("with_file_list", [False, True])
def test_run_case_optional_file_list_tools(with_file_list):
    """验证工具分发与提示词接入，不模拟在线模型的选择能力。"""
    from ai_agent.runtime.prompt_rules import MATERIAL_ACCESS_RULES

    captured = {}
    pdf_tool = SimpleNamespace(name="read_pdf_pages")
    list_tool = (
        SimpleNamespace(name="list_material_files")
        if with_file_list
        else None
    )

    def fake_execute(**kwargs):
        captured["tools"] = [
            tool.name for tool in kwargs["extra_tools"]
        ]
        captured["prompt"] = kwargs["request_messages"][0].content

        kwargs["audit_trace"].update({
            "schema_version": 1,
            "finished": True,
            "final_answer": "读取失败。",
            "model_responses": [],
            "tool_results": [{
                "tool_name": "read_pdf_pages",
                "executed": True,
                # 与下方OnlineCase一致，明确模拟指定文件、指定页码的读取。
                "args": {
                    "relative_path": "missing.pdf",
                    "start_page": 1,
                    "end_page": 1,
                },
                "result": json.dumps({
                    "ok": False,
                    "error": {"code": "FileNotFoundError"},
                }),
            }],
        })

    record = run_case(
        OnlineCase(
            case_id="optional-list",
            relative_path="missing.pdf",
            page=1,
            expected_pdf_success=False,
        ),
        plain_llm=object(),
        tool_llm=object(),
        pdf_tool=pdf_tool,
        file_list_tool=list_tool,
        execute=fake_execute,
    )

    expected_tools = ["read_pdf_pages"]
    if with_file_list:
        expected_tools.append("list_material_files")

    assert captured["tools"] == expected_tools
    assert (MATERIAL_ACCESS_RULES in captured["prompt"]) == with_file_list
    assert record["system_prompt"] == captured["prompt"]
    assert record["tool_profile"] == (
        "pdf_and_file_list" if with_file_list else "pdf_only"
    )
    assert record["first_executed_tool"] == "read_pdf_pages"

    # 区分“确实尝试”“实际失败”和“失败符合预期”。
    assert record["requested_pdf_attempted"] is True
    assert record["requested_pdf_read_succeeded"] is False
    assert record["pdf_success_matches_expected"] is True

@pytest.mark.parametrize(
    "actual_source,actual_page,tool_ok,expected_success,expected_match",
    [
        ("book.pdf", 5, True, True, True),       # 正确目标成功
        ("book.pdf", 5, False, False, True),     # 正确目标失败，符合负例
        ("book.pdf", 5, False, True, False),     # 预期成功却失败
        ("other.pdf", 5, True, True, None),      # 错误文件不能算通过
        ("book.pdf", 6, True, True, None),       # 错误页码不能算通过
        (None, None, False, False, None),       # 没执行不能算预期失败
    ],
)
def test_read_expectation_is_scoped_to_requested_target(
    actual_source,
    actual_page,
    tool_ok,
    expected_success,
    expected_match,
):
    """用可控轨迹验证目标匹配，不调用模型或读取真实PDF。"""
    def fake_execute(**kwargs):
        records = []

        if actual_source is not None:
            payload = (
                {
                    "ok": True,
                    "data": {
                        "source": actual_source,
                        "pages": [{
                            "page": actual_page,
                            "text": "合成测试正文",
                        }],
                    },
                }
                if tool_ok
                else {
                    "ok": False,
                    "error": {"code": "FileNotFoundError"},
                }
            )

            records.append({
                "tool_name": "read_pdf_pages",
                "executed": True,
                "args": {
                    "relative_path": actual_source,
                    "start_page": actual_page,
                    "end_page": actual_page,
                },
                "result": json.dumps(payload),
            })

        kwargs["audit_trace"].update({
            "schema_version": 1,
            "finished": True,
            "final_answer": "合成测试回答",
            "model_responses": [],
            "tool_results": records,
        })

    record = run_case(
        OnlineCase(
            case_id="target-check",
            relative_path="book.pdf",
            page=5,
            expected_pdf_success=expected_success,
        ),
        plain_llm=object(),
        tool_llm=object(),
        pdf_tool=SimpleNamespace(name="read_pdf_pages"),
        execute=fake_execute,
    )

    target_attempted = actual_source == "book.pdf" and actual_page == 5

    assert record["requested_pdf_attempted"] is target_attempted
    assert record["requested_pdf_read_succeeded"] is (
        tool_ok if target_attempted else None
    )
    assert record["pdf_success_matches_expected"] is expected_match

def test_locate_question_does_not_leak_label_page():
    """改变隐藏标签，不应改变模型收到的问题。"""
    first = OnlineCase(
        case_id="locate",
        relative_path="book.pdf",
        page=5,
        expected_pdf_success=True,
        request_style="locate",
        topic="查找某段介绍",
    )
    second = first.model_copy(update={"page": 99})

    assert build_question(first) == build_question(second)
    assert "查找某段介绍" in build_question(first)


def test_locate_case_requires_topic():
    """在付费执行前拒绝缺少检索目标的案例。"""
    with pytest.raises(ValueError, match="topic"):
        OnlineCase(
            case_id="invalid",
            relative_path="book.pdf",
            page=5,
            expected_pdf_success=True,
            request_style="locate",
        )


def test_locate_runner_wires_retrieval_and_handoff(monkeypatch):
    """只验证入口接线；链路判定本身由独立固定案例测试。"""
    import ai_agent.evaluation.interactive_runner as runner_module

    retrieval = SimpleNamespace(name="retrieve_learning_context")
    captured = {}

    def fake_execute(**kwargs):
        captured["retrieval"] = kwargs["retrieval_tool"]
        captured["question"] = kwargs["user_input"]
        kwargs["audit_trace"].update({
            "schema_version": 1,
            "finished": True,
            "final_answer": "合成回答",
            "model_responses": [],
            "tool_results": [],
        })

    def fake_handoff(trace, *, source, page):
        captured["target"] = (source, page)
        return True

    # 用替身隔离“是否调用检查器”与“检查器如何判断”两个测试责任。
    monkeypatch.setattr(
        runner_module,
        "check_retrieval_to_pdf_handoff",
        fake_handoff,
    )

    record = run_case(
        OnlineCase(
            case_id="locate",
            relative_path="book.pdf",
            page=5,
            expected_pdf_success=True,
            request_style="locate",
            topic="查找某段介绍",
        ),
        plain_llm=object(),
        tool_llm=object(),
        pdf_tool=SimpleNamespace(name="read_pdf_pages"),
        retrieval_tool=retrieval,
        execute=fake_execute,
    )

    assert captured["retrieval"] is retrieval
    assert captured["target"] == ("book.pdf", 5)
    assert "物理第5页" not in captured["question"]
    assert record["tool_profile"] == "pdf_and_retrieval"
    assert record["retrieval_to_pdf_handoff"] is True

    # 接线测试里的True来自替身，不应掩盖实际没有读取记录。
    assert record["requested_pdf_attempted"] is False
    assert record["requested_pdf_read_succeeded"] is None
    # 接线测试没有模拟模型响应；没有记录不等于实际用量为0。
    assert record["chat_usage"]["recorded_response_count"] == 0
    assert record["chat_usage"]["usage_complete"] is False
    assert record["chat_usage"]["total_tokens"] is None
    assert "interactive_max_retrieval_calls" in (
        record["eval_config"]["settings"]
    )
