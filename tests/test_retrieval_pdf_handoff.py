"""检索到PDF读取的位置传递与观察顺序测试。

使用合成审计轨迹，不调用模型、Embedding或真实文件。
通过仅说明检查规则符合这些固定案例，不代表在线Agent已具备该能力。
"""

import pytest

from ai_agent.evaluation.interactive_checks import (
    check_retrieval_to_pdf_handoff,
)


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("normal", True),
        ("wrong_page", False),
        ("wrong_file", False),
        ("read_first", False),
        ("same_response", False),
        ("not_executed", False),
        ("missing_call_id", None),
        ("unfinished", None),
        ("no_hit", False),
    ],
)
def test_retrieval_pdf_handoff(scenario, expected):
    header = "[source=book.pdf, page=20, chunk=8, role=hit]"

    if scenario == "wrong_page":
        header = "[source=book.pdf, page=21, chunk=8, role=hit]"
    elif scenario == "wrong_file":
        header = "[source=other.pdf, page=20, chunk=8, role=hit]"
    elif scenario == "no_hit":
        header = "没有达到相关性阈值的新增资料。"

    retrieval_call = {
        "id": "retrieve-001",
        "name": "retrieve_learning_context",
    }
    read_call = {
        "id": "read-001",
        "name": "read_pdf_pages",
    }

    responses = [
        {"tool_calls": [retrieval_call]},
        {"tool_calls": [read_call]},
    ]
    records = [
        {
            "tool_call_id": "retrieve-001",
            "tool_name": "retrieve_learning_context",
            "executed": True,
            "result": header + "\n合成资料正文",
        },
        {
            "tool_call_id": "read-001",
            "tool_name": "read_pdf_pages",
            "executed": True,
            "args": {
                "relative_path": "book.pdf",
                "start_page": 20,
                "end_page": 20,
            },
        },
    ]

    if scenario == "read_first":
        responses.reverse()
        records.reverse()
    elif scenario == "same_response":
        # 程序即使依次执行，两组参数也都是模型事先生成的。
        responses = [{"tool_calls": [retrieval_call, read_call]}]
    elif scenario == "not_executed":
        records[0]["executed"] = False
    elif scenario == "missing_call_id":
        records[1].pop("tool_call_id")

    trace = {
        "schema_version": 1,
        "finished": scenario != "unfinished",
        "model_responses": responses,
        "tool_results": records,
    }

    result = check_retrieval_to_pdf_handoff(
        trace,
        source="book.pdf",
        page=20,
    )
    assert result is expected

    print(f"{scenario}: handoff={result}")


@pytest.mark.parametrize("page", [0, -1, True, "20"])
def test_handoff_rejects_invalid_page(page):
    """布尔值虽然属于Python整数体系，但不能作为页码。"""
    with pytest.raises(ValueError, match="page"):
        check_retrieval_to_pdf_handoff(
            {"schema_version": 1, "finished": True},
            source="book.pdf",
            page=page,
        )