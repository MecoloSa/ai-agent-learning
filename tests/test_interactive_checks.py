"""交互证据检查器的固定正反例测试。

把精简案例转换为审计结构，验证检查器与人工预期一致。
不运行真实Agent，不调用模型；通过不代表真实模型质量提高。
"""

import json
from pathlib import Path

import pytest

from ai_agent.evaluation.interactive_checks import check_pdf_answer


CASE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "evaluation"
    / "interactive_pdf_checks.jsonl"
)

CASES = [
    json.loads(line)
    for line in CASE_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[case["case_id"] for case in CASES],
)
def test_fixed_pdf_evidence_cases(case):
    tool_result = case["result"]
    if not isinstance(tool_result, str):
        tool_result = json.dumps(tool_result, ensure_ascii=False)

    # 这里只是构造检查器输入，不伪称执行过真实工具。
    trace = {
        "schema_version": 1,
        "finished": case.get("finished", True),
        "final_answer": case["answer"],
        "tool_results": [{
            "tool_name": "read_pdf_pages",
            "executed": True,
            "result": tool_result,
        }],
    }

    report = check_pdf_answer(trace)

    for dimension, expected in case["expected"].items():
        assert report[dimension] == expected, dimension

    if "note_contains" in case:
        assert any(
            case["note_contains"] in note
            for note in report["review_notes"]
        )

    print(json.dumps({
        "case_id": case["case_id"],
        "report": report,
    }, ensure_ascii=False, indent=2))