"""交互配置快照与token摘要的离线测试。

使用合成用量与配置，不创建客户端、不访问在线服务。
"""

from types import SimpleNamespace

import pytest

from ai_agent.evaluation.interactive_metrics import (
    EVAL_CONFIG_KEYS,
    build_eval_config_snapshot,
    summarize_chat_usage,
)


def test_usage_sums_all_recorded_responses():
    trace = {
        "model_responses": [
            {"usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
            }},
            {"usage": {
                "input_tokens": 200,
                "output_tokens": 20,
                "total_tokens": 220,
            }},
        ],
    }

    report = summarize_chat_usage(trace)

    assert report["input_tokens"] == 300
    assert report["output_tokens"] == 30
    assert report["total_tokens"] == 330
    assert report["usage_complete"] is True
    assert report["responses_with_complete_usage"] == 2


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"input_tokens": True},
        {"input_tokens": -1},
        {"input_tokens": "100"},
    ],
)
def test_unknown_or_invalid_usage_is_not_zero(usage):
    report = summarize_chat_usage({
        "model_responses": [{"usage": usage}],
    })

    assert report["input_tokens"] is None
    assert report["usage_complete"] is False


def test_missing_one_field_does_not_erase_other_known_fields():
    report = summarize_chat_usage({
        "model_responses": [{
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }],
    })

    assert report["input_tokens"] == 100
    assert report["output_tokens"] == 10
    assert report["total_tokens"] is None
    assert report["usage_complete"] is False


def test_no_responses_means_unknown_usage():
    report = summarize_chat_usage({"model_responses": []})

    assert report["recorded_response_count"] == 0
    assert report["total_tokens"] is None
    assert report["usage_complete"] is False


def test_config_snapshot_excludes_secrets():
    settings = SimpleNamespace(
        **{key: 1 for key in EVAL_CONFIG_KEYS},
        chat_api_key="synthetic-secret",
        embedding_api_key="another-synthetic-secret",
    )

    snapshot = build_eval_config_snapshot(settings)

    assert set(snapshot["settings"]) == set(EVAL_CONFIG_KEYS)
    assert "chat_api_key" not in snapshot["settings"]
    assert "embedding_api_key" not in snapshot["settings"]
    assert "synthetic-secret" not in repr(snapshot)

def make_comparison_record(*, tool_limit=2, tokens=100):
    """构造比较函数输入；所有内容都是合成数据。"""
    settings = {key: 1 for key in EVAL_CONFIG_KEYS}
    settings["interactive_max_retrieval_calls"] = tool_limit

    return {
        "case": {"case_id": "synthetic"},
        "tool_profile": "pdf_and_retrieval",
        "system_prompt": "合成提示词",
        "prompt_version": "synthetic-v1",
        "model_name": "synthetic-model",
        "material_dir": "synthetic-materials",
        "eval_config": {"schema_version": 1, "settings": settings},
        "trace": {"model_responses": [{
            "usage": {
                "input_tokens": tokens,
                "output_tokens": 10,
                "total_tokens": tokens + 10,
            },
        }]},
        "run_status": "returned",
        "requested_pdf_read_succeeded": True,
        "retrieval_to_pdf_handoff": True,
        "raw_model_checks": {"page_citation_supported": True},
        "final_output_checks": {"page_citation_supported": True},
        "latency_ms": 1000,
    }


def test_compare_single_budget_change():
    from ai_agent.evaluation.interactive_metrics import compare_interactive_records

    before = make_comparison_record(tool_limit=2, tokens=100)
    after = make_comparison_record(tool_limit=3, tokens=120)
    report = compare_interactive_records(before, after)

    assert report["config_changes"] == {
        "interactive_max_retrieval_calls": {"before": 2, "after": 3},
    }
    assert report["after_minus_before"]["input_tokens"] == 20
    assert report["before"]["handoff"] is True
    assert report["after"]["handoff"] is True


def test_compare_rejects_multiple_budget_changes():
    from ai_agent.evaluation.interactive_metrics import compare_interactive_records

    before = make_comparison_record(tool_limit=2)
    after = make_comparison_record(tool_limit=3)
    after["eval_config"]["settings"]["interactive_retrieval_max_context_chars"] = 16000

    with pytest.raises(ValueError, match="单个预算"):
        compare_interactive_records(before, after)


def test_compare_rejects_different_prompt():
    from ai_agent.evaluation.interactive_metrics import compare_interactive_records

    before = make_comparison_record()
    after = make_comparison_record()
    after["system_prompt"] = "另一个提示词"

    with pytest.raises(ValueError, match="system_prompt"):
        compare_interactive_records(before, after)


def test_compare_preserves_unknown_usage():
    from ai_agent.evaluation.interactive_metrics import compare_interactive_records

    before = make_comparison_record()
    after = make_comparison_record()
    after["trace"]["model_responses"][0]["usage"] = None

    report = compare_interactive_records(before, after)

    assert report["after"]["usage_complete"] is False
    assert report["after_minus_before"]["total_tokens"] is None

def test_tool_outputs_count_only_executed_records():
    from ai_agent.evaluation.interactive_metrics import summarize_tool_outputs

    report = summarize_tool_outputs({
        "tool_results": [
            {"tool_name": "retrieve_learning_context",
             "executed": True, "result": "abc"},
            {"tool_name": "retrieve_learning_context",
             "executed": True, "result": "de"},
            {"tool_name": "retrieve_learning_context",
             "executed": False, "result": "不应计入"},
        ],
    })

    assert report["retrieve_learning_context"] == {
        "executed_call_count": 2,
        "result_chars": 5,
        "max_result_chars": 3,
    }


def test_missing_tool_output_is_unknown():
    from ai_agent.evaluation.interactive_metrics import summarize_tool_outputs

    report = summarize_tool_outputs({
        "tool_results": [{
            "tool_name": "read_pdf_pages",
            "executed": True,
        }],
    })

    assert report["read_pdf_pages"]["executed_call_count"] == 1
    assert report["read_pdf_pages"]["result_chars"] is None


def test_no_tool_records_is_supported():
    from ai_agent.evaluation.interactive_metrics import summarize_tool_outputs

    # 兼容只验证模型用量的旧合成案例。
    assert summarize_tool_outputs({}) == {}