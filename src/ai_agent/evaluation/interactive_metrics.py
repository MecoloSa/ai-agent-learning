"""交互实验的配置快照与聊天token摘要。

只读取已有配置和审计轨迹，不调用模型、不改变执行。
统计的是已记录响应的提供商用量，不是完整账单。
"""

from __future__ import annotations


# 使用允许列表，禁止直接序列化Settings：
# Settings同时包含密钥，不能为了复现实验而把它们保存下来。
EVAL_CONFIG_KEYS = (
    "chat_model",
    "embedding_model",
    "llm_temperature",
    "request_timeout",
    "llm_retry_attempts",
    "interactive_max_retrieval_calls",
    "interactive_max_queries_per_tool_call",
    "interactive_retrieval_max_context_chars",
    "rag_retrieval_k",
    "rag_min_relevance",
    "rag_relative_margin",
    "rag_min_hits_per_query",
    "rag_chunk_size",
    "rag_chunk_overlap",
    "rag_rerank_enabled",
    "rag_rerank_backend",
    "rag_rerank_chat_model",
    "rag_rerank_model",
    "rag_rerank_candidate_limit",
    "rag_rerank_timeout_s",
    "rag_retrieval_strategy",
    "rag_hybrid_min_rerank_score",
)

def build_eval_config_snapshot(settings) -> dict:
    """记录当前实验相关配置；不是完整环境或索引版本快照。"""
    return {
        "schema_version": 1,
        "settings": {
            key: getattr(settings, key)
            for key in EVAL_CONFIG_KEYS
        },
    }


def summarize_chat_usage(trace: dict) -> dict:
    """按字段累计已记录响应的token，缺失或非法值保持未知。"""
    responses = trace.get("model_responses", [])
    fields = ("input_tokens", "output_tokens", "total_tokens")

    usages = [
        response.get("usage")
        if isinstance(response.get("usage"), dict)
        else {}
        for response in responses
    ]

    def valid_token_count(value) -> bool:
        # bool也是int的子类，因此不能使用isinstance(value, int)。
        return type(value) is int and value >= 0

    complete_count = sum(
        all(valid_token_count(usage.get(field)) for field in fields)
        for usage in usages
    )

    totals = {}
    for field in fields:
        values = [usage.get(field) for usage in usages]

        # 没有响应或任意响应缺少该字段，都不能宣称知道完整累计值。
        # 不用0替代未知，也不从其他字段推算提供商没有报告的值。
        totals[field] = (
            sum(values)
            if values and all(valid_token_count(value) for value in values)
            else None
        )

    # 动态评估：先检查usage_complete，再比较累计token；
    # 不能把未知值当作更省，也不能只比较最后一次请求。
    return {
        "schema_version": 1,
        "recorded_response_count": len(responses),
        "responses_with_complete_usage": complete_count,
        "usage_complete": bool(responses) and complete_count == len(responses),
        **totals,
    }

def summarize_tool_outputs(trace: dict) -> dict:
    """按工具统计实际执行次数与返回字符数，不判断内容质量。"""
    groups = {}

    for record in trace.get("tool_results", []):
        if record.get("executed") is not True:
            # 被预算或重复保护拒绝的调用，不算实际执行。
            continue

        name = record.get("tool_name")
        if not isinstance(name, str) or not name:
            continue

        groups.setdefault(name, []).append(record.get("result"))

    report = {}
    for name, results in groups.items():
        known = all(isinstance(result, str) for result in results)
        lengths = [len(result) for result in results] if known else []

        # 缺失返回文本时保留未知，不能当作返回0字符。
        # 动态评估：比较预算变化是否真的改变返回规模，再看token与质量。
        report[name] = {
            "executed_call_count": len(results),
            "result_chars": sum(lengths) if known else None,
            "max_result_chars": max(lengths) if known else None,
        }

    return report

def compare_interactive_records(before: dict, after: dict) -> dict:
    """比较同一案例的单变量预算实验，不自动判断哪份回答更好。

    只检查当前结果中已保存的条件。
    资料内容、索引版本、服务端模型变化仍需实验者控制。
    """
    fixed_fields = (
        "case",
        "tool_profile",
        "system_prompt",
        "prompt_version",
        "model_name",
        "material_dir",
    )
    for field in fixed_fields:
        if before.get(field) is None or after.get(field) is None:
            raise ValueError(f"缺少对照条件：{field}")
        if before[field] != after[field]:
            raise ValueError(f"对照条件不同：{field}")

    configs = []
    for record in (before, after):
        snapshot = record.get("eval_config")
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
            raise ValueError("需要带有配置快照的新实验结果")

        settings = snapshot.get("settings")
        if not isinstance(settings, dict) or set(settings) != set(EVAL_CONFIG_KEYS):
            raise ValueError("配置快照字段不完整或版本不一致")
        configs.append(settings)

    left, right = configs
    changes = {
        key: {"before": left[key], "after": right[key]}
        for key in EVAL_CONFIG_KEYS
        if left[key] != right[key]
    }
    allowed_changes = {
        "interactive_max_retrieval_calls",
        "interactive_retrieval_max_context_chars",
    }

    # 允许相同配置重复运行；若调参，每次最多改变一个预算。
    if len(changes) > 1 or not set(changes).issubset(allowed_changes):
        raise ValueError("本比较仅支持同配置重复或单个预算参数变化")

    def summarize(record):
        # 从原始轨迹重新计算，避免依赖手工修改过的摘要字段。
        usage = summarize_chat_usage(record["trace"])
        raw = record.get("raw_model_checks") or {}
        final = record.get("final_output_checks") or {}

        return {
            "run_status": record.get("run_status"),
            "target_read_succeeded": record.get("requested_pdf_read_succeeded"),
            "handoff": record.get("retrieval_to_pdf_handoff"),
            "raw_citation_supported": raw.get("page_citation_supported"),
            "final_citation_supported": final.get("page_citation_supported"),
            "model_response_count": usage["recorded_response_count"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "usage_complete": usage["usage_complete"],
            "tool_outputs": summarize_tool_outputs(record["trace"]),
            "latency_ms": record.get("latency_ms"),
        }

    old = summarize(before)
    new = summarize(after)

    def delta(field):
        values = (old[field], new[field])
        # 缺失值不能转换成0，否则可能把“未知”误认为节省。
        if not all(type(value) is int and value >= 0 for value in values):
            return None
        return new[field] - old[field]

    # 动态评估：差值为负表示减少，不表示质量更好；
    # 先核对质量维度，再解释成本与耗时变化。
    return {
        "case_id": before["case"]["case_id"],
        "config_changes": changes,
        "before": old,
        "after": new,
        "after_minus_before": {
            field: delta(field)
            for field in ("input_tokens", "output_tokens", "total_tokens", "latency_ms")
        },
        "note": (
            "单次配对仅供观察；未自动验证语义、资料/索引版本或服务端变化，"
            "不据此宣称稳定提升。"
        ),
    }
