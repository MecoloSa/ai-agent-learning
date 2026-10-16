"""离线复算追加实验，不发模型请求；检验同池、同题覆盖及重复运行稳定性。"""
from collections import defaultdict
import hashlib
import statistics

from run import HERE, read_jsonl, write_json


def main():
    rows = read_jsonl(HERE / "deepseek_followup_traces_partial.jsonl")
    digest = hashlib.sha256((HERE / "inputs.local.json").read_bytes()).hexdigest()
    assert all(r["input_sha256"] == digest for r in rows)
    result = {"input_sha256": digest, "models": {}}
    for model in ["deepseek-v4.1-flash", "qwen3.6-flash"]:
        selected = [r for r in rows if r["model"] == model]
        pairs = defaultdict(list)
        for r in selected:
            pairs[r["case_id"]].append(r)
        assert len(selected) == 36 and len(pairs) == 18
        assert all(sorted(r["repeat"] for r in pair) == [0, 1] for pair in pairs.values())
        result["models"][model] = {
            "requests": len(selected),
            "strict_valid": sum(r.get("strict_valid", False) for r in selected),
            "completed_requests": sum(r.get("completed_missing_count", 0) > 0 for r in selected),
            "fallbacks": sum(r["status"] != "ok" for r in selected),
            "top1_consistent_cases": sum(pair[0]["ranked_page_ids"][0] == pair[1]["ranked_page_ids"][0]
                                         for pair in pairs.values()),
            "miss_recall8": [{"case_id": r["case_id"], "repeat": r["repeat"]}
                             for r in selected if r["recall8"] < 1],
            "response_models": sorted({r.get("response_model", "unknown") for r in selected}),
            "reasoning_usage_reported": sum(r.get("reasoning_tokens") is not None for r in selected),
        }
    by_key = {(r["model"], r["case_id"], r["repeat"]): r for r in rows}
    differences = [r["latency_ms"] - by_key[("qwen3.6-flash", r["case_id"], r["repeat"])]["latency_ms"]
                   for r in rows if r["model"] == "deepseek-v4.1-flash"]
    result["paired_median_deepseek_minus_qwen_ms"] = statistics.median(differences)
    result["paired_deepseek_faster_requests"] = sum(d < 0 for d in differences)
    result["additional_model_requests"] = 72
    result["additional_model_tokens"] = sum(r.get("total_tokens", 0) or 0 for r in rows)
    write_json(HERE / "deepseek_followup_analysis.json", result)
    print(result)


if __name__ == "__main__":
    main()
