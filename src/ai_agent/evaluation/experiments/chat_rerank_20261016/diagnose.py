"""每个候选模型复查一个失败例；只保存排序输出，不保存教材或思考内容。"""
import json
import os
import time

from openai import OpenAI
from run import HERE, MODELS, PROMPT, read_jsonl, validate_order, write_json
from ai_agent.infrastructure.network import get_ipv4_chat_client

cases = json.loads((HERE / "inputs.local.json").read_text(encoding="utf-8"))
traces = read_jsonl(HERE / "traces.jsonl")
url = os.environ["CHAT_BASE_URL"]
assert "token-plan" in url
client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=url,
                http_client=get_ipv4_chat_client(url), timeout=45, max_retries=0)
rows = []
for model in MODELS:
    failed = next(r for r in traces if r["model"] == model and r["status"] == "fallback")
    case = next(c for c in cases if c["case_id"] == failed["case_id"])
    payload = {"question": case["question"], "candidates": [
        {"index": i, "source": d["metadata"]["source"], "text": d["text"]}
        for i, d in enumerate(case["documents"])]}
    started = time.perf_counter()
    response = client.chat.completions.create(model=model, temperature=0, max_tokens=160,
        extra_body={"enable_thinking": False}, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    parsed = json.loads(response.choices[0].message.content)
    row = {"model": model, "case_id": case["case_id"], "json_keys": list(parsed),
           "total_tokens": response.usage.total_tokens,
           "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
    order = parsed.get("order")
    if isinstance(order, list) and all(type(x) is int for x in order):
        row.update({"returned_order": order, "missing": sorted(set(range(16)) - set(order)),
                    "out_of_range": sorted(set(order) - set(range(16)))})
    try:
        validate_order(response.choices[0].message.content, 16)
        row["valid"] = True
    except ValueError:
        row["valid"] = False
    rows.append(row)
write_json(HERE / "diagnosis.json", rows)
print(json.dumps(rows, ensure_ascii=False, indent=2))
