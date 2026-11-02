"""记录现有Embedding客户端的小样本服务端usage；默认预检，不输出密钥或原始响应。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ai_agent.config import create_embeddings, get_settings

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/evaluation/history_retrieval_coverage.json"


class MeteredClient:
    def __init__(self, client):
        self.client = client
        self.records = []

    def call(self, *args, **kwargs):
        started = time.monotonic()
        response = self.client.call(*args, **kwargs)
        usage = response.get("usage", {}) or {}
        # 仅保存允许的计量字段，不转存密钥、输入正文或供应商完整响应。
        self.records.append({"model": kwargs.get("model"), "text_type": kwargs.get("text_type"),
                             "status_code": response.get("status_code"),
                             "usage": {k: usage[k] for k in ("total_tokens", "input_tokens", "output_tokens") if k in usage},
                             "latency_ms": round((time.monotonic() - started) * 1000)})
        return response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / ".evaluation-results/history-embedding-usage-v1.json")
    args = parser.parse_args(argv)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    texts = [text for source in fixture["sources"] for text in source["user_turns"][:source["through_turn"]]]
    query = "路上用耳朵学东西"
    if not args.embed:
        print(json.dumps({"mode": "preflight_only", "user_messages": texts, "query": query}, ensure_ascii=False, indent=2))
        return 0
    embedding = create_embeddings()
    meter = MeteredClient(embedding.client.client)
    embedding.client.client = meter
    embedding.embed_documents(texts)
    embedding.embed_query(query)
    result = {"mode": "online_embedding_usage_probe", "model": get_settings().embedding_model,
              "document_count": len(texts), "query_count": 1, "provider_calls": meter.records,
              "note": "usage为供应商返回用量，费用需匹配实际地域与账单规则；不代表已核对账户账单。"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
