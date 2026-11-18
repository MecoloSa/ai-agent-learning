"""汇总同次HTTP/SDK/LangChain诊断，不推断服务端内部根因。"""
import argparse
import json
from pathlib import Path


def summarize(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(s) for s in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    mismatches = []
    empties = []
    for row in rows:
        http = row.get("http_response", {}).get("choices", [])
        sdk = row.get("sdk_response", {}).get("choices", [])
        lc = row.get("langchain_response")
        if http and lc:
            raw_content = http[0]["message"].get("content") or ""
            if raw_content != lc["content"] or (sdk and (sdk[0]["message"].get("content") or "") != raw_content):
                mismatches.append(row["index"])
        if row.get("empty"):
            empties.append({"index": row["index"], "http_status": row.get("http_status"),
                "request_id": row.get("public_headers", {}).get("x-request-id"),
                "http_public_response": row.get("http_response"),
                "wire_request_keys": row.get("wire_request_keys"),
                "sdk_content": sdk[0]["message"].get("content") if sdk else None,
                "langchain_content": lc.get("content") if lc else None})
    return {"directory": str(directory), "manifest": manifest, "actual_http_requests": len(rows),
        "http_200": sum(row.get("http_status") == 200 for row in rows),
        "empty_count": len(empties), "cross_layer_content_mismatch_indices": mismatches,
        "identical_request_json_within_group": bool(rows) and all(row.get("wire_request") == rows[0].get("wire_request") for row in rows),
        "empty_evidence": empties}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("不覆盖已有诊断报告")
    result = [summarize(path) for path in args.groups]
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("directory", "actual_http_requests", "empty_count",
        "cross_layer_content_mismatch_indices", "identical_request_json_within_group")} for r in result], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
