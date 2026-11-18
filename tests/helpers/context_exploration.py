"""只读请求组成审计与原始回执外置原型；不接入业务执行，不请求模型。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from copy import deepcopy

from langchain_core.utils.function_calling import convert_to_openai_tool

from ai_agent.resources.tools import build_agent_tools


def retry_empty_generation_once(invoke, messages):
    """离线候选边界：只重试无正文且无工具请求的生成，不执行任何工具。"""
    response = invoke(deepcopy(messages))
    if (isinstance(response.content, str) and not response.content.strip()
            and not response.tool_calls):
        return invoke(deepcopy(messages))
    return response


class RawReceiptStore:
    """实验性内容寻址存储：保存原文，通过摘要ID取回原文，不做语义摘要。"""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, content):
        identity = hashlib.sha256(content.encode("utf-8")).hexdigest()
        path = self.root / f"{identity}.txt"
        if not path.exists():
            # 按字节保存，避免Windows换行转换破坏原文及内容指纹。
            path.write_bytes(content.encode("utf-8"))
        return {"receipt_id": identity, "original_chars": len(content),
            "preview": content[:160] + "\n[中间内容省略；预览不是完整证据]\n" + content[-160:],
            "evidence_kind": "tool_result_preview", "complete": False}

    def get(self, identity):
        if not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise ValueError("无效receipt_id")
        content = (self.root / f"{identity}.txt").read_bytes().decode("utf-8")
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != identity:
            raise ValueError("原始回执校验失败")
        return content


def audit_requests(directory):
    directory = Path(directory)
    rows = [json.loads(s) for s in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    schema_root = directory / "materials" / f"{rows[0]['case_id']}-{rows[0]['repeat']}"
    schemas = [convert_to_openai_tool(t) for t in build_agent_tools(str(schema_root))]
    schema_chars = len(json.dumps(schemas, ensure_ascii=False))
    result = []
    for row in rows:
        real = [r for r in row["requests"] if r["source"] == "model_request"]
        for i, request in enumerate(real):
            messages = request["request_messages"]
            roles = {role: sum(len(str(m["content"])) for m in messages if m["role"] == role)
                for role in ("system", "human", "ai", "tool")}
            result.append({"case_id": row["case_id"], "repeat": row["repeat"], "request": i + 1,
                "message_content_chars_by_role": roles,
                "serialized_messages_chars": len(json.dumps(messages, ensure_ascii=False)),
                "tool_schema_chars": schema_chars,
                "actual_input_tokens": row["model_responses"][i]["usage"].get("input_tokens"),
                "note": "字符仅用于组成观察，不是模型token；schema为相同工具的OpenAI格式重建"})
    return result


def offload_probe(root):
    store = RawReceiptStore(root)
    originals = [json.dumps({"ok": True, "data": {"source": f"notes-{i}.md",
        "citation": f"[source=notes-{i}.md]", "scope": "prefix", "returned_chars": 4000,
        "truncated": False, "text": str(i) + "资料正文" * 999 + "结束。"}}, ensure_ascii=False)
        for i in range(10)]
    # returned_chars按实际生成正文计算；此处仅生成实验回执，不伪装真实工具调用。
    for i, original in enumerate(originals):
        data = json.loads(original)
        data["data"]["returned_chars"] = len(data["data"]["text"])
        originals[i] = json.dumps(data, ensure_ascii=False)
    refs = [store.put(content) for content in originals]
    reopened = RawReceiptStore(root)
    assert all(reopened.get(ref["receipt_id"]) == content for ref, content in zip(refs, originals))
    return {"mode": "offline_storage_prototype_no_model", "receipts": len(originals),
        "raw_chars": sum(map(len, originals)), "preview_reference_chars": len(json.dumps(refs, ensure_ascii=False)),
        "exact_readback": True, "production_enabled": False,
        "limitation": "只证明保存和回读无损，不证明模型会定位回读；未测在线质量/token/费用"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "report.json"
    if path.exists():
        parser.error("不覆盖已有观察")
    report = {"request_composition": audit_requests(args.results_dir),
        "offload_prototype": offload_probe(args.output_dir / "raw-receipts")}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["offload_prototype"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
