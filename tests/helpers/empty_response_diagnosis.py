"""复用失败请求，逐层观察HTTP/SDK/LangChain；不加语义重试，不保存私有推理文本。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
from urllib.parse import urlsplit

from langchain_core.messages import messages_from_dict
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.chat_models.base import BaseChatOpenAI

from ai_agent.config import create_llm
from ai_agent.resources.tools import build_agent_tools

ROOT = Path(__file__).resolve().parents[2]
FAILED = ROOT / ".evaluation-results/discovery-limit-online-20261103/results.jsonl"


def public_response(payload):
    """保留公开回答和状态；不写入reasoning_content等私有推理字段。"""
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}
    result = {k: payload[k] for k in ("id", "model", "object", "created", "usage") if k in payload}
    result["top_level_keys"] = list(payload)
    result["choices"] = []
    for choice in payload.get("choices") or []:
        message = choice.get("message") or {}
        result["choices"].append({"finish_reason": choice.get("finish_reason"),
            "message_keys": list(message), "message": {k: message[k]
                for k in ("role", "content", "tool_calls", "function_call", "refusal") if k in message},
            "private_field_lengths": {k: len(v) if isinstance(v, str) else 0
                for k, v in message.items() if "reason" in k}})
    return result


def load_failed(case="discovery_failure"):
    if case == "fixed_failure":
        path = ROOT / ".evaluation-results/receipt-ab-online-20261103/results.jsonl"
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
        row = next(r for r in rows if r["case_id"] == "failure_then_success" and r["arm"] == "compact" and r["repeat"] == 2)
        records = row["request_messages"]
        tools = []
    else:
        rows = [json.loads(s) for s in FAILED.read_text(encoding="utf-8").splitlines()]
        row = next(r for r in rows if r.get("discovery_arm") == "baseline" and r["repeat"] == 2)
        records = row["requests"][-1]["request_messages"]
        root = ROOT / ".evaluation-results/discovery-limit-online-20261103/materials/baseline-2"
        tools = build_agent_tools(str(root))
        assert [{"name": t.name, "args": t.args} for t in tools] == row["tool_schemas"]
    messages = messages_from_dict([{"type": m["role"], "data": {k: v for k, v in m.items() if k != "role"}}
        for m in records])
    return messages, tools, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-calls", type=int, default=12)
    parser.add_argument("--stop-on-empty", action="store_true")
    parser.add_argument("--case", choices=("discovery_failure", "fixed_failure"), default="discovery_failure")
    parser.add_argument("--direct-http", action="store_true")
    parser.add_argument("--variant", choices=("baseline", "no_tools", "temperature_zero", "stream", "no_thinking"), default="baseline")
    args = parser.parse_args()
    if not 1 <= args.max_calls <= 30:
        parser.error("每组1—30次，按诊断结果决定是否继续")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        parser.error("不覆盖已有诊断")
    messages, tools, records = load_failed(args.case)
    llm = create_llm()
    if args.variant == "temperature_zero":
        llm = llm.model_copy(update={"temperature": 0.0})
    if args.variant == "stream":
        llm = llm.model_copy(update={"streaming": True})
    if args.variant == "no_thinking":
        llm = llm.model_copy(update={"extra_body": {"enable_thinking": False}})
    bound = llm if args.variant == "no_tools" or not tools else llm.bind_tools(tools)
    client = llm.root_client._client
    endpoint = str(llm.openai_api_base)
    hostname = urlsplit(endpoint).hostname or ""
    manifest = {"case": args.case, "variant": args.variant, "model": llm.model_name, "temperature": llm.temperature,
        "max_calls": args.max_calls, "streaming": llm.streaming,
        "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest(),
        "endpoint_is_aliyuncs": hostname.endswith("aliyuncs.com"),
        "tools_bound": bool(tools) and args.variant != "no_tools", "sdk_max_retries": llm.max_retries,
        "client": "direct_http" if args.direct_http else "langchain_and_openai_sdk",
        "input_sha256": hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "input.json").write_text(json.dumps({"messages": records,
        "tools": [convert_to_openai_tool(t) for t in tools]}, ensure_ascii=False, indent=2), encoding="utf-8")
    active = {}

    def on_request(request):
        payload = json.loads(request.read())
        active["wire_request"] = {k: v for k, v in payload.items()
            if k in ("model", "messages", "tools", "temperature", "stream", "max_tokens", "max_completion_tokens", "tool_choice", "stop", "top_p", "enable_thinking")}
        active["wire_request_keys"] = list(payload)

    def on_response(response):
        response.read()
        active["http_status"] = response.status_code
        active["public_headers"] = {k: response.headers[k] for k in ("content-type", "server", "x-request-id", "request-id") if k in response.headers}
        if "text/event-stream" in response.headers.get("content-type", ""):
            # 流式分支仅记录公开delta字段，不存储推理正文。
            chunks = []
            for line in response.text.splitlines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    data = json.loads(line[6:])
                    for choice in data.get("choices") or []:
                        delta = choice.get("delta") or {}
                        chunks.append({"finish_reason": choice.get("finish_reason"),
                            "delta": {k: v for k, v in delta.items() if k in ("content", "tool_calls", "role")}})
            active["http_chunks"] = chunks
        else:
            active["http_response"] = public_response(response.json())

    original = BaseChatOpenAI._create_chat_result
    def observe_sdk(self, response, generation_info=None):
        active["sdk_response"] = public_response(response if isinstance(response, dict) else response.model_dump())
        return original(self, response, generation_info)

    client.event_hooks["request"].append(on_request)
    client.event_hooks["response"].append(on_response)
    try:
        with patch.object(BaseChatOpenAI, "_create_chat_result", observe_sdk), (output / "results.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(args.max_calls):
                active = {"index": index + 1}
                started = time.perf_counter()
                try:
                    if args.direct_http:
                        if args.variant == "stream":
                            raise ValueError("直接HTTP对照仅使用非流式")
                        kwargs = {} if not tools or args.variant == "no_tools" else {"tools": [convert_to_openai_tool(t) for t in tools]}
                        payload = llm._get_request_payload(messages, **kwargs)
                        # 复用同一连接策略和凭据；不记录Authorization或密钥。
                        raw = client.post(endpoint.rstrip("/") + "/chat/completions", json=payload,
                            headers={"Authorization": "Bearer " + llm.openai_api_key.get_secret_value()},
                            timeout=llm.request_timeout)
                        raw.raise_for_status()
                        choice = raw.json()["choices"][0]
                        message = choice["message"]
                        active["empty"] = not message.get("content") and not message.get("tool_calls")
                    else:
                        response = bound.invoke(messages)
                        active["langchain_response"] = {"content": response.content, "tool_calls": response.tool_calls,
                            "usage": response.usage_metadata, "finish_reason": response.response_metadata.get("finish_reason")}
                        active["empty"] = not response.content and not response.tool_calls
                    active["status"] = "completed"
                except Exception as exc:
                    active.update(status="failed", error_type=type(exc).__name__)
                active["latency_ms"] = round((time.perf_counter() - started) * 1000)
                stream.write(json.dumps(active, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{args.variant} {index + 1}: status={active['status']} empty={active.get('empty')} "
                    f"http={active.get('http_status')} usage={active.get('langchain_response', {}).get('usage')}", flush=True)
                if active["status"] != "completed" or (args.stop_on_empty and active.get("empty")):
                    break
    finally:
        client.event_hooks["request"].remove(on_request)
        client.event_hooks["response"].remove(on_response)


if __name__ == "__main__":
    main()
