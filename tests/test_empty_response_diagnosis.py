import json

from tests.helpers.empty_response_diagnosis import public_response
from ai_agent.config import create_llm


def test_capture_keeps_empty_response_without_private_reasoning():
    payload = {"choices": [{"message": {"role": "assistant", "content": "",
        "reasoning_content": "private reasoning must not be saved"}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 0}}
    result = public_response(payload)
    assert result["choices"][0]["message"]["content"] == ""
    assert result["choices"][0]["private_field_lengths"]["reasoning_content"] > 0
    assert "private reasoning must not be saved" not in json.dumps(result)


def test_none_reasoning_has_zero_length():
    result = public_response({"choices": [{"message": {"content": "", "reasoning_content": None}}]})
    assert result["choices"][0]["private_field_lengths"]["reasoning_content"] == 0


def test_langchain_parser_does_not_invent_text_for_upstream_empty():
    # 仅创建客户端并运行本地解析，不发送模型请求。
    result = create_llm()._create_chat_result({"model": "test", "choices": [{
        "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}})
    assert result.generations[0].message.content == ""
    assert result.generations[0].message.tool_calls == []
