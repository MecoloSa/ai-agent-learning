"""参数错误回传与有界修正测试，不请求模型、不修改真实存档。"""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, StrictInt, ValidationError

import ai_agent.main as main

pytestmark = pytest.mark.skipif(
    not hasattr(main, "_format_tool_validation_error"),
    reason="参数错误结构化回传方案尚未应用",
)


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: StrictInt


@pytest.mark.parametrize("payload, kind", [
    ({}, "missing"),
    ({"count": "PRIVATE-VALUE"}, "int_type"),
    ({"count": 1, "other": "PRIVATE-VALUE"}, "extra_forbidden"),
])
def test_errors_expose_location_and_type_not_input(payload, kind):
    with pytest.raises(ValidationError) as captured:
        Arguments.model_validate(payload)
    result = json.loads(main._format_tool_validation_error(captured.value))
    assert result["error"]["code"] == "ValidationError"
    assert result["error"]["issues"][0]["type"] == kind
    assert result["error"]["issues"][0]["loc"]
    assert "PRIVATE-VALUE" not in json.dumps(result)


def test_error_list_is_bounded():
    with pytest.raises(ValidationError) as captured:
        Arguments.model_validate({f"field{i}": "PRIVATE-VALUE" for i in range(8)})
    result = json.loads(main._format_tool_validation_error(captured.value))
    assert len(result["error"]["issues"]) == 5
    assert result["error"]["issues_truncated"] is True


def test_corrected_request_uses_existing_loop_budget(monkeypatch):
    successful = []

    class Tool:
        name = "probe"
        def invoke(self, args):
            parsed = Arguments.model_validate(args)
            successful.append(parsed.count)
            return json.dumps({"ok": True, "data": parsed.count})

    calls = 0
    def respond(llm, messages, *, stage):
        nonlocal calls
        calls += 1
        if calls == 2:
            observation = json.loads(messages[-1].content)
            assert observation["error"]["issues"] == [{"loc": ["count"], "type": "int_type"}]
        if calls <= 2:
            return AIMessage(content="", tool_calls=[{
                "name": "probe", "args": {"count": "bad" if calls == 1 else 1},
                "id": f"call-{calls}",
            }])
        return AIMessage(content="已根据结果回答")

    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(interactive_max_retrieval_calls=3))
    monkeypatch.setattr(main, "invoke_llm_response", respond)
    trace = {}
    main._invoke_interactive_turn(
        request_messages=[], user_input="请核对资料", plain_llm=object(),
        tool_llm=object(), retrieval_tool=Tool(), audit_trace=trace,
    )
    assert successful == [1]
    assert len(trace["tool_results"]) == 2
    assert trace["finished"] is True
