from copy import deepcopy
import pytest
from tenacity import stop_after_attempt
from langchain_core.messages import AIMessage, HumanMessage
import ai_agent.runtime.agents as agents


def message(text="回答", **kwargs):
    return AIMessage(content=text, usage_metadata={"input_tokens": 10, "output_tokens": 0 if not text.strip() else 2,
        "total_tokens": 10 if not text.strip() else 12}, **kwargs)


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(agents, "RETRY_STOP", stop_after_attempt(3))
    monkeypatch.setattr(agents, "RETRY_WAIT", lambda state: 0)


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
    def invoke(self, messages):
        self.requests.append(deepcopy(messages))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def test_empty_retries_same_messages_and_sums_usage():
    model = SequenceModel([message(""), message()])
    history = [HumanMessage(content="保留的原始问题")]
    result = agents._invoke_llm_raw(model, history, stage="test")
    assert model.requests[0] == model.requests[1] == history
    assert result.usage_metadata == {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22}
    assert result.response_metadata["provider_usage_metadata"]["input_tokens"] == 10
    assert result.response_metadata["invocation_audit"]["attempt_count"] == 2


def test_two_empty_responses_stop_before_total_limit():
    model = SequenceModel([message(""), message(""), message()])
    with pytest.raises(agents.EmptyLLMResponseError) as caught:
        agents._invoke_llm_raw(model, [], stage="test")
    assert len(model.requests) == 2
    assert caught.value.invocation_audit["aggregate_usage"]["input_tokens"] == 20


@pytest.mark.parametrize("responses", [
    [TimeoutError("timeout"), message(""), message()],
    [message(""), TimeoutError("timeout"), message()],
])
def test_network_and_empty_share_total_budget(responses):
    model = SequenceModel(responses)
    result = agents._invoke_llm_raw(model, [], stage="test")
    assert len(model.requests) == 3
    assert result.usage_metadata is None
    assert result.response_metadata["invocation_audit"]["usage_complete"] is False


def test_one_attempt_config_does_not_grant_extra_request(monkeypatch):
    monkeypatch.setattr(agents, "RETRY_STOP", stop_after_attempt(1))
    model = SequenceModel([message(""), message()])
    with pytest.raises(agents.LLMInvocationError):
        agents._invoke_llm_raw(model, [], stage="test")
    assert len(model.requests) == 1


def test_tool_request_is_not_empty_answer():
    response = message("", tool_calls=[{"id": "r", "name": "read_learning_file", "args": {"relative_path": "a.md"}, "type": "tool_call"}])
    model = SequenceModel([response])
    assert agents._invoke_llm_raw(model, [], stage="test") is response
    assert len(model.requests) == 1


def test_permission_failure_is_not_retried():
    model = SequenceModel([ValueError("invalid api key"), message()])
    with pytest.raises(agents.LLMInvocationError):
        agents._invoke_llm_raw(model, [], stage="test")
    assert len(model.requests) == 1


@pytest.mark.parametrize("metadata, extra", [
    ({"finish_reason": "length"}, {}),
    ({"finish_reason": "content_filter"}, {}),
    ({"finish_reason": "stop"}, {"refusal": "refused"}),
])
def test_explicit_provider_end_states_are_not_empty_recovery(metadata, extra):
    response = message("", response_metadata=metadata, additional_kwargs=extra)
    model = SequenceModel([response])
    assert agents._invoke_llm_raw(model, [], stage="test") is response
    assert len(model.requests) == 1


def test_tool_is_not_executed_again_after_empty(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("有据回答。", encoding="utf-8")
    call = message("", tool_calls=[{"id": "r", "name": "read_learning_file",
        "args": {"relative_path": "a.md"}, "type": "tool_call"}])
    model = SequenceModel([call, message(""), message("有据回答。[source=a.md]")])
    monkeypatch.setattr(agents, "create_llm_with_tools", lambda tools: model)
    answer, context = agents.analyse_materials_agentic(str(tmp_path), "解释资料")
    assert "有据回答" in answer
    assert len(model.requests) == 3
    assert model.requests[1] == model.requests[2]
    assert context.count('"source": "a.md"') == 1
