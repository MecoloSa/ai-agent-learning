"""模型分工测试。

验证配置选择、工具绑定和日志识别，不调用在线模型。
Max 仅验证工厂创建能力，不测试尚未实现的页面或命令行入口。
"""

from types import SimpleNamespace

import pytest

import ai_agent.config as config_module
from ai_agent.runtime.agents import _get_model_name


@pytest.fixture
def factory_calls(monkeypatch):
    """替换真实客户端，收集模型构造参数。"""
    calls = []

    settings = SimpleNamespace(
        chat_model="qwen3.7-plus",
        feedback_model="qwen3.8-flash",
        enhanced_model="qwen3.8-max",
        llm_temperature=0.2,
        request_timeout=180,
        api_force_ipv4=False,
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        config_module, "_required_env", lambda name: "test-only",
    )

    class FakeModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.model_name = kwargs["model"]

        def bind_tools(self, tools):
            # 模拟工具绑定包装；不执行工具，也不发送模型请求。
            return SimpleNamespace(bound=self, tools=list(tools))

    monkeypatch.setattr(config_module, "ChatOpenAI", FakeModel)
    return calls


@pytest.mark.parametrize("role, expected", [
    ("standard", "qwen3.7-plus"),
    ("feedback", "qwen3.8-flash"),
    ("enhanced", "qwen3.8-max"),
])
def test_factory_selects_expected_model(factory_calls, role, expected):
    model = config_module.create_llm(role=role)

    assert model.model_name == expected
    assert factory_calls[-1]["max_retries"] == 0


def test_default_model_remains_plus(factory_calls):
    model = config_module.create_llm()

    assert model.model_name == "qwen3.7-plus"


def test_unknown_role_is_rejected(factory_calls):
    with pytest.raises(ValueError, match="未知模型用途"):
        config_module.create_llm(role="invalid")

    assert factory_calls == []


def test_tools_bind_to_default_plus(factory_calls):
    tool = SimpleNamespace(name="test_tool")

    model = config_module.create_llm_with_tools([tool])

    assert model.bound.model_name == "qwen3.7-plus"
    assert model.tools == [tool]
    assert _get_model_name(model) == "qwen3.7-plus"


def test_unknown_model_is_not_mislabeled_as_plus():
    assert _get_model_name(object()) == "unknown"
