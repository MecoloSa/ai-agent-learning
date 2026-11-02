import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def helper():
    path = Path(__file__).parent / "helpers/history_embedding_usage_probe.py"
    spec = importlib.util.spec_from_file_location("embedding_usage_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_meter_preserves_response_and_records_only_allowed_usage(helper):
    response = {"status_code": 200, "usage": {"total_tokens": 21, "secret_field": "private"},
                "output": {"embeddings": [[1.0]]}, "request_id": "private-id"}
    class Client:
        def call(self, **kwargs):
            assert kwargs["api_key"] == "private-key"
            return response
    meter = helper.MeteredClient(Client())
    assert meter.call(model="test", text_type="query", api_key="private-key", input="private-text") is response
    assert meter.records[0]["usage"] == {"total_tokens": 21}
    assert "private" not in json.dumps(meter.records)


def test_usage_preflight_makes_no_requests(helper, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(helper, "create_embeddings", lambda: pytest.fail("预检不能创建客户端"))
    output = tmp_path / "result.json"
    assert helper.main(["--output", str(output)]) == 0
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["mode"] == "preflight_only"
