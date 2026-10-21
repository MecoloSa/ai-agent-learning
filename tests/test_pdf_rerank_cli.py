"""评测CLI默认DeepSeek与显式开关；假RAG/模型，验证实际路由参数。"""
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from ai_agent.evaluation import pdf_rag_eval as module
from ai_agent.retrieval.reranking import RerankOutcome


@pytest.mark.parametrize("flags,backend,model", [
    ([], "chat", "deepseek-v4.1-flash"),
    (["--rerank-backend", "dashscope"], "dashscope", "qwen3.7-text-rerank"),
    (["--rerank-model", "qwen3.6-flash"], "chat", "qwen3.6-flash"),
    (["--no-rerank"], None, None),
])
def test_pdf_cli_defaults_and_switches(tmp_path, monkeypatch, capsys, flags, backend, model):
    settings = replace(module.get_settings(), rag_rerank_enabled=True, rag_rerank_backend="chat",
        rag_rerank_chat_model="deepseek-v4.1-flash", rag_rerank_model="qwen3.7-text-rerank",
        chat_api_key="chat-key", chat_base_url="https://chat.example/v1", embedding_api_key="normal-key")
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setenv("NORMAL_DASHSCOPE_URL", "https://normal.example/compatible-mode/v1")
    doc = Document(page_content="正文", metadata={"source": "book.pdf", "page": 1, "chunk_index": 0})

    class FakeRAG:
        def __init__(self, _):
            self.store = SimpleNamespace(get=lambda **_: {"documents": [doc.page_content], "metadatas": [doc.metadata]})

        def sync(self):
            pass

        def retrieve(self, **kwargs):
            assert not kwargs.get("rerank", False)
            kwargs["candidate_audit"].append({**doc.metadata, "raw_rank": 1, "score": .9, "decision": "selected"})
            return [doc]

    monkeypatch.setattr(module, "LearningRAG", FakeRAG)
    calls = []

    def fake_rerank(query, candidates, **options):
        calls.append(options)
        return RerankOutcome(candidates, 12, "fake", options["backend"])

    monkeypatch.setattr(module, "rerank_candidates", fake_rerank)
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"case_id": "test", "question": "问题", "gold_evidence": [
        {"source": "book.pdf", "pages": [1]}]}, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "output.jsonl"
    monkeypatch.setattr(sys, "argv", ["pdf_rag_eval", "--materials", str(tmp_path), "--cases", str(cases),
                                      "--output", str(output), *flags])
    module.main()
    summary = json.loads(capsys.readouterr().out)
    if backend is None:
        assert calls == [] and "rerank" not in summary
    else:
        assert calls[0]["backend"] == backend and calls[0]["model"] == model
        assert calls[0]["api_key"] == ("chat-key" if backend == "chat" else "normal-key")
        assert summary["rerank"]["backend"] == backend
        assert summary["rerank"]["model"] == model
        assert json.loads(output.read_text(encoding="utf-8"))["rerank_backend"] == backend
