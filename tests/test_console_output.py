"""只验证展示：日志级别和全文开关不能改变业务输入、文件内容或失败状态。"""
import logging
import sys
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from ai_agent import agents, main as main_module, rag as rag_module
from ai_agent.console_logging import configure_console_logging
from ai_agent import console_logging


@pytest.mark.parametrize("verbose,configured,expected", [
    (False, "INFO", logging.INFO), (True, "INFO", logging.DEBUG), (False, "DEBUG", logging.DEBUG),
])
def test_console_level_keeps_debug_opt_in_and_dependency_noise_off(monkeypatch, verbose, configured, expected):
    captured = {}
    levels = {}
    # 只替换被测模块的logging引用，避免干扰pytest自己的根日志捕获器。
    monkeypatch.setattr(console_logging, "logging", SimpleNamespace(
        INFO=logging.INFO, DEBUG=logging.DEBUG, WARNING=logging.WARNING,
        basicConfig=lambda **kwargs: captured.update(kwargs),
        getLogger=lambda name: SimpleNamespace(setLevel=lambda level: levels.update({name: level}))))
    configure_console_logging(verbose=verbose, log_level=configured)
    assert captured["level"] == expected
    assert all(level == logging.WARNING for level in levels.values())


def test_default_model_log_is_compact_without_changing_response(caplog):
    response = AIMessage(content="正常回答", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
    calls = []
    llm = SimpleNamespace(model_name="fake", invoke=lambda messages: calls.append(messages) or response)
    messages = [HumanMessage(content="不应在默认日志显示的资料")]
    with caplog.at_level(logging.INFO, logger="ai_agent.agents"):
        actual = agents._invoke_llm_raw(llm, messages, stage="测试阶段")
    assert actual is response and calls == [messages]
    assert "模型调用完成" in caplog.text and "tokens=12" in caplog.text
    assert "input_tokens" not in caplog.text and "不应在默认日志显示的资料" not in caplog.text


def test_retrieval_debug_visibility_does_not_change_hits_or_audit(caplog):
    results = []
    for level in [logging.INFO, logging.DEBUG]:
        caplog.clear()
        doc = Document(page_content="只用于输出校验的证据", metadata={"source": "book.pdf", "page": 1, "chunk_index": 0})
        rag = object.__new__(rag_module.LearningRAG)
        rag.store = SimpleNamespace(similarity_search_with_relevance_scores=lambda **kwargs: [(doc, .9)])
        rag.sync = lambda: None
        audit = []
        with caplog.at_level(level, logger="ai_agent.rag"):
            hits = rag.retrieve("输出校验问题", k=1, min_relevance=.3, relative_margin=.2,
                                min_hits=1, candidate_audit=audit)
        results.append(([(hit.page_content, hit.metadata) for hit in hits], audit))
        if level == logging.INFO:
            assert "RAG 查询" not in caplog.text and "RAG 命中" not in caplog.text
        else:
            assert "RAG 查询" in caplog.text and "RAG 命中" in caplog.text
    assert results[0] == results[1]


@pytest.mark.parametrize("orchestrator", ["legacy", "langgraph"])
def test_show_output_only_changes_console_not_saved_document_or_business_calls(tmp_path, monkeypatch, capsys, orchestrator):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return "唯一分析正文", "唯一计划正文", "原始上下文"

    monkeypatch.setattr(main_module, "run_learning_assistant", fake_run)
    outputs = []
    saved = []
    for show in [False, True]:
        path = tmp_path / f"plan-{show}.md"
        argv = ["ai_agent", "--materials", "materials", "--goal", "保持业务不变", "--days", "3",
                "--mode", "agent", "--orchestrator", orchestrator, "--output", str(path)]
        if show:
            argv += ["--show-output", "-v"]
        monkeypatch.setattr(sys, "argv", argv)
        main_module.main()
        outputs.append(capsys.readouterr().out)
        saved.append(path.read_bytes())
    assert saved[0] == saved[1]
    assert calls[0] == calls[1]
    assert "唯一分析正文" not in outputs[0] and "唯一计划正文" not in outputs[0]
    assert "唯一分析正文" in outputs[1] and "唯一计划正文" in outputs[1]
    assert all("已保存到" in output for output in outputs)


def test_failure_still_visible_and_keeps_nonzero_exit(tmp_path, monkeypatch, capsys, caplog):
    def fail(*args, **kwargs):
        raise agents.AnalysisExecutionError("model_round_limit", "测试终止")

    monkeypatch.setattr(main_module, "run_learning_assistant", fail)
    path = tmp_path / "never-written.md"
    monkeypatch.setattr(sys, "argv", ["ai_agent", "--goal", "测试", "--output", str(path)])
    with pytest.raises(SystemExit) as exc:
        main_module.main()
    assert exc.value.code == 1 and not path.exists()
    assert "资料分析未完成" in capsys.readouterr().out
    assert "model_round_limit" in caplog.text
