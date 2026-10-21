from __future__ import annotations

import builtins

import ai_agent.main as main_module


def test_interactive_rag_receives_integer_context_budget(monkeypatch):
    """防止尾逗号把字符预算意外变成单元素元组。"""
    captured: dict[str, object] = {}

    class FakeTool:
        name = "retrieve_learning_context"

    def fake_build_tool(root_dir: str, **kwargs):
        captured.update(kwargs)
        return FakeTool()

    monkeypatch.setattr(main_module, "needs_rag", lambda _: True)
    monkeypatch.setattr(main_module, "create_llm", lambda: object())
    monkeypatch.setattr(
        main_module,
        "create_llm_with_tools",
        lambda tools: object(),
    )
    monkeypatch.setattr(
        main_module,
        "build_retrieve_learning_context_tool",
        fake_build_tool,
    )
    monkeypatch.setattr(builtins, "input", lambda _: "exit")

    main_module.run_interactive_session(
        analysis="资料分析",
        plan="学习计划",
        material_context="初始证据",
        learning_goal="学习 Agent",
        strategy="window",
        material_dir="materials",
    )

    assert type(captured["max_context_chars"]) is int


def test_interactive_without_rag_still_has_line_reader(monkeypatch):
    """无向量检索时，仍绑定受限按行读取工具。"""
    captured: dict[str, object] = {}
    inputs = iter(["解释第一天的安排", "exit"])

    def fake_turn(**kwargs):
        captured["retrieval_tool"] = kwargs["retrieval_tool"]
        captured["read_tool"] = kwargs["read_tool"]
        captured["tool_llm"] = kwargs["tool_llm"]
        return "根据现有计划回答。"

    monkeypatch.setattr(main_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(main_module, "create_llm", lambda: object())
    monkeypatch.setattr(
        main_module,
        "_invoke_interactive_turn",
        fake_turn,
    )
    monkeypatch.setattr(
        builtins,
        "input",
        lambda _: next(inputs),
    )

    main_module.run_interactive_session(
        analysis="资料分析",
        plan="学习计划",
        material_context="初始证据",
        learning_goal="学习 Agent",
        strategy="window",
        material_dir="materials",
    )

    assert captured["retrieval_tool"] is None
    assert captured["read_tool"].name == "read_material_lines"
    assert captured["tool_llm"] is not None

def test_interactive_binds_and_dispatches_pdf_tool(monkeypatch, tmp_path):
    """同时检查模型绑定与程序分发，防止只接好其中一端。"""
    captured = {}
    inputs = iter(["解释当前安排", "exit"])

    def fake_bind(tools):
        captured["bound_names"] = {tool.name for tool in tools}
        return object()

    def fake_turn(**kwargs):
        captured["extra_names"] = {
            tool.name for tool in kwargs["extra_tools"]
        }
        return "根据当前计划解释。"

    monkeypatch.setattr(main_module, "needs_rag", lambda _: False)
    monkeypatch.setattr(main_module, "create_llm", lambda: object())
    monkeypatch.setattr(main_module, "create_llm_with_tools", fake_bind)
    monkeypatch.setattr(main_module, "_invoke_interactive_turn", fake_turn)
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    main_module.run_interactive_session(
        analysis="资料分析",
        plan="学习计划",
        material_context="初始证据",
        learning_goal="学习Agent",
        strategy="window",
        material_dir=str(tmp_path),
    )

    assert "read_pdf_pages" in captured["bound_names"]
    assert "read_pdf_pages" in captured["extra_names"]

def test_interactive_prompt_contains_shared_pdf_citation_rules(monkeypatch):
    """有无语义检索，PDF按页读取都应使用同一份引用约定。"""
    from types import SimpleNamespace

    from ai_agent.runtime.prompt_rules import PDF_CITATION_RULES

    # 使用最小配置，测试提示词拼接，不创建模型或访问资料。
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(
            interactive_preload_materials=False,
            interactive_max_retrieval_calls=2,
            interactive_max_queries_per_tool_call=2,
        ),
    )

    for retrieval_available in (False, True):
        prompt = main_module._build_interactive_system_prompt(
            analysis="资料分析",
            plan="学习计划",
            material_context="资料摘录",
            learning_goal="理解资料内容",
            retrieval_available=retrieval_available,
            on_demand_eligible=True,
        )

        assert PDF_CITATION_RULES in prompt

        # 共享的是引用约定，不应把隔离实验的场景限制带进真实交互。
        assert "你正在参加隔离的PDF资料核对实验" not in prompt

def test_interactive_prompt_contains_material_access_rules(monkeypatch):
    """无论是否启用语义检索，都应保留按已知位置直接读取的规则。"""
    from types import SimpleNamespace

    from ai_agent.runtime.prompt_rules import MATERIAL_ACCESS_RULES

    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(
            interactive_preload_materials=False,
            interactive_max_retrieval_calls=2,
            interactive_max_queries_per_tool_call=2,
        ),
    )

    for retrieval_available in (False, True):
        prompt = main_module._build_interactive_system_prompt(
            analysis="资料分析",
            plan="学习计划",
            material_context="资料摘录",
            learning_goal="理解资料",
            retrieval_available=retrieval_available,
            on_demand_eligible=True,
        )

        # 确认规则进入最终提示词，而不只是新增了一个未使用的变量。
        assert MATERIAL_ACCESS_RULES in prompt

    # 同时保护直接读取与信息不足时的澄清分支，避免规则变得过于绝对。
    assert "不要仅为确认文件是否存在而先列目录" in MATERIAL_ACCESS_RULES
    assert "可以先发现、检索或向用户澄清" in MATERIAL_ACCESS_RULES
    assert "不得绕过工具的安全限制" in MATERIAL_ACCESS_RULES
