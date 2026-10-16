"""交互资料上下文策略测试。

验证按需检索模式、完整预载对照、无检索回退，以及计划更新后
不会重新携带大段正文。使用假模型，不访问在线服务或真实快照。
"""

import builtins
from dataclasses import replace

import pytest

import ai_agent.main as main_module


def configure_policy(monkeypatch, *, preload):
    settings = replace(
        main_module.get_settings(),
        interactive_preload_materials=preload,
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)


@pytest.mark.parametrize(
    "retrieval_available, preload, eligible, expected_raw",
    [
        (True, False, True, False),   # 工具可用且覆盖资料：按需检索。
        (True, False, False, True),   # 覆盖不足：保留原始上下文。
        (True, True, True, True),     # 手动开启完整预载。
        (True, True, False, True),
        (False, False, True, True),   # 无工具，即使类型支持也不能移除。
        (False, False, False, True),
    ],
)
def test_prompt_selects_evidence_policy(
    monkeypatch, retrieval_available, preload, eligible, expected_raw,
):
    configure_policy(monkeypatch, preload=preload)
    raw_material = "唯一正文标记-RAW-EVIDENCE"

    prompt = main_module._build_interactive_system_prompt(
        analysis="分析标记",
        plan="计划标记",
        material_context=raw_material,
        learning_goal="目标标记",
        retrieval_available=retrieval_available,
        on_demand_eligible=eligible,
    )

    assert (raw_material in prompt) is expected_raw
    assert "分析标记" in prompt
    assert "计划标记" in prompt
    assert "目标标记" in prompt

    expected_mode = "preloaded" if expected_raw else "on_demand"
    assert f"策略={expected_mode}" in prompt

    if not expected_raw:
        assert "本轮未预载资料正文" in prompt


def test_same_inputs_produce_smaller_prompt_in_on_demand_mode(monkeypatch):
    raw_material = "正文标记" * 25_000

    configure_policy(monkeypatch, preload=True)
    full_prompt = main_module._build_interactive_system_prompt(
        "资料分析", "当前计划", raw_material, "当前目标",
        retrieval_available=True,
        on_demand_eligible=True,
    )

    configure_policy(monkeypatch, preload=False)
    lean_prompt = main_module._build_interactive_system_prompt(
        "资料分析", "当前计划", raw_material, "当前目标",
        retrieval_available=True,
        on_demand_eligible=True,
    )

    # 同一输入比较，只改变证据策略，不引入模型生成的随机差异。
    assert "正文标记" in full_prompt
    assert "正文标记" not in lean_prompt
    assert len(lean_prompt) < len(full_prompt)

    print(f"\n完整预载提示词：{len(full_prompt)} 字符")
    print(f"按需检索提示词：{len(lean_prompt)} 字符")
    print("这里只验证字符规模，不宣称模型质量或费用改善。")


def test_plan_update_keeps_on_demand_policy(monkeypatch, tmp_path):
    configure_policy(monkeypatch, preload=False)
    material_root = tmp_path / "materials"
    material_root.mkdir()

    # 只用于资料类型扫描；模型与 RAG 已被替换，不解析这个空文件。
    (material_root / "book.pdf").touch()
    inputs = iter(["请修改计划", "解释新计划", "exit"])
    requests = []

    class FakeTool:
        name = "retrieve_learning_context"

    def fake_turn(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return "===NEW_PLAN_START===新计划===NEW_PLAN_END==="
        return "这是对新计划的解释。"

    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(main_module, "needs_rag", lambda _: True)
    monkeypatch.setattr(
        main_module, "create_llm", lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        main_module, "create_llm_with_tools", lambda tools: object(),
    )
    monkeypatch.setattr(
        main_module,
        "build_retrieve_learning_context_tool",
        lambda *args, **kwargs: FakeTool(),
    )
    # 现有会话还有一处独立的 LearningRAG 初始化，此处隔离它的副作用。
    monkeypatch.setattr(
        main_module, "LearningRAG", lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(main_module, "_invoke_interactive_turn", fake_turn)
    monkeypatch.setattr(
        main_module,
        "save_state_snapshot",
        lambda state: tmp_path / "unused.json",
    )

    result = main_module.run_interactive_session(
        analysis="资料分析",
        plan="旧计划",
        material_context="唯一正文标记" * 10_000,
        learning_goal="当前目标",
        strategy="window",
        material_dir=str(material_root),
    )

    assert result == ("新计划", "当前目标")
    assert len(requests) == 2

    for request in requests:
        prompt = str(request["request_messages"][0].content)
        assert "唯一正文标记" not in prompt
        assert "策略=on_demand" in prompt
        assert request["retrieval_tool"] is not None
        assert request["tool_llm"] is not None

    second_prompt = str(requests[1]["request_messages"][0].content)
    assert "当前学习计划：\n新计划" in second_prompt

@pytest.mark.parametrize(
    "filenames, expected",
    [
        (["book.pdf", "notes.md", "outline.txt"], True),
        (["book.pdf", "example.py"], False),
        (["example.py"], False),
        ([], False),
    ],
)
def test_on_demand_eligibility_matches_material_coverage(
    tmp_path, filenames, expected,
):
    """资料类型没有访问入口时，不能移除预载上下文。"""
    for filename in filenames:
        (tmp_path / filename).touch()

    actual = main_module._supports_on_demand_materials(str(tmp_path))

    print(f"\n资料={filenames}，允许按需模式={actual}")
    assert actual is expected


def test_coverage_scan_failure_disables_on_demand(monkeypatch, tmp_path):
    """无法确认覆盖范围时保守回退，不冒险删除正文。"""
    def fail_scan(root):
        raise PermissionError("模拟扫描权限不足")

    monkeypatch.setattr(main_module, "iter_material_files", fail_scan)

    assert main_module._supports_on_demand_materials(str(tmp_path)) is False


def test_mixed_materials_keep_python_context(monkeypatch, tmp_path):
    """串联真实类型检查与提示词组装，防止源码被移出后无法找回。"""
    configure_policy(monkeypatch, preload=False)
    (tmp_path / "book.pdf").touch()
    (tmp_path / "example.py").touch()

    eligible = main_module._supports_on_demand_materials(str(tmp_path))
    raw_context = (
        "PDF资料摘录\n"
        "Python源码：def finish_loop(): return '源码唯一标记'"
    )

    prompt = main_module._build_interactive_system_prompt(
        analysis="资料分析",
        plan="学习计划",
        material_context=raw_context,
        learning_goal="理解资料",
        retrieval_available=True,
        on_demand_eligible=eligible,
    )

    assert eligible is False
    assert "策略=preloaded" in prompt
    assert raw_context in prompt


def test_missing_coverage_argument_keeps_context(monkeypatch):
    """调用方漏传检查结果时，默认保留正文，而非默认移除。"""
    configure_policy(monkeypatch, preload=False)

    prompt = main_module._build_interactive_system_prompt(
        "资料分析", "学习计划", "不可丢失的正文", "学习目标",
        retrieval_available=True,
    )

    assert "不可丢失的正文" in prompt
    assert "策略=preloaded" in prompt