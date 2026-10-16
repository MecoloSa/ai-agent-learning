"""测试公共隔离规则。

旧交互测试新增了自动保存副作用，统一重定向到 pytest 临时目录。
专门的自动保存测试自行控制写入，不受此规则影响。
"""

from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def isolate_existing_session_autosave(request, monkeypatch, tmp_path):
    # 只处理会执行交互循环的已有测试文件，避免影响其他测试。
    session_test_files = {
        "test_interactive.py",
        "test_learning_commands.py",
        "test_feedback_parser.py",
        "test_learning_context.py",
        "test_state_recovery.py",
        "test_learning_state_context.py",
        "test_state_autosave.py",
    }
    if request.node.path.name not in session_test_files:
        return

    import ai_agent.main as main_module
    # 旧会话测试已用假问答函数；新增工具绑定也必须保持离线。
    # 各测试若需要检查绑定参数，可以再自行覆盖这个替身。
    monkeypatch.setattr(
        main_module,
        "create_llm_with_tools",
        lambda tools: object(),
    )
    from ai_agent.state_store import save_learning_state

    def save_to_temporary_directory(state):
        path = tmp_path / f"auto-{uuid4().hex}.json"
        return save_learning_state(state, path)

    # main.py 已导入保存函数，因此替换 main 使用的那个引用。
    # 只重定向自动保存，不干扰已有手动保存测试的路径设置。
    monkeypatch.setattr(
        main_module,
        "save_state_snapshot",
        save_to_temporary_directory,
    )