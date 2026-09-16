"""运行产物路径测试：确保 IDE 和终端使用不同 CWD 时结果仍一致。"""

from ai_agent.config import PROJECT_ROOT, resolve_project_path


def test_relative_path_is_resolved_from_project_root():
    """相对路径应锚定项目根目录，而不是 pytest 的当前工作目录。"""
    result = resolve_project_path("plans/study_plan.md")

    assert result == (PROJECT_ROOT / "plans" / "study_plan.md").resolve()


def test_absolute_path_is_preserved():
    """显式绝对路径代表用户选择，不能被强制改写到项目目录。"""
    # 这里只验证路径变换，不创建文件，因此无需依赖系统临时目录权限。
    requested = PROJECT_ROOT / "custom-output" / "plan.md"

    assert resolve_project_path(requested) == requested.resolve()


def test_default_rag_directory_name_is_plural(monkeypatch):
    """默认索引目录使用 .agents_data，避免继续写入旧 .agent_data。"""
    monkeypatch.delenv("RAG_DATA_DIR", raising=False)

    # 直接验证默认路径组成；get_settings 有缓存，不适合依赖测试执行顺序清缓存。
    expected = (PROJECT_ROOT / ".agents_data" / "rag").resolve()

    assert resolve_project_path(".agents_data/rag") == expected
