"""tools.py 的单元测试：路径穿越防护与边界条件。

运行：uv run pytest tests/test_tools.py -v
"""
from __future__ import annotations # 向前兼容：在当前 Python 版本中使用未来版本的新功能或行为。

"""pytest 测试规则
pytest 的规则（约定大于配置）：
- 扫描 testpaths（你配置的 tests/）下的文件
- 文件名以 test_ 开头的被当作测试文件
- 类名以 Test 开头、不含 __init__ 的类被当作测试容器
- 函数名以 test_ 开头的被当作一个测试用例

tmp_path 是 pytest 自动注入的参数，你不用自己创建。
每次运行这个测试，pytest 会创建一个独一无二的临时目录，
"""

from pathlib import Path

import pytest

from ai_agent.tools import (
    list_learning_files,
    read_learning_file,
    search_learning_files,
    summarize_text_statistics,
    MAX_CHARS_PER_FILE,
)
from ai_agent.materials import safe_material_path


# ---------------------------------------------------------------------------
# safe_material_path：路径穿越防护（安全护栏）
# 测试模式：AAA 模式 —— Arrange（准备）  →  Act（执行）  →  Assert（断言）
# ---------------------------------------------------------------------------

class TestSafeMaterialPath:
    """验证公共路径解析函数能拦截越界访问。"""

    def test_normal_relative_path_within_root(self, tmp_path: Path):
        """正常子路径：应原样返回，不抛错。"""
        # Arrange：造一个子目录
        (tmp_path / "sub").mkdir()
        # Act：调用被测函数
        result = safe_material_path(str(tmp_path), "sub/notes.md")
        # Assert：断言结果在根目录内
        assert result == (tmp_path / "sub" / "notes.md").resolve()
        assert tmp_path in result.parents

    def test_directory_is_not_a_supported_material(self, tmp_path: Path):
        """路径虽未越界，但资料读取接口只接受受支持的文件。"""
        with pytest.raises(ValueError, match="不支持的资料类型"):
            safe_material_path(str(tmp_path), ".")

    """@pytest.mark.parametrize —— 一次声明，多次运行
    pytest 会把这个方法复制成 5 个独立的测试用例
    注意，顶格的 """""" 是一条模块级表达式语句，它的出现让 Python 认为"类的定义到此结束"
    """
    @pytest.mark.parametrize("malicious", [
        "../secret.txt",           # 直接跳出上一级
        "../../etc/passwd",        # 连续跳出多级
        "sub/../../etc/shadow",    # 先进入再跳出
        "/etc/passwd",             # 绝对路径（Unix）
        "/windows/system32/config",# 绝对路径（Windows 风格）
    ])
    def test_traversal_attacks_are_blocked(self, tmp_path: Path, malicious: str):
        """各类穿越手法都应被拒绝。"""
        # with 的本质是：进入时做某事，退出时做某事。这里进入时开始监听异常，退出时验证异常确实发生了
        with pytest.raises(ValueError, match="不允许读取资料目录以外的文件"):
            safe_material_path(str(tmp_path), malicious)


# ---------------------------------------------------------------------------
# list_learning_files：目录扫描的边界条件
# ---------------------------------------------------------------------------

class TestListLearningFiles:
    """验证空目录、不存在目录、文件类型过滤、数量上限。"""

    def test_empty_directory(self, tmp_path: Path):
        """空目录应返回友好提示，而非空列表或报错。"""
        assert list_learning_files.invoke({"root_dir": str(tmp_path)}) == "没有找到可分析的资料文件。"

    def test_nonexistent_directory(self, tmp_path: Path):
        """不存在的目录应返回提示，而非抛异常。"""
        missing = tmp_path / "no_such_dir"
        result = list_learning_files.invoke({"root_dir": str(missing)})
        assert result.startswith("目录不存在：")

    def test_only_lists_supported_suffixes(self, tmp_path: Path):
        """.txt/.md/.py/.pdf 列出，.exe/.bin 忽略。"""
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.md").write_text("b")
        (tmp_path / "c.py").write_text("c")
        (tmp_path / "d.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "e.exe").write_bytes(b"\x00\x01")  # 不支持
        (tmp_path / "f.bin").write_bytes(b"\x00\x01")  # 不支持
        result = list_learning_files.invoke({"root_dir": str(tmp_path)})
        assert "a.txt" in result
        assert "b.md" in result
        assert "c.py" in result
        assert "d.pdf" in result
        assert "e.exe" not in result
        assert "f.bin" not in result

    def test_truncates_at_50_files(self, tmp_path: Path):
        """超过 50 个文件时只展示前 50 个。"""
        for i in range(60):
            (tmp_path / f"file_{i:02d}.txt").write_text("x")
        result = list_learning_files.invoke({"root_dir": str(tmp_path)})
        # 返回按行拼接，行数应等于 50
        assert len(result.splitlines()) == 50


# ---------------------------------------------------------------------------
# read_learning_file：读取的边界条件（含穿越在工具层的拦截）
# ---------------------------------------------------------------------------

class TestReadLearningFile:
    """验证不存在、PDF、非 UTF-8、截断、穿越防护。"""

    def test_nonexistent_file(self, tmp_path: Path):
        """文件不存在应返回提示字符串，而非抛异常。"""
        result = read_learning_file.invoke(
            {"root_dir": str(tmp_path), "relative_path": "missing.txt"}
        )
        assert result == "文件不存在：missing.txt"

    def test_pdf_is_skipped(self, tmp_path: Path):
        """PDF 不直接读取，返回提示。"""
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
        result = read_learning_file.invoke(
            {"root_dir": str(tmp_path), "relative_path": "doc.pdf"}
        )
        assert "PDF" in result

    def test_non_utf8_gbk_falls_back(self, tmp_path: Path):
        """GBK 编码的中文文件不应崩溃，应走 errors=replace 回退。"""
        (tmp_path / "gbk.txt").write_bytes("你好，世界".encode("gbk"))
        result = read_learning_file.invoke(
            {"root_dir": str(tmp_path), "relative_path": "gbk.txt"}
        )
        # 不崩溃即通过；内容可能含替换字符 \ufffd
        assert isinstance(result, str)
        assert len(result) > 0

    def test_truncation_at_max_chars(self, tmp_path: Path):
        """文件超过 max_chars 时被截断。"""
        content = "A" * 500
        (tmp_path / "long.txt").write_text(content)
        result = read_learning_file.invoke(
            {"root_dir": str(tmp_path), "relative_path": "long.txt", "max_chars": 100}
        )
        assert len(result) == 100

    def test_capped_by_global_max(self, tmp_path: Path):
        """max_chars 超过全局上限时，仍被限制在 MAX_CHARS_PER_FILE。"""
        content = "B" * (MAX_CHARS_PER_FILE + 500)
        (tmp_path / "huge.txt").write_text(content)
        result = read_learning_file.invoke(
            {"root_dir": str(tmp_path), "relative_path": "huge.txt",
             "max_chars": MAX_CHARS_PER_FILE + 1000}
        )
        assert len(result) == MAX_CHARS_PER_FILE

    def test_traversal_blocked_at_tool_layer(self, tmp_path: Path):
        """穿越攻击应在工具层（经 _safe_path）被拦截，而非读出外部文件。"""
        (tmp_path / "target.txt").write_text("sensitive data")
        # 试图跳出 tmp_path 读上一级的 target.txt
        with pytest.raises(ValueError, match="不允许读取资料目录以外的文件"):
            read_learning_file.invoke(
                {"root_dir": str(tmp_path), "relative_path": "../target.txt"}
            )


# ---------------------------------------------------------------------------
# search_learning_files：检索的边界条件
# ---------------------------------------------------------------------------

class TestSearchLearningFiles:
    """验证无匹配、多匹配、数量上限。"""

    def test_no_match(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("hello world")
        result = search_learning_files.invoke(
            {"root_dir": str(tmp_path), "keyword": "不存在的词"}
        )
        assert result.startswith("未找到关键词")

    def test_multiple_matches(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("python is great\npython is fun")
        (tmp_path / "b.py").write_text("# python script\nprint('python')")
        result = search_learning_files.invoke(
            {"root_dir": str(tmp_path), "keyword": "python"}
        )
        assert "a.md" in result
        assert "b.py" in result
        assert len(result.splitlines()) == 4

    def test_capped_at_12_results(self, tmp_path: Path):
        (tmp_path / "big.txt").write_text("\n".join(f"keyword line {i}" for i in range(20)))
        result = search_learning_files.invoke(
            {"root_dir": str(tmp_path), "keyword": "keyword"}
        )
        assert len(result.splitlines()) == 12


# ---------------------------------------------------------------------------
# summarize_text_statistics：统计的边界条件
# ---------------------------------------------------------------------------

class TestSummarizeTextStatistics:
    """验证空串、纯中文、混合文本。"""

    def test_empty_string(self):
        result = summarize_text_statistics.invoke({"text": ""})
        assert "字符数：0" in result
        assert "高频词：无" in result

    def test_pure_chinese(self):
        result = summarize_text_statistics.invoke({"text": "学习学习计划计划"})
        assert "学习" in result  # 中文词被正则捕获

    def test_mixed_text(self):
        result = summarize_text_statistics.invoke({"text": "hello hello world 你好"})
        assert "hello" in result
