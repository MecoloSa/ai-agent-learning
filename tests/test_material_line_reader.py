"""受限资料按行读取测试。

使用临时文件验证范围、来源、截断和目录边界。
不调用模型，不解析 PDF，不执行 Python 资料。
"""

import pytest

import ai_agent.materials as materials_module
from ai_agent.materials import read_material_lines


@pytest.mark.parametrize("suffix", [".txt", ".md", ".py"])
def test_reads_requested_lines_without_modifying_file(tmp_path, suffix):
    path = tmp_path / f"notes{suffix}"
    original = "第一行\n第二行\n第三行\n第四行\n"
    path.write_text(original, encoding="utf-8")

    result = read_material_lines(
        str(tmp_path), path.name, 2, 3,
    )

    assert result["source"] == path.name
    assert result["returned_start_line"] == 2
    assert result["returned_end_line"] == 3
    assert result["truncated"] is False
    assert result["lines"] == [
        {"line": 2, "text": "第二行", "partial": False},
        {"line": 3, "text": "第三行", "partial": False},
    ]
    assert path.read_text(encoding="utf-8") == original

    print(f"\n读取 {path.name} 的第2～3行：{result['lines']}")


@pytest.mark.parametrize(
    "start, end",
    [
        (0, 2),
        (3, 2),
        (1, 201),
        (True, 2),
        (1.5, 2),
        ("1", 2),
    ],
)
def test_invalid_ranges_are_rejected(tmp_path, start, end):
    with pytest.raises(ValueError):
        read_material_lines(str(tmp_path), "notes.md", start, end)


def test_returned_range_does_not_invent_missing_lines(tmp_path):
    (tmp_path / "notes.md").write_text(
        "第一行\n第二行\n", encoding="utf-8",
    )

    result = read_material_lines(str(tmp_path), "notes.md", 1, 20)

    assert result["requested_end_line"] == 20
    assert result["returned_end_line"] == 2
    assert len(result["lines"]) == 2


def test_start_beyond_file_is_rejected(tmp_path):
    (tmp_path / "notes.md").write_text("只有一行", encoding="utf-8")

    with pytest.raises(ValueError, match="起始行超出"):
        read_material_lines(str(tmp_path), "notes.md", 5, 6)


def test_path_outside_material_root_is_rejected(tmp_path):
    root = tmp_path / "materials"
    root.mkdir()
    (tmp_path / "outside.txt").write_text(
        "不能读取", encoding="utf-8",
    )

    with pytest.raises(ValueError, match="目录以外"):
        read_material_lines(str(root), "../outside.txt", 1, 1)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_material_lines(str(tmp_path), "missing.md", 1, 1)


def test_pdf_is_not_treated_as_line_based_text(tmp_path):
    # 只验证类型拒绝，不创建或解析真实 PDF。
    with pytest.raises(ValueError, match="PDF"):
        read_material_lines(str(tmp_path), "book.pdf", 1, 2)


def test_character_limit_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(materials_module, "MAX_READ_TEXT_CHARS", 5)
    (tmp_path / "notes.txt").write_text(
        "abcdefghij\n第二行\n", encoding="utf-8",
    )

    result = read_material_lines(str(tmp_path), "notes.txt", 1, 2)

    assert result["truncated"] is True
    assert result["lines"] == [
        {"line": 1, "text": "abcde", "partial": True},
    ]

    print(f"\n字符截断结果：{result}")


def test_python_material_is_read_not_executed(tmp_path):
    text = 'raise RuntimeError("这段资料不得执行")\n'
    (tmp_path / "example.py").write_text(text, encoding="utf-8")

    result = read_material_lines(str(tmp_path), "example.py", 1, 1)

    assert result["lines"][0]["text"] == text.rstrip("\n")