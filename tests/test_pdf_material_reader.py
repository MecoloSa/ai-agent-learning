"""PDF指定页读取的离线测试。

用假解析器提供可控页面，验证页码、截断及失败处理。
这些测试不证明真实PDF的文字提取质量；真实文件需要另行抽查。
不调用LLM，不写入真实学习状态。
"""

from types import SimpleNamespace

import pytest

import ai_agent.pdf_materials as pdf_module


def mock_pdf(monkeypatch, tmp_path, texts, *, encrypted=False):
    """建立临时路径，用假解析器隔离PDF格式本身的不确定性。"""
    (tmp_path / "book.pdf").write_bytes(b"mock PDF")
    extracted_pages = []

    class FakePage:
        def __init__(self, number, text):
            self.number = number
            self.text = text

        def extract_text(self):
            extracted_pages.append(self.number)
            if isinstance(self.text, Exception):
                raise self.text
            return self.text

    reader = SimpleNamespace(
        is_encrypted=encrypted,
        pages=[
            FakePage(number, text)
            for number, text in enumerate(texts, start=1)
        ],
    )
    monkeypatch.setattr(
        pdf_module,
        "PdfReader",
        lambda *args, **kwargs: reader,
    )
    return extracted_pages


def test_reads_only_requested_physical_pages(monkeypatch, tmp_path):
    extracted = mock_pdf(
        monkeypatch, tmp_path,
        ["第一页", "第二页", "第三页"],
    )

    result = pdf_module.read_pdf_pages(
        str(tmp_path), "book.pdf", 2, 3,
    )

    assert extracted == [2, 3]
    assert result["returned_start_page"] == 2
    assert result["returned_end_page"] == 3
    assert result["pages"][0]["text"] == "第二页"
    assert result["pages"][0]["citation"] == "[source=book.pdf, page=2]"
    assert result["truncated"] is False


def test_document_end_is_not_budget_truncation(monkeypatch, tmp_path):
    mock_pdf(monkeypatch, tmp_path, ["第一页", "第二页", "第三页"])

    result = pdf_module.read_pdf_pages(
        str(tmp_path), "book.pdf", 2, 4,
    )

    assert result["requested_end_page"] == 4
    assert result["returned_end_page"] == 3
    assert result["end_clipped_to_document"] is True
    assert result["truncated"] is False


def test_character_budget_marks_partial_page(monkeypatch, tmp_path):
    extracted = mock_pdf(monkeypatch, tmp_path, ["ABCDEFG", "第二页"])
    monkeypatch.setattr(pdf_module, "MAX_READ_PDF_TEXT_CHARS", 4)

    result = pdf_module.read_pdf_pages(
        str(tmp_path), "book.pdf", 1, 2,
    )

    assert extracted == [1]
    assert result["pages"][0]["text"] == "ABCD"
    assert result["pages"][0]["partial"] is True
    assert result["truncated"] is True
    assert result["returned_end_page"] == 1


def test_no_text_page_is_reported_not_invented(monkeypatch, tmp_path):
    mock_pdf(monkeypatch, tmp_path, [None])

    result = pdf_module.read_pdf_pages(
        str(tmp_path), "book.pdf", 1, 1,
    )

    assert result["pages"][0]["text"] == ""
    assert result["pages"][0]["text_status"] == "no_text"
    assert result["pages"][0]["partial"] is False


@pytest.mark.parametrize(
    "start, end",
    [
        (True, 1),
        ("1", 1),
        (1.5, 2),
        (0, 1),
        (2, 1),
        (1, 4),  # 超过单次3页上限。
        (4, 4),  # 超过文档范围。
    ],
)
def test_rejects_invalid_page_range(monkeypatch, tmp_path, start, end):
    mock_pdf(monkeypatch, tmp_path, ["第一页", "第二页", "第三页"])

    with pytest.raises(ValueError):
        pdf_module.read_pdf_pages(
            str(tmp_path), "book.pdf", start, end,
        )


def test_rejects_outside_root(tmp_path):
    root = tmp_path / "materials"
    root.mkdir()
    (tmp_path / "outside.pdf").write_bytes(b"outside")

    with pytest.raises(ValueError):
        pdf_module.read_pdf_pages(
            str(root), "../outside.pdf", 1, 1,
        )


def test_rejects_missing_pdf(tmp_path):
    with pytest.raises(FileNotFoundError):
        pdf_module.read_pdf_pages(
            str(tmp_path), "missing.pdf", 1, 1,
        )


def test_rejects_non_pdf(tmp_path):
    (tmp_path / "notes.md").write_text("资料", encoding="utf-8")

    with pytest.raises(ValueError):
        pdf_module.read_pdf_pages(
            str(tmp_path), "notes.md", 1, 1,
        )


def test_rejects_encrypted_pdf(monkeypatch, tmp_path):
    mock_pdf(monkeypatch, tmp_path, ["第一页"], encrypted=True)

    with pytest.raises(ValueError, match="加密"):
        pdf_module.read_pdf_pages(
            str(tmp_path), "book.pdf", 1, 1,
        )


def test_page_failure_is_not_silently_skipped(monkeypatch, tmp_path):
    mock_pdf(
        monkeypatch, tmp_path,
        ["第一页", RuntimeError("模拟提取失败")],
    )

    with pytest.raises(ValueError, match="第2页"):
        pdf_module.read_pdf_pages(
            str(tmp_path), "book.pdf", 1, 2,
        )


def test_rejects_unparseable_pdf(monkeypatch, tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"not a PDF")

    def broken_reader(*args, **kwargs):
        raise RuntimeError("模拟解析失败")

    monkeypatch.setattr(pdf_module, "PdfReader", broken_reader)

    with pytest.raises(ValueError, match="无法解析"):
        pdf_module.read_pdf_pages(
            str(tmp_path), "broken.pdf", 1, 1,
        )