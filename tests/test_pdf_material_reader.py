"""PDF指定页读取的离线测试。

用假解析器提供可控页面，验证页码、截断及失败处理。
这些测试不证明真实PDF的文字提取质量；真实文件需要另行抽查。
不调用LLM，不写入真实学习状态。
"""

import json
from types import SimpleNamespace

import pytest

import ai_agent.resources.pdf_materials as pdf_module
from ai_agent.resources.tools import build_read_pdf_pages_tool
from ai_agent.runtime.execution_guard import execution_failure_reason, tool_result_succeeded


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


def test_out_of_bounds_receipt_is_a_failure_and_extracts_no_text(monkeypatch, tmp_path):
    pages = mock_pdf(monkeypatch, tmp_path, ["第一页", "第二页"])
    tool = build_read_pdf_pages_tool(str(tmp_path))
    raw = tool.invoke({"relative_path": "book.pdf", "start_page": 5, "end_page": 5})
    assert json.loads(raw) == {
        "ok": False,
        "error": {"code": "pdf_page_out_of_bounds", "message": "起始页超出PDF范围",
                  "requested_start_page": 5, "total_pages": 2, "retryable": False},
    }
    assert not tool_result_succeeded("read_pdf_pages", raw)
    assert pages == []
    assert json.loads(tool.invoke({"relative_path": "book.pdf", "start_page": 2, "end_page": 2}))["ok"]
    assert pages == [2]


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


def test_pdf_tool_returns_actionable_range_rejection(tmp_path):
    reader = build_read_pdf_pages_tool(str(tmp_path))
    result = reader.invoke({"relative_path": "book.pdf", "start_page": 8, "end_page": 12})
    error = json.loads(result)["error"]
    assert error["code"] == "pdf_page_range_limit"
    assert (error["requested_start_page"], error["requested_end_page"], error["max_pages"]) == (8, 12, 3)
    assert error["retryable"] is False
    assert not tool_result_succeeded("read_pdf_pages", result)
    assert execution_failure_reason("read_pdf_pages", result) == "请求页数超过单次上限，请缩小范围。"
