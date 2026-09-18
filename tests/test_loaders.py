import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pypdf.errors import PdfReadError

from core.loaders import DocumentLoadError, DocumentPage, PdfDocumentLoader


def test_document_page_dataclass():
    """Verify DocumentPage appropriately stores the representation."""
    page = DocumentPage(text="Hello", source="test.pdf", page_number=1)
    assert page.text == "Hello"
    assert page.source == "test.pdf"
    assert page.page_number == 1


@patch("core.loaders.PdfReader")
def test_pdf_loader_valid_document(mock_reader_class):
    """Verify loader correctly yields multiple pages incrementally."""
    mock_reader = MagicMock()
    mock_reader.is_encrypted = False

    page1, page2, page3 = MagicMock(), MagicMock(), MagicMock()
    page1.extract_text.return_value = "Page 1 text"
    page2.extract_text.return_value = "Page 2 text"
    page3.extract_text.return_value = "Page 3 text"

    mock_reader.pages = [page1, page2, page3]
    mock_reader_class.return_value = mock_reader

    file_obj = io.BytesIO(b"fake pdf bytes")
    loader = PdfDocumentLoader(file=file_obj, source="test.pdf")

    # The loader should be an iterator
    page_iterator = loader.load()
    pages = list(page_iterator)

    assert len(pages) == 3
    assert pages[0].page_number == 1
    assert pages[0].text == "Page 1 text"
    assert pages[0].source == "test.pdf"

    assert pages[1].page_number == 2
    assert pages[1].text == "Page 2 text"

    assert pages[2].page_number == 3
    assert pages[2].text == "Page 3 text"


@patch("core.loaders.PdfReader")
def test_pdf_loader_skips_empty_and_whitespace(mock_reader_class):
    """Verify loader correctly skips empty and whitespace-only pages."""
    mock_reader = MagicMock()
    mock_reader.is_encrypted = False

    page1, page2, page3, page4, page5 = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    page1.extract_text.return_value = "Valid text"
    page2.extract_text.return_value = ""  # Empty string
    page3.extract_text.return_value = "   \n\t  "  # Whitespace only
    page4.extract_text.return_value = None  # None returned
    page5.extract_text.return_value = "Valid text 2"

    mock_reader.pages = [page1, page2, page3, page4, page5]
    mock_reader_class.return_value = mock_reader

    loader = PdfDocumentLoader(file=io.BytesIO(b""), source="test.pdf")
    pages = list(loader.load())

    # Should only return physical pages 1 and 5
    assert len(pages) == 2

    assert pages[0].page_number == 1
    assert pages[0].text == "Valid text"

    assert pages[1].page_number == 5
    assert pages[1].text == "Valid text 2"


@patch("core.loaders.PdfReader")
def test_pdf_loader_path_input(mock_reader_class, tmp_path):
    """Verify loader cleanly accepts local Path objects and reads them."""
    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    page1 = MagicMock()
    page1.extract_text.return_value = "Path test"
    mock_reader.pages = [page1]
    mock_reader_class.return_value = mock_reader

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"dummy pdf bytes")

    loader = PdfDocumentLoader(file=pdf_path, source="local.pdf")
    pages = list(loader.load())

    assert len(pages) == 1
    assert pages[0].text == "Path test"
    assert pages[0].source == "local.pdf"


def test_pdf_loader_invalid_path():
    """Verify missing files are correctly wrapped in a DocumentLoadError."""
    loader = PdfDocumentLoader(file=Path("/does/not/exist.pdf"), source="missing.pdf")
    with pytest.raises(DocumentLoadError, match="Could not open file"):
        list(loader.load())


@patch("core.loaders.PdfReader")
def test_pdf_loader_read_error(mock_reader_class):
    """Verify pypdf exceptions are safely wrapped in DocumentLoadError."""
    mock_reader_class.side_effect = PdfReadError("Malformed PDF")

    loader = PdfDocumentLoader(file=io.BytesIO(b"bad bytes"), source="bad.pdf")
    with pytest.raises(DocumentLoadError, match="Failed to read PDF 'bad.pdf'"):
        list(loader.load())


@patch("core.loaders.PdfReader")
def test_pdf_loader_encrypted_pdf(mock_reader_class):
    """Verify encrypted PDFs are explicitly rejected with a clear message."""
    mock_reader = MagicMock()
    mock_reader.is_encrypted = True
    mock_reader_class.return_value = mock_reader

    loader = PdfDocumentLoader(file=io.BytesIO(b"encrypted"), source="encrypted.pdf")
    with pytest.raises(
        DocumentLoadError, match="Cannot read encrypted PDF: encrypted.pdf"
    ):
        list(loader.load())


@patch("core.loaders.PdfReader")
def test_pdf_loader_page_extraction_failure_continues(mock_reader_class, caplog):
    """Verify that failing to extract a single page allows the loader to salvage the rest."""
    mock_reader = MagicMock()
    mock_reader.is_encrypted = False

    page1, page2, page3 = MagicMock(), MagicMock(), MagicMock()
    page1.extract_text.return_value = "Page 1"
    page2.extract_text.side_effect = Exception("Extraction crashed")
    page3.extract_text.return_value = "Page 3"

    mock_reader.pages = [page1, page2, page3]
    mock_reader_class.return_value = mock_reader

    loader = PdfDocumentLoader(file=io.BytesIO(b""), source="error.pdf")
    pages = list(loader.load())

    # Should yield physical page 1 and 3, skip 2, and log a warning
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 3

    assert "Failed to extract text from error.pdf, page 2" in caplog.text


def test_pdf_document_loader_real_pdf_extraction():
    """
    Genuine integration test that DOES NOT mock PdfReader.
    This guarantees that the actual installed pypdf package successfully reads
    a deterministic valid PDF byte stream through our loader logic.
    """
    import logging

    # A minimal valid PDF 1.4 byte string containing the text 'Hello World'
    minimal_pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 << "
        b"/Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
        b"/MediaBox [0 0 100 100] /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 10 10 Td (Hello World) Tj ET\n"
        b"endstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000060 00000 n \n"
        b"0000000119 00000 n \n0000000298 00000 n \n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n392\n%%EOF\n"
    )

    file_obj = io.BytesIO(minimal_pdf_bytes)

    # Suppress the 'incorrect startxref' pypdf stderr warning for test cleanliness
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    loader = PdfDocumentLoader(file=file_obj, source="real_integration.pdf")
    pages = list(loader.load())

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].source == "real_integration.pdf"
    assert "Hello World" in pages[0].text
