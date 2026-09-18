import pytest

from core.chunking import DocumentChunk, RecursiveTextChunker
from core.loaders import DocumentPage


@pytest.fixture
def chunker():
    """Provides a fresh RecursiveTextChunker instance for testing."""
    return RecursiveTextChunker(chunk_size=100, chunk_overlap=20)


def test_document_chunk_dataclass():
    """Verify DocumentChunk correctly stores its required provenance fields."""
    chunk = DocumentChunk(text="Test", source="test.pdf", page_number=5, chunk_index=0)
    assert chunk.text == "Test"
    assert chunk.source == "test.pdf"
    assert chunk.page_number == 5
    assert chunk.chunk_index == 0


def test_chunker_short_text(chunker):
    """Verify text much shorter than chunk_size produces exactly one chunk."""
    page = DocumentPage(text="Short text.", source="doc.pdf", page_number=1)
    chunks = chunker.chunk_page(page)

    assert len(chunks) == 1
    assert chunks[0].text == "Short text."
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_number == 1
    assert chunks[0].source == "doc.pdf"


def test_chunker_long_text_produces_multiple_chunks(chunker):
    """Verify text exceeding chunk_size is split into overlapping chunks."""
    long_text = "A" * 150
    page = DocumentPage(text=long_text, source="doc.pdf", page_number=1)
    chunks = chunker.chunk_page(page)

    assert len(chunks) > 1
    # Check index ordering
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    # Check size constraint
    assert len(chunks[0].text) <= 100


def test_chunker_overlap_behavior(chunker):
    """Verify chunks share the configured overlap."""
    # chunk_size=100, overlap=20
    text = "Word. " * 20  # Length 120
    page = DocumentPage(text=text, source="doc.pdf", page_number=1)
    chunks = chunker.chunk_page(page)

    assert len(chunks) == 2

    # With RecursiveCharacterTextSplitter, overlapping behavior preserves trailing/leading context.
    # The end of the first chunk should be found near the beginning of the second chunk.
    # We use a 20-character overlap buffer.
    overlap_snippet = chunks[0].text[-20:]
    assert overlap_snippet in chunks[1].text


def test_chunk_index_resets_for_new_page(chunker):
    """Verify chunk_index is localized to the page and resets on a new page."""
    page1 = DocumentPage(text="A" * 150, source="doc.pdf", page_number=1)
    page2 = DocumentPage(text="B" * 150, source="doc.pdf", page_number=2)

    chunks1 = chunker.chunk_page(page1)
    chunks2 = chunker.chunk_page(page2)

    assert len(chunks1) > 1
    assert chunks1[0].chunk_index == 0
    assert chunks1[1].chunk_index == 1

    assert len(chunks2) > 1
    assert chunks2[0].chunk_index == 0  # Crucial: Must reset to 0!


def test_empty_and_whitespace_text(chunker):
    """Verify empty and purely whitespace text returns an empty list."""
    empty_page = DocumentPage(text="", source="doc.pdf", page_number=1)
    whitespace_page = DocumentPage(text="   \n\t  ", source="doc.pdf", page_number=2)

    assert chunker.chunk_page(empty_page) == []
    assert chunker.chunk_page(whitespace_page) == []


def test_deterministic_execution(chunker):
    """Verify repeated execution on the same input yields mathematically identical output."""
    page = DocumentPage(
        text="Deterministic text execution." * 10, source="x.pdf", page_number=1
    )

    run1 = chunker.chunk_page(page)
    run2 = chunker.chunk_page(page)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.text == c2.text
        assert c1.chunk_index == c2.chunk_index
        assert c1.page_number == c2.page_number


def test_text_exactly_at_chunk_size(chunker):
    """Verify text perfectly hitting the chunk_size constraint."""
    exact_text = "A" * 100
    page = DocumentPage(text=exact_text, source="doc.pdf", page_number=1)
    chunks = chunker.chunk_page(page)

    assert len(chunks) == 1
    assert chunks[0].text == exact_text
