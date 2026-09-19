import pytest

from core.context import SimpleContextBuilder
from core.retrieval import RetrievedChunk


def test_empty_input():
    builder = SimpleContextBuilder()
    result = builder.build([])
    assert result.formatted_text == ""
    assert result.used_chunks == []


def test_one_chunk():
    builder = SimpleContextBuilder()
    chunk = RetrievedChunk(
        text="This is chunk one.",
        source="source.pdf",
        page_number=5,
        chunk_index=0,
        distance=0.1,
    )
    result = builder.build([chunk])
    expected_text = "Document: source.pdf (Page 5)\nThis is chunk one."
    assert result.formatted_text == expected_text
    assert result.used_chunks == [chunk]


def test_multiple_chunks():
    builder = SimpleContextBuilder()
    chunk1 = RetrievedChunk(
        text="Chunk 1.", source="doc.pdf", page_number=1, chunk_index=0, distance=0.1
    )
    chunk2 = RetrievedChunk(
        text="Chunk 2.", source="doc.pdf", page_number=1, chunk_index=1, distance=0.2
    )
    result = builder.build([chunk1, chunk2])
    expected_text = (
        "Document: doc.pdf (Page 1)\nChunk 1.\n\nDocument: doc.pdf (Page 1)\nChunk 2."
    )
    assert result.formatted_text == expected_text
    assert result.used_chunks == [chunk1, chunk2]


def test_ordering_preservation():
    builder = SimpleContextBuilder()
    chunk1 = RetrievedChunk(
        text="Chunk A", source="doc.pdf", page_number=1, chunk_index=0, distance=0.1
    )
    chunk2 = RetrievedChunk(
        text="Chunk B", source="doc.pdf", page_number=1, chunk_index=1, distance=0.0
    )
    result = builder.build([chunk2, chunk1])  # Notice different order
    assert result.used_chunks == [chunk2, chunk1]


def test_duplicate_removal_keeps_first():
    builder = SimpleContextBuilder()
    chunk1 = RetrievedChunk(
        text="Chunk A", source="doc.pdf", page_number=1, chunk_index=0, distance=0.1
    )
    chunk2 = RetrievedChunk(
        text="Chunk B", source="doc.pdf", page_number=1, chunk_index=0, distance=0.2
    )  # Same key!
    result = builder.build([chunk1, chunk2])
    assert result.used_chunks == [chunk1]
    assert "Chunk A" in result.formatted_text
    assert "Chunk B" not in result.formatted_text


def test_same_text_different_provenance_not_duplicate():
    builder = SimpleContextBuilder()
    chunk1 = RetrievedChunk(
        text="Same text", source="doc1.pdf", page_number=1, chunk_index=0, distance=0.1
    )
    chunk2 = RetrievedChunk(
        text="Same text", source="doc2.pdf", page_number=1, chunk_index=0, distance=0.1
    )
    result = builder.build([chunk1, chunk2])
    assert result.used_chunks == [chunk1, chunk2]


def test_max_chars_validation():
    builder = SimpleContextBuilder()
    with pytest.raises(ValueError, match="max_chars must be greater than 0"):
        builder.build([], max_chars=0)
    with pytest.raises(ValueError, match="max_chars must be greater than 0"):
        builder.build([], max_chars=-5)


def test_chunk_exceeding_limit_is_not_partially_truncated():
    builder = SimpleContextBuilder()
    chunk1 = RetrievedChunk(
        text="1234567890", source="doc.pdf", page_number=1, chunk_index=0, distance=0.1
    )
    # Format will be: "Document: doc.pdf (Page 1)\n1234567890" => len = 37
    result = builder.build([chunk1], max_chars=36)
    assert result.formatted_text == ""
    assert result.used_chunks == []


def test_once_chunk_exceeds_budget_later_chunks_not_added():
    builder = SimpleContextBuilder()
    chunk1 = RetrievedChunk(
        text="A", source="d.pdf", page_number=1, chunk_index=0, distance=0.1
    )  # len=25
    chunk2 = RetrievedChunk(
        text="B" * 50, source="d.pdf", page_number=1, chunk_index=1, distance=0.1
    )  # len=74
    chunk3 = RetrievedChunk(
        text="C", source="d.pdf", page_number=1, chunk_index=2, distance=0.1
    )  # len=25
    result = builder.build([chunk1, chunk2, chunk3], max_chars=60)

    # After chunk 1 is added, current_length = 25.
    # We evaluate chunk 2. It requires 2 (for \n\n) + 74 = 76. Total would be 101 > 60.
    # The loop should break.
    assert result.used_chunks == [chunk1]


def test_missing_page_number_formatting():
    builder = SimpleContextBuilder()
    chunk = RetrievedChunk(
        text="No page.",
        source="source.txt",
        page_number=None,
        chunk_index=0,
        distance=0.1,
    )
    result = builder.build([chunk])
    expected_text = "Document: source.txt\nNo page."
    assert result.formatted_text == expected_text


def test_exact_fit():
    builder = SimpleContextBuilder()
    chunk = RetrievedChunk(
        text="A", source="doc.pdf", page_number=1, chunk_index=0, distance=0.1
    )
    # Format: Document: doc.pdf (Page 1)\nA -> len = 28
    result = builder.build([chunk], max_chars=28)
    assert len(result.used_chunks) == 1
    assert len(result.formatted_text) == 28
