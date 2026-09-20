from unittest.mock import MagicMock, call

import pytest

from core.chunking import DocumentChunk
from core.ingestion import DocumentIngestionService
from core.loaders import DocumentPage


@pytest.fixture
def mocks():
    """Provides isolated mocks for all injected dependencies."""
    loader = MagicMock()
    chunker = MagicMock()
    embedder = MagicMock()
    vector_store = MagicMock()
    return loader, chunker, embedder, vector_store


def test_invalid_batch_size(mocks):
    """Verify batch_size validation prevents negative or zero sizes."""
    loader, chunker, embedder, vector_store = mocks
    with pytest.raises(ValueError, match="strictly greater than zero"):
        DocumentIngestionService(loader, chunker, embedder, vector_store, batch_size=0)
    with pytest.raises(ValueError, match="strictly greater than zero"):
        DocumentIngestionService(loader, chunker, embedder, vector_store, batch_size=-1)


def test_empty_document(mocks):
    """Verify ingestion of a document with zero pages handles safely."""
    loader, chunker, embedder, vector_store = mocks
    loader.load.return_value = iter([])

    service = DocumentIngestionService(loader, chunker, embedder, vector_store)
    result = service.ingest("test-doc-id")

    assert result.total_pages == 0
    assert result.total_chunks == 0
    assert result.source == "unknown"

    chunker.chunk_page.assert_not_called()
    embedder.embed_texts.assert_not_called()
    vector_store.add.assert_not_called()


def test_single_page_single_chunk(mocks):
    """Verify standard flow for a basic 1-page, 1-chunk document."""
    loader, chunker, embedder, vector_store = mocks

    page = DocumentPage(text="Page 1", source="test.pdf", page_number=1)
    chunk = DocumentChunk(
        text="Chunk 1", source="test.pdf", page_number=1, chunk_index=0
    )

    loader.load.return_value = iter([page])
    chunker.chunk_page.return_value = [chunk]
    embedder.embed_texts.return_value = [[0.1, 0.2]]

    service = DocumentIngestionService(loader, chunker, embedder, vector_store)
    result = service.ingest("test-doc-id")

    assert result.total_pages == 1
    assert result.total_chunks == 1
    assert result.source == "test.pdf"

    chunker.chunk_page.assert_called_once_with(page)
    embedder.embed_texts.assert_called_once_with(["Chunk 1"])

    vector_store.add.assert_called_once()
    args, kwargs = vector_store.add.call_args
    assert kwargs["texts"] == ["Chunk 1"]
    assert kwargs["embeddings"] == [[0.1, 0.2]]
    assert len(kwargs["ids"]) == 1
    assert kwargs["metadatas"] == [
        {"source": "test.pdf", "page_number": 1, "chunk_index": 0, "document_id": "test-doc-id"}
    ]


def test_batch_size_logic(mocks):
    """Verify batch size 2 with 5 chunks splits into batches of 2, 2, 1."""
    loader, chunker, embedder, vector_store = mocks

    page = DocumentPage(text="Content", source="batch.pdf", page_number=1)
    loader.load.return_value = iter([page])

    # 5 chunks
    chunks = [
        DocumentChunk(text=f"C{i}", source="batch.pdf", page_number=1, chunk_index=i)
        for i in range(5)
    ]
    chunker.chunk_page.return_value = chunks

    embedder.embed_texts.return_value = [[0.1]] * 2

    service = DocumentIngestionService(
        loader, chunker, embedder, vector_store, batch_size=2
    )
    result = service.ingest("test-doc-id")

    assert result.total_pages == 1
    assert result.total_chunks == 5

    # embedder and vector_store should be called 3 times (sizes: 2, 2, 1)
    assert embedder.embed_texts.call_count == 3
    assert vector_store.add.call_count == 3

    # Check the exact text batches sent to embedder
    embedder.embed_texts.assert_has_calls(
        [
            call(["C0", "C1"]),
            call(["C2", "C3"]),
            call(["C4"]),
        ]
    )


def test_multiple_pages_multiple_chunks(mocks):
    """Verify processing multiple pages containing multiple chunks."""
    loader, chunker, embedder, vector_store = mocks

    page1 = DocumentPage(text="P1", source="multi.pdf", page_number=1)
    page2 = DocumentPage(text="P2", source="multi.pdf", page_number=2)

    loader.load.return_value = iter([page1, page2])

    def mock_chunk_page(page):
        if page.page_number == 1:
            return [
                DocumentChunk(
                    text="P1C0", source="multi.pdf", page_number=1, chunk_index=0
                )
            ]
        else:
            return [
                DocumentChunk(
                    text="P2C0", source="multi.pdf", page_number=2, chunk_index=0
                ),
                DocumentChunk(
                    text="P2C1", source="multi.pdf", page_number=2, chunk_index=1
                ),
            ]

    chunker.chunk_page.side_effect = mock_chunk_page
    embedder.embed_texts.return_value = [[0.0]] * 3

    service = DocumentIngestionService(
        loader, chunker, embedder, vector_store, batch_size=10
    )
    result = service.ingest("test-doc-id")

    assert result.total_pages == 2
    assert result.total_chunks == 3
    assert result.source == "multi.pdf"

    # All processed in 1 batch
    vector_store.add.assert_called_once()
    kwargs = vector_store.add.call_args[1]
    assert kwargs["texts"] == ["P1C0", "P2C0", "P2C1"]

    meta = kwargs["metadatas"]
    assert meta[0] == {"source": "multi.pdf", "page_number": 1, "chunk_index": 0, "document_id": "test-doc-id"}
    assert meta[1] == {"source": "multi.pdf", "page_number": 2, "chunk_index": 0, "document_id": "test-doc-id"}
    assert meta[2] == {"source": "multi.pdf", "page_number": 2, "chunk_index": 1, "document_id": "test-doc-id"}


def test_propagation_of_loader_errors(mocks):
    """Verify loader exceptions are safely bubbled up without swallowing."""
    loader, chunker, embedder, vector_store = mocks
    loader.load.side_effect = Exception("Loader crashed")

    service = DocumentIngestionService(loader, chunker, embedder, vector_store)

    with pytest.raises(Exception, match="Loader crashed"):
        service.ingest("test-doc-id")


def test_propagation_of_embedding_errors(mocks):
    """Verify embedding exceptions bubble up to the caller."""
    loader, chunker, embedder, vector_store = mocks

    page = DocumentPage(text="Content", source="err.pdf", page_number=1)
    loader.load.return_value = iter([page])
    chunker.chunk_page.return_value = [DocumentChunk("C", "err.pdf", 1, 0)]

    embedder.embed_texts.side_effect = Exception("Embedder crashed")

    service = DocumentIngestionService(loader, chunker, embedder, vector_store)

    with pytest.raises(Exception, match="Embedder crashed"):
        service.ingest("test-doc-id")


def test_propagation_of_vectorstore_errors(mocks):
    """Verify database exceptions bubble up to the caller."""
    loader, chunker, embedder, vector_store = mocks

    page = DocumentPage(text="Content", source="err.pdf", page_number=1)
    loader.load.return_value = iter([page])
    chunker.chunk_page.return_value = [DocumentChunk("C", "err.pdf", 1, 0)]
    embedder.embed_texts.return_value = [[0.1]]

    vector_store.add.side_effect = Exception("DB offline")

    service = DocumentIngestionService(loader, chunker, embedder, vector_store)

    with pytest.raises(Exception, match="DB offline"):
        service.ingest("test-doc-id")
