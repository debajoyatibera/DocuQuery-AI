from typing import Any, Collection, Dict, List, Optional

import pytest

from core.embeddings import EmbeddingProvider
from core.retrieval import RetrievedChunk, VectorStoreRetriever
from core.vectorstore import VectorStore


class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self.embed_calls = 0

    @property
    def dimension(self) -> int:
        return 3

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Just return a dummy embedding for each text
        self.embed_calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]


class MockVectorStore(VectorStore):
    def __init__(self, mock_results=None):
        self.mock_results = mock_results if mock_results is not None else []
        self.last_query_embedding = None
        self.last_n_results = None
        self.last_document_ids = None

    def add(self, texts, embeddings, metadatas, ids):
        pass

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        document_ids: Optional[Collection[str]] = None,
    ) -> List[Dict[str, Any]]:
        self.last_query_embedding = query_embedding
        self.last_n_results = n_results
        self.last_document_ids = document_ids
        return self.mock_results


def test_successful_retrieval():
    mock_results = [
        {
            "id": "1",
            "text": "Hello world",
            "metadata": {"source": "doc1.pdf", "page_number": 1, "chunk_index": 0},
            "distance": 0.1,
        }
    ]
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore(mock_results)
    retriever = VectorStoreRetriever(embedding_provider, vector_store)

    results = retriever.retrieve("test query")

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].text == "Hello world"
    assert results[0].source == "doc1.pdf"
    assert results[0].page_number == 1
    assert results[0].chunk_index == 0
    assert results[0].distance == 0.1

    # Check that query embedding was passed correctly
    assert vector_store.last_query_embedding == [0.1, 0.2, 0.3]
    assert vector_store.last_n_results == 4  # Default k=4


def test_custom_k():
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore([])
    retriever = VectorStoreRetriever(embedding_provider, vector_store)

    retriever.retrieve("test query", k=10)
    assert vector_store.last_n_results == 10


def test_document_ids_are_forwarded_to_vector_store():
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore([])
    retriever = VectorStoreRetriever(embedding_provider, vector_store)

    document_ids = ["doc-a", "doc-b"]
    retriever.retrieve("test query", document_ids=document_ids)

    assert vector_store.last_document_ids is document_ids


def test_empty_document_ids_return_without_embedding():
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore([])
    retriever = VectorStoreRetriever(embedding_provider, vector_store)

    results = retriever.retrieve("test query", document_ids=[])

    assert results == []
    assert embedding_provider.embed_calls == 0
    assert vector_store.last_query_embedding is None


def test_empty_search_result_returns_empty_list():
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore([])
    retriever = VectorStoreRetriever(embedding_provider, vector_store)

    results = retriever.retrieve("test query")
    assert results == []


def test_multiple_retrieved_results_preserve_ordering():
    mock_results = [
        {
            "id": "1",
            "text": "first",
            "metadata": {"source": "doc1.pdf", "chunk_index": 0},
            "distance": 0.1,
        },
        {
            "id": "2",
            "text": "second",
            "metadata": {"source": "doc1.pdf", "chunk_index": 1},
            "distance": 0.2,
        },
    ]
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore(mock_results)
    retriever = VectorStoreRetriever(embedding_provider, vector_store)

    results = retriever.retrieve("test query")
    assert len(results) == 2
    assert results[0].text == "first"
    assert results[1].text == "second"


def test_empty_query_raises_value_error():
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore()
    retriever = VectorStoreRetriever(embedding_provider, vector_store)

    with pytest.raises(ValueError, match="cannot be empty or whitespace"):
        retriever.retrieve("")


def test_whitespace_only_query_raises_value_error():
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore()
    retriever = VectorStoreRetriever(embedding_provider, vector_store)

    with pytest.raises(ValueError, match="cannot be empty or whitespace"):
        retriever.retrieve("   \n \t  ")


def test_surrounding_query_whitespace_is_stripped():
    vector_store = MockVectorStore([])

    # Wrap the mock embedding provider to check the text passed to it
    class CheckWhitespaceEmbeddingProvider(MockEmbeddingProvider):
        def embed_texts(self, texts: List[str]) -> List[List[float]]:
            assert texts == ["clean query"]
            return super().embed_texts(texts)

    retriever = VectorStoreRetriever(CheckWhitespaceEmbeddingProvider(), vector_store)
    retriever.retrieve("   clean query  \n")


def test_invalid_k_raises_value_error():
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore()
    retriever = VectorStoreRetriever(embedding_provider, vector_store)

    with pytest.raises(ValueError, match="k must be greater than zero"):
        retriever.retrieve("test", k=0)

    with pytest.raises(ValueError, match="k must be greater than zero"):
        retriever.retrieve("test", k=-1)


def test_missing_required_metadata_raises_value_error():
    mock_results = [
        {
            "id": "1",
            "text": "first",
            "metadata": {"page_number": 1},  # Missing source and chunk_index
            "distance": 0.1,
        }
    ]
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore(mock_results)
    retriever = VectorStoreRetriever(embedding_provider, vector_store)

    with pytest.raises(ValueError, match="Missing required metadata keys"):
        retriever.retrieve("test query")
