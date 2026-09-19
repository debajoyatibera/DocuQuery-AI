from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from core.embeddings import EmbeddingProvider
from core.vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page_number: Optional[int]
    chunk_index: int
    distance: float


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        """Retrieve top k chunks for the given query."""
        pass


class VectorStoreRetriever(Retriever):
    def __init__(
        self, embedding_provider: EmbeddingProvider, vector_store: VectorStore
    ):
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def retrieve(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        query_stripped = query.strip()
        if not query_stripped:
            raise ValueError("Query cannot be empty or whitespace.")
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        # Embed query
        embeddings = self._embedding_provider.embed_texts([query_stripped])
        query_embedding = embeddings[0]

        # Search Vector Store
        raw_results = self._vector_store.search(
            query_embedding=query_embedding, n_results=k
        )

        # Map results
        retrieved_chunks = []
        for res in raw_results:
            metadata = res.get("metadata", {})
            if "source" not in metadata or "chunk_index" not in metadata:
                raise ValueError(
                    "Missing required metadata keys: 'source' or 'chunk_index'."
                )

            chunk = RetrievedChunk(
                text=res.get("text", ""),
                source=metadata["source"],
                page_number=metadata.get("page_number"),
                chunk_index=metadata["chunk_index"],
                distance=res.get("distance", 0.0),
            )
            retrieved_chunks.append(chunk)

        return retrieved_chunks
