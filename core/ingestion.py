import logging
from dataclasses import dataclass
from typing import List

from core.chunking import DocumentChunk, TextChunker
from core.embeddings import EmbeddingProvider
from core.loaders import DocumentLoader
from core.vectorstore import VectorStore, generate_chunk_id

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Summary of a completed document ingestion process."""

    document_id: str
    source: str
    total_pages: int
    total_chunks: int


class DocumentIngestionService:
    """
    Orchestrates the RAG document ingestion pipeline.
    Coordinates the loader, chunker, embedder, and vector store without
    being coupled to their concrete implementations.
    """

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        batch_size: int = 32,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be strictly greater than zero.")

        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.batch_size = batch_size

    def ingest(self, document_id: str) -> IngestionResult:
        """
        Executes the ingestion pipeline.
        Yields pages, chunks them, and processes chunks in batches to optimize
        embedding and storage operations.
        """
        if not document_id or not document_id.strip():
            raise ValueError("document_id cannot be empty or whitespace only.")

        total_pages = 0
        total_chunks = 0
        source = "unknown"

        current_batch: List[DocumentChunk] = []

        # The loader might raise DocumentLoadError if the file is completely invalid
        for page in self.loader.load():
            total_pages += 1
            if total_pages == 1:
                source = page.source

            # The chunker might raise an exception if it fails catastrophically
            chunks = self.chunker.chunk_page(page)
            if not chunks:
                continue

            for chunk in chunks:
                chunk.document_id = document_id
                current_batch.append(chunk)
                total_chunks += 1

                if len(current_batch) >= self.batch_size:
                    self._process_batch(current_batch)
                    current_batch = []

        # Process any remaining chunks in the final partial batch
        if current_batch:
            self._process_batch(current_batch)

        return IngestionResult(
            document_id=document_id,
            source=source,
            total_pages=total_pages,
            total_chunks=total_chunks,
        )

    def _process_batch(self, batch: List[DocumentChunk]) -> None:
        """
        Embeds and stores a batch of DocumentChunks.
        """
        if not batch:
            return

        texts = []
        metadatas = []
        ids = []

        for chunk in batch:
            texts.append(chunk.text)
            metadatas.append(
                {
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "document_id": chunk.document_id,
                }
            )
            # Use the established deterministic ID generation from the vector store layer
            chunk_id = generate_chunk_id(
                document_id=chunk.document_id,
                source=chunk.source,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
            )
            ids.append(chunk_id)

        # The embedder or vector_store might raise exceptions; we let them bubble up
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
