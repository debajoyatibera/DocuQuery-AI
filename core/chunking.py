from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from core.loaders import DocumentPage


@dataclass
class DocumentChunk:
    """
    Represents a specific chunk of text from a source document.
    Maintains perfect provenance to its exact original page.
    """

    text: str
    source: str
    page_number: int
    chunk_index: int


class TextChunker(ABC):
    """
    Abstract interface for chunking a DocumentPage into DocumentChunks.
    Guarantees isolation of the chunking algorithm from application orchestration.
    """

    @abstractmethod
    def chunk_page(self, page: DocumentPage) -> List[DocumentChunk]:
        """
        Takes exactly one DocumentPage and returns a list of DocumentChunks.
        Must reset chunk indexing to 0 for every page.
        """
        pass


class RecursiveTextChunker(TextChunker):
    """
    Implementation of TextChunker using LangChain's RecursiveCharacterTextSplitter.
    Defaults to 700 character chunks with 100 character overlap.
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        # Lazy import to keep LangChain dependency strictly encapsulated
        # and prevent heavy module loading if an alternative chunker is used.
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk_page(self, page: DocumentPage) -> List[DocumentChunk]:
        """
        Splits the given DocumentPage into chunks.
        Index sequence starts at 0 for each individual page.
        """
        if not page or not page.text:
            return []

        clean_text = page.text.strip()
        if not clean_text:
            return []

        # Split text using LangChain
        text_chunks = self._splitter.split_text(clean_text)

        # Map raw text strings back into our domain dataclass
        document_chunks = []
        for index, chunk_text in enumerate(text_chunks):
            document_chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=index,
                )
            )

        return document_chunks
