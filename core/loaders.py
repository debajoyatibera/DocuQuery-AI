import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Union

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


@dataclass
class DocumentPage:
    """
    Represents a single physical page extracted from a document.
    Ensures text is bundled with its precise source and page provenance.
    """

    text: str
    source: str
    page_number: int  # 1-based indexing


class DocumentLoadError(Exception):
    """
    Application-level exception raised for document loading failures.
    Encapsulates underlying parsing/IO exceptions to decouple from specific libraries.
    """

    pass


class DocumentLoader(ABC):
    """
    Abstract interface for loading documents into pages incrementally.
    """

    @abstractmethod
    def load(self) -> Iterator[DocumentPage]:
        """
        Incrementally yields DocumentPage objects to avoid loading
        entire massive documents into memory simultaneously.
        """
        pass


class PdfDocumentLoader(DocumentLoader):
    """
    Extracts text from a PDF file incrementally.
    Accepts file-like objects (e.g., Streamlit UploadedFile) or local Paths.
    """

    def __init__(self, file: Union[str, Path, BinaryIO], source: str):
        self.file = file
        self.source = source

    def load(self) -> Iterator[DocumentPage]:
        """
        Iterates over the PDF pages and yields non-empty DocumentPage objects.
        """
        if isinstance(self.file, (str, Path)):
            try:
                # Use a context manager to ensure Windows file handles are cleanly released
                with open(self.file, "rb") as f:
                    yield from self._extract_pages(f)
            except (OSError, IOError) as e:
                raise DocumentLoadError(f"Could not open file {self.file}: {e}") from e
        else:
            # Assume file-like object (e.g., BytesIO from Streamlit)
            yield from self._extract_pages(self.file)

    def _extract_pages(self, file_obj: BinaryIO) -> Iterator[DocumentPage]:
        try:
            reader = PdfReader(file_obj)
        except PdfReadError as e:
            raise DocumentLoadError(f"Failed to read PDF '{self.source}': {e}") from e
        except Exception as e:
            raise DocumentLoadError(
                f"Unexpected error loading PDF '{self.source}': {e}"
            ) from e

        if reader.is_encrypted:
            raise DocumentLoadError(f"Cannot read encrypted PDF: {self.source}")

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
            except Exception as e:
                # Log the error but attempt to salvage the rest of the document
                logger.warning(
                    "Failed to extract text from %s, page %d: %s",
                    self.source,
                    page_idx,
                    e,
                )
                continue

            if text is None:
                continue

            clean_text = text.strip()
            if not clean_text:
                continue

            yield DocumentPage(
                text=clean_text,
                source=self.source,
                page_number=page_idx,
            )
