from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from core.retrieval import RetrievedChunk


@dataclass
class BuiltContext:
    formatted_text: str
    used_chunks: List[RetrievedChunk]


class ContextBuilder(ABC):
    @abstractmethod
    def build(
        self,
        chunks: List[RetrievedChunk],
        max_chars: int = 10000,
    ) -> BuiltContext:
        """Build the context payload from a list of retrieved chunks."""
        pass


class SimpleContextBuilder(ContextBuilder):
    def build(
        self,
        chunks: List[RetrievedChunk],
        max_chars: int = 10000,
    ) -> BuiltContext:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")

        if not chunks:
            return BuiltContext(formatted_text="", used_chunks=[])

        used_chunks = []
        formatted_blocks = []
        seen_keys = set()
        current_length = 0

        for chunk in chunks:
            # Deduplication key: source + page_number + chunk_index
            key = (chunk.source, chunk.page_number, chunk.chunk_index)
            if key in seen_keys:
                continue

            # Format the chunk
            page_info = (
                f" (Page {chunk.page_number})" if chunk.page_number is not None else ""
            )
            block = f"Document: {chunk.source}{page_info}\n{chunk.text}"

            # Calculate the added length including separators
            added_length = len(block)
            if formatted_blocks:
                added_length += 2  # account for '\n\n' separator

            # Ensure we do not exceed the token/character budget
            if current_length + added_length > max_chars:
                break

            # Add to built context
            seen_keys.add(key)
            formatted_blocks.append(block)
            used_chunks.append(chunk)
            current_length += added_length

        formatted_text = "\n\n".join(formatted_blocks)
        return BuiltContext(formatted_text=formatted_text, used_chunks=used_chunks)
