from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    This abstraction ensures the core RAG pipeline does not directly depend on
    a specific model implementation (e.g., SentenceTransformers, OpenAI).
    This allows easy swapping of models or switching between local and API-based
    inference without changing the rest of the application.
    """

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of strings into a list of embedding vectors.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimension of the embedding vectors produced by this provider.
        """
        pass


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Implementation of EmbeddingProvider using local SentenceTransformers.

    Loads the model lazily (upon initialization, not import) and runs
    entirely locally without API keys. Defaults to CPU execution.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        # Import inside the class to avoid slowing down module import
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._device = device
        self._model = SentenceTransformer(model_name, device=device)

        # Handle API change in newer sentence-transformers versions
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings.")

        if not texts:
            raise ValueError("Input list cannot be empty.")

        for text in texts:
            if not isinstance(text, str):
                raise ValueError("All elements in the input list must be strings.")
            if not text.strip():
                raise ValueError("Empty strings are not allowed.")

        # Encode and convert numpy array to list of floats (standard for ChromaDB)
        embeddings = self._model.encode(texts)
        return embeddings.tolist()
