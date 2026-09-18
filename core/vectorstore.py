import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List


def generate_chunk_id(
    source: str, page_number: int, chunk_index: int, text: str
) -> str:
    """
    Generate a deterministic SHA-256 ID for a document chunk.
    This ensures identical documents generate the same IDs,
    preventing duplicate chunk ingestion.
    """
    unique_string = f"{source}_{page_number}_{chunk_index}_{text}"
    return hashlib.sha256(unique_string.encode("utf-8")).hexdigest()


class VectorStore(ABC):
    """
    Abstract base class for vector database operations.

    This interface abstracts away the underlying database implementation (e.g. ChromaDB),
    ensuring the core RAG logic remains decoupled from specific storage vendors.
    """

    @abstractmethod
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """
        Add documents to the vector store.
        """
        pass

    @abstractmethod
    def search(
        self, query_embedding: List[float], n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store using the query embedding.
        Returns a list of dictionaries containing: id, text, metadata, distance
        """
        pass


class ChromaVectorStore(VectorStore):
    """
    Implementation of VectorStore using ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "docuquery",
    ):
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        if not texts or not embeddings or not metadatas or not ids:
            raise ValueError(
                "Inputs (texts, embeddings, metadatas, ids) cannot be empty."
            )

        if (
            len(texts) != len(embeddings)
            or len(texts) != len(metadatas)
            or len(texts) != len(ids)
        ):
            raise ValueError("All input lists must have the same length.")

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self, query_embedding: List[float], n_results: int = 5
    ) -> List[Dict[str, Any]]:
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")

        # query expects a list of query embeddings
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        formatted_results = []

        # Results from Chroma are nested lists because it supports multiple query embeddings
        # We only pass one query_embedding, so we index [0]
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(ids)):
                formatted_results.append(
                    {
                        "id": ids[i],
                        "text": documents[i] if documents else "",
                        "metadata": metadatas[i] if metadatas else {},
                        "distance": distances[i] if distances else 0.0,
                    }
                )

        return formatted_results
