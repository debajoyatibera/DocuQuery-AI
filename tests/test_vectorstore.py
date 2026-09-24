import pytest

from core.vectorstore import ChromaVectorStore, VectorStore, generate_chunk_id


@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory for ChromaDB tests to avoid polluting real data/.
    Using pytest's tmp_path allows the directory to be cleaned up naturally by pytest
    after the process terminates, avoiding strict Windows file lock PermissionErrors.
    """
    return str(tmp_path / "chroma_data")


@pytest.fixture
def vector_store(temp_dir):
    """Provides a fresh ChromaVectorStore instance for testing."""
    return ChromaVectorStore(persist_directory=temp_dir)


def test_vectorstore_initialization(vector_store):
    """Test that the vector store initializes and inherits from the ABC."""
    assert isinstance(vector_store, VectorStore)
    assert vector_store.collection.name == "docuquery"
    assert vector_store.collection.metadata["hnsw:space"] == "cosine"


def test_generate_chunk_id():
    """Test deterministic SHA-256 ID generation."""
    id1 = generate_chunk_id("test-doc-id", "doc.pdf", 1, 0, "Hello")
    id2 = generate_chunk_id("test-doc-id", "doc.pdf", 1, 0, "Hello")
    id3 = generate_chunk_id("test-doc-id", "doc.pdf", 1, 1, "Hello")
    id4 = generate_chunk_id("other-doc-id", "doc.pdf", 1, 0, "Hello")

    assert isinstance(id1, str)
    assert len(id1) == 64  # SHA-256 hex digest length
    assert id1 == id2  # Identical inputs produce identical IDs
    assert id1 != id3  # Different chunk index produces different ID
    assert id1 != id4  # Different source produces different ID


def test_add_and_search_one_document(vector_store):
    """Test adding a single document and searching for it."""
    texts = ["Test document"]
    embeddings = [[0.1, 0.2, 0.3]]
    metadatas = [{"source": "test.pdf", "page_number": 1, "chunk_index": 0}]
    ids = [generate_chunk_id("test-doc-id", "test.pdf", 1, 0, "Test document")]

    vector_store.add(texts, embeddings, metadatas, ids)

    results = vector_store.search(query_embedding=[0.1, 0.2, 0.3], n_results=1)

    assert len(results) == 1
    assert results[0]["id"] == ids[0]
    assert results[0]["text"] == texts[0]
    assert results[0]["metadata"] == metadatas[0]
    assert isinstance(results[0]["distance"], float)


def test_add_multiple_documents(vector_store):
    """Test adding multiple documents and verifying search order by distance."""
    texts = ["Doc A", "Doc B"]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    metadatas = [{"source": "A"}, {"source": "B"}]
    ids = ["id_A", "id_B"]

    vector_store.add(texts, embeddings, metadatas, ids)

    results = vector_store.search(query_embedding=[1.0, 0.0], n_results=2)
    assert len(results) == 2
    # The closest one should be Doc A (cosine distance 0.0)
    assert results[0]["id"] == "id_A"
    assert results[0]["distance"] < results[1]["distance"]


def test_search_with_one_document_id_returns_only_that_document(vector_store):
    vector_store.add(
        texts=["Doc A", "Doc B"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {"source": "A", "document_id": "doc-a"},
            {"source": "B", "document_id": "doc-b"},
        ],
        ids=["id_A", "id_B"],
    )

    results = vector_store.search(
        query_embedding=[1.0, 0.0],
        n_results=2,
        document_ids=["doc-b"],
    )

    assert [result["id"] for result in results] == ["id_B"]


def test_search_with_multiple_document_ids_returns_only_selected_documents(
    vector_store,
):
    vector_store.add(
        texts=["Doc A", "Doc B", "Doc C"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        metadatas=[
            {"source": "A", "document_id": "doc-a"},
            {"source": "B", "document_id": "doc-b"},
            {"source": "C", "document_id": "doc-c"},
        ],
        ids=["id_A", "id_B", "id_C"],
    )

    results = vector_store.search(
        query_embedding=[1.0, 0.0],
        n_results=3,
        document_ids=["doc-a", "doc-c"],
    )

    assert {result["id"] for result in results} == {"id_A", "id_C"}


def test_search_with_empty_document_ids_returns_no_results(vector_store):
    vector_store.add(
        texts=["Doc A"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"source": "A", "document_id": "doc-a"}],
        ids=["id_A"],
    )

    results = vector_store.search(
        query_embedding=[1.0, 0.0],
        n_results=1,
        document_ids=[],
    )

    assert results == []


def test_search_with_unknown_document_id_returns_no_results(vector_store):
    vector_store.add(
        texts=["Doc A"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"source": "A", "document_id": "doc-a"}],
        ids=["id_A"],
    )

    results = vector_store.search(
        query_embedding=[1.0, 0.0],
        n_results=1,
        document_ids=["unknown-doc"],
    )

    assert results == []


def test_search_passes_expected_document_filter_to_chroma():
    class FakeCollection:
        def __init__(self):
            self.received_kwargs = None

        def query(self, **kwargs):
            self.received_kwargs = kwargs
            return {"ids": [[]]}

    collection = FakeCollection()
    vector_store = object.__new__(ChromaVectorStore)
    vector_store.collection = collection

    vector_store.search(
        query_embedding=[1.0, 0.0],
        n_results=2,
        document_ids=["doc-a", "doc-b"],
    )

    assert collection.received_kwargs["where"] == {
        "document_id": {"$in": ["doc-a", "doc-b"]}
    }


def test_search_without_document_ids_preserves_unrestricted_query_shape():
    class FakeCollection:
        def __init__(self):
            self.received_kwargs = None

        def query(self, **kwargs):
            self.received_kwargs = kwargs
            return {"ids": [[]]}

    collection = FakeCollection()
    vector_store = object.__new__(ChromaVectorStore)
    vector_store.collection = collection

    vector_store.search(query_embedding=[1.0, 0.0], n_results=2)

    assert "where" not in collection.received_kwargs


def test_empty_inputs(vector_store):
    """Test that empty inputs raise a sensible ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        vector_store.add([], [], [], [])

    with pytest.raises(ValueError, match="query_embedding cannot be empty"):
        vector_store.search([])


def test_mismatched_input_lengths(vector_store):
    """Test that mismatched input lengths raise a ValueError."""
    with pytest.raises(ValueError, match="must have the same length"):
        vector_store.add(["A"], [[1.0]], [{"source": "A"}], ["id1", "id2"])


def test_invalid_n_results(vector_store):
    """Test that an invalid n_results value raises a ValueError."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        vector_store.search([1.0], n_results=0)


def test_persistence(temp_dir):
    """Test that ChromaDB properly persists data across different instances."""
    store1 = ChromaVectorStore(persist_directory=temp_dir)
    store1.add(
        texts=["Persistent text"],
        embeddings=[[0.5, 0.5]],
        metadatas=[{"test": True}],
        ids=["p_id1"],
    )

    # Create a new instance pointing to the same temp directory
    store2 = ChromaVectorStore(persist_directory=temp_dir)
    results = store2.search(query_embedding=[0.5, 0.5], n_results=1)

    assert len(results) == 1
    assert results[0]["id"] == "p_id1"
    assert results[0]["text"] == "Persistent text"
