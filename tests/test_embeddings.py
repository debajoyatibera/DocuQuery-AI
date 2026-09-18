import math

import pytest

from core.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider


@pytest.fixture(scope="module")
def provider():
    """Fixture to load the model once for all tests in this module."""
    return SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")


def test_provider_initialization(provider):
    """Test that the provider is correctly instantiated and holds the model name."""
    assert isinstance(provider, EmbeddingProvider)
    assert provider.model_name == "all-MiniLM-L6-v2"


def test_embedding_dimension(provider):
    """Test that the embedding dimension is explicitly available and is exactly 384."""
    assert provider.dimension == 384


def test_embed_valid_texts(provider):
    """Test that embedding a valid list of texts returns lists of finite floats."""
    texts = ["This is a test document.", "Another document for testing."]
    embeddings = provider.embed_texts(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384

    # Check if values are finite floats
    for emb in embeddings:
        for val in emb:
            assert isinstance(val, float)
            assert math.isfinite(val)


def test_empty_input_list(provider):
    """Test that passing an empty list raises a ValueError."""
    with pytest.raises(ValueError, match="Input list cannot be empty."):
        provider.embed_texts([])


def test_invalid_input_type(provider):
    """Test that passing a non-list input raises a ValueError."""
    with pytest.raises(ValueError, match="Input must be a list of strings."):
        provider.embed_texts("Not a list")  # type: ignore


def test_invalid_elements_in_list(provider):
    """Test that passing a list with non-string elements raises a ValueError."""
    with pytest.raises(
        ValueError, match="All elements in the input list must be strings."
    ):
        provider.embed_texts(["Valid", 123, "Also valid"])  # type: ignore


def test_empty_string_in_list(provider):
    """Test that passing a list containing empty or whitespace-only strings raises a ValueError."""
    with pytest.raises(ValueError, match="Empty strings are not allowed."):
        provider.embed_texts(["Valid string", "   ", "Another string"])
