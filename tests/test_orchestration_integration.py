import gc
import io
import logging
import os

import pytest

from core.answer_generation import QwenLocalAnswerGenerator
from core.chunking import RecursiveTextChunker
from core.context import SimpleContextBuilder
from core.embeddings import SentenceTransformerEmbeddingProvider
from core.ingestion import DocumentIngestionService
from core.loaders import PdfDocumentLoader
from core.orchestration import NO_EVIDENCE_RESPONSE, RAGOrchestrator
from core.retrieval import VectorStoreRetriever
from core.vectorstore import ChromaVectorStore


@pytest.fixture
def mock_pdf_file():
    # A tiny valid PDF containing a clear, answerable statement.
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 300 300] /Contents 5 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"5 0 obj\n<< /Length 85 >>\nstream\nBT /F1 12 Tf 10 10 Td (Employees must scan their RFID card once when entering the office.) Tj ET\nendstream\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000060 00000 n \n"
        b"0000000117 00000 n \n0000000249 00000 n \n0000000323 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n460\n%%EOF\n"
    )
    return io.BytesIO(pdf_bytes)


@pytest.fixture
def setup_pipeline(tmp_path, mock_pdf_file):
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    persist_dir = str(tmp_path / "chroma_integ")

    loader = PdfDocumentLoader(file=mock_pdf_file, source="policy.pdf")
    chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
    embedder = SentenceTransformerEmbeddingProvider()
    vector_store = ChromaVectorStore(
        persist_directory=persist_dir, collection_name="orchestration_test"
    )

    service = DocumentIngestionService(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )
    service.ingest()

    retriever = VectorStoreRetriever(
        vector_store=vector_store, embedding_provider=embedder
    )
    context_builder = SimpleContextBuilder()

    yield retriever, context_builder

    try:
        del vector_store
    except NameError:
        pass
    gc.collect()


@pytest.mark.integration
def test_end_to_end_orchestration_real_model(setup_pipeline):
    model_path = os.environ.get("DOCUQUERY_QWEN_MODEL_PATH")
    if not model_path:
        pytest.skip(
            "DOCUQUERY_QWEN_MODEL_PATH is not set. Skipping real model integration test."
        )

    retriever, context_builder = setup_pipeline

    generator = QwenLocalAnswerGenerator(model_path=model_path, temperature=0.1)

    orchestrator = RAGOrchestrator(
        retriever=retriever, context_builder=context_builder, answer_generator=generator
    )

    result = orchestrator.answer(
        "How many times must an employee scan their RFID card when entering the office?"
    )

    assert result.answer_text
    assert len(result.evidence) > 0

    # Check provenance
    assert result.evidence[0].source == "policy.pdf"
    assert result.evidence[0].page_number == 1

    # Semantic check
    answer_lower = result.answer_text.lower()
    assert "once" in answer_lower or "1" in answer_lower or "one" in answer_lower


@pytest.mark.integration
def test_end_to_end_orchestration_no_evidence(tmp_path):
    # This test verifies short-circuiting at integration level using an empty DB.
    # It does not require downloading/loading a model because the AnswerGenerator is short-circuited.
    persist_dir = str(tmp_path / "chroma_empty")

    embedder = SentenceTransformerEmbeddingProvider()
    vector_store = ChromaVectorStore(
        persist_directory=persist_dir, collection_name="empty_test"
    )

    # An empty vector store will return 0 chunks
    retriever = VectorStoreRetriever(
        vector_store=vector_store, embedding_provider=embedder
    )
    context_builder = SimpleContextBuilder()

    # Using a fake generator to absolutely prove it's NOT called, avoiding model load time
    from tests.test_orchestration import FakeAnswerGenerator

    generator = FakeAnswerGenerator()

    orchestrator = RAGOrchestrator(
        retriever=retriever, context_builder=context_builder, answer_generator=generator
    )

    result = orchestrator.answer("Is the office open on weekends?")

    # Generator should NOT be called
    assert generator.received_question is None

    # Verify controlled response and empty evidence
    assert result.answer_text == NO_EVIDENCE_RESPONSE
    assert result.evidence == []

    try:
        del vector_store
    except NameError:
        pass
    gc.collect()
