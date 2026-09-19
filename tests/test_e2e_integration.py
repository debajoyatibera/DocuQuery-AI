import gc
import io
import logging

import pytest

from core.chunking import RecursiveTextChunker
from core.embeddings import SentenceTransformerEmbeddingProvider
from core.ingestion import DocumentIngestionService
from core.loaders import PdfDocumentLoader
from core.vectorstore import ChromaVectorStore


@pytest.mark.integration
def test_end_to_end_ingestion_and_retrieval(tmp_path):
    """
    Genuine end-to-end integration test.
    Verifies that all components function together securely without mocks.
    """
    # Suppress pypdf startxref warning for the artificial PDF to keep test output clean
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 300 300] /Contents 6 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"5 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 300 300] /Contents 7 0 R >>\nendobj\n"
        b"6 0 obj\n<< /Length 61 >>\nstream\nBT /F1 12 Tf 10 10 Td (DocuQuery AI is a RAG system.) Tj ET\nendstream\nendobj\n"
        b"7 0 obj\n<< /Length 69 >>\nstream\nBT /F1 12 Tf 10 10 Td (It supports vector similarity search.) Tj ET\nendstream\nendobj\n"
        b"xref\n0 8\n0000000000 65535 f \n0000000009 00000 n \n0000000060 00000 n \n"
        b"0000000125 00000 n \n0000000257 00000 n \n0000000331 00000 n \n"
        b"0000000463 00000 n \n0000000574 00000 n \n"
        b"trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n694\n%%EOF\n"
    )

    file_obj = io.BytesIO(pdf_bytes)

    loader = PdfDocumentLoader(file=file_obj, source="e2e_integration.pdf")
    chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)

    # This invokes the real model download/load process
    embedder = SentenceTransformerEmbeddingProvider()

    persist_dir = str(tmp_path / "chroma_e2e")

    try:
        vector_store = ChromaVectorStore(
            persist_directory=persist_dir, collection_name="e2e_test"
        )

        service = DocumentIngestionService(
            loader=loader,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            batch_size=32,
        )

        result = service.ingest()

        # 1. Verify ingestion result counts
        assert result.total_pages == 2
        assert result.total_chunks == 2

        # 2. Verify ChromaDB contains exactly 2 records
        assert vector_store.collection.count() == 2

        # 3. Verify embedding dimension
        query = "What is DocuQuery?"
        q_emb = embedder.embed_texts([query])[0]
        assert len(q_emb) == 384

        # 4. Verify similarity retrieval
        search_results = vector_store.search(q_emb, n_results=1)
        assert len(search_results) == 1

        best_match = search_results[0]
        assert "DocuQuery AI is a RAG system." in best_match["text"]

        # 5. Verify retrieved metadata provenance
        metadata = best_match["metadata"]
        assert metadata["source"] == "e2e_integration.pdf"
        assert metadata["page_number"] == 1
        assert metadata["chunk_index"] == 0

        # Ensure complete release of the vector store to free Windows file locks
        del vector_store
        gc.collect()

        # 6. Verify persistence after reopening
        vector_store_reopened = ChromaVectorStore(
            persist_directory=persist_dir, collection_name="e2e_test"
        )
        assert vector_store_reopened.collection.count() == 2

    finally:
        # Guarantee we don't hold file locks if assertions fail
        try:
            del vector_store
        except NameError:
            pass
        try:
            del vector_store_reopened
        except NameError:
            pass
        gc.collect()
