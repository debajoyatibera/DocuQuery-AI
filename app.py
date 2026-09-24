import hashlib
import os
from typing import Any, Dict

import streamlit as st

from core.answer_generation import (
    GenerationError,
    QwenLocalAnswerGenerator,
)
from core.chunking import RecursiveTextChunker
from core.context import SimpleContextBuilder
from core.embeddings import SentenceTransformerEmbeddingProvider
from core.ingestion import DocumentIngestionService
from core.loaders import DocumentLoadError, PdfDocumentLoader
from core.orchestration import RAGOrchestrator
from core.retrieval import VectorStoreRetriever
from core.vectorstore import ChromaVectorStore


@st.cache_resource
def get_embedding_provider() -> SentenceTransformerEmbeddingProvider:
    return SentenceTransformerEmbeddingProvider(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
    )


@st.cache_resource
def get_chunker() -> RecursiveTextChunker:
    return RecursiveTextChunker()


@st.cache_resource
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_directory="data/chroma_db",
        collection_name="docuquery",
    )


@st.cache_resource
def get_answer_generator(model_path: str) -> QwenLocalAnswerGenerator:
    return QwenLocalAnswerGenerator(model_path=model_path)


@st.cache_resource
def get_orchestrator(model_path: str) -> RAGOrchestrator:
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    retriever = VectorStoreRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    context_builder = SimpleContextBuilder()
    answer_generator = get_answer_generator(model_path)
    return RAGOrchestrator(
        retriever=retriever,
        context_builder=context_builder,
        answer_generator=answer_generator,
    )


def document_id_for(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def show_processed_documents(processed_documents: Dict[str, Dict[str, Any]]) -> None:
    if not processed_documents:
        return

    st.sidebar.subheader("Indexed documents")
    for document in processed_documents.values():
        st.sidebar.write(
            f"{document['source']} ({document['total_chunks']} chunks)"
        )


def document_display_labels(
    processed_documents: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    source_counts: Dict[str, int] = {}
    for document in processed_documents.values():
        source = document["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    labels: Dict[str, str] = {}
    for document_id, document in processed_documents.items():
        source = document["source"]
        label = source
        if source_counts[source] > 1:
            prefix_length = 8
            label = f"{source} ({document_id[:prefix_length]})"
            while label in labels.values() and prefix_length < len(document_id):
                prefix_length += 4
                label = f"{source} ({document_id[:prefix_length]})"
        labels[document_id] = label

    return labels


def ingest_uploaded_files(uploaded_files: list[Any]) -> None:
    processed_documents = st.session_state.processed_documents

    try:
        embedding_provider = get_embedding_provider()
        chunker = get_chunker()
        vector_store = get_vector_store()
    except Exception:
        st.error(
            "The document indexing resources could not be initialized. "
            "Check the runtime dependencies and try again."
        )
        return

    for uploaded_file in uploaded_files:
        file_name = "uploaded file"
        try:
            file_name = uploaded_file.name
            file_bytes = uploaded_file.getvalue()
            document_id = document_id_for(file_bytes)

            if document_id in processed_documents:
                st.info(f"Already indexed: {file_name}")
                continue

            # Cross-session duplicate checks are unavailable without bypassing
            # the current VectorStore abstraction, so failed duplicate adds are
            # reported without manipulating Chroma directly.
            uploaded_file.seek(0)
            loader = PdfDocumentLoader(
                file=uploaded_file,
                source=file_name,
            )
            ingestion_service = DocumentIngestionService(
                loader=loader,
                chunker=chunker,
                embedder=embedding_provider,
                vector_store=vector_store,
            )
            result = ingestion_service.ingest(document_id)

            if result.total_chunks == 0:
                st.warning(
                    f"No text could be indexed from {file_name}. "
                    "The document was not marked as processed."
                )
                continue

            processed_documents[document_id] = {
                "source": file_name,
                "total_pages": result.total_pages,
                "total_chunks": result.total_chunks,
            }
            if document_id not in st.session_state.selected_document_ids:
                st.session_state.selected_document_ids.append(document_id)
            st.success(
                f"Indexed {file_name}: "
                f"{result.total_chunks} chunks from {result.total_pages} pages."
            )
        except DocumentLoadError:
            st.error(f"Could not read {file_name} as a PDF.")
        except Exception:
            st.error(
                f"Could not index {file_name}. The document may already be "
                "present in persistent storage; cross-session duplicate checks "
                "are not exposed by the current vector-store abstraction."
            )


def main() -> None:
    st.set_page_config(
        page_title="DocuQuery AI",
        layout="centered",
    )

    st.title("DocuQuery AI")
    st.caption(
        "A local document question-answering workspace with scoped retrieval "
        "and source evidence."
    )

    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = {}
    if "selected_document_ids" not in st.session_state:
        st.session_state.selected_document_ids = []

    st.session_state.selected_document_ids = [
        document_id
        for document_id in st.session_state.selected_document_ids
        if document_id in st.session_state.processed_documents
    ]

    model_path = os.environ.get("DOCUQUERY_QWEN_MODEL_PATH", "").strip()
    model_available = bool(model_path) and os.path.isfile(model_path) and os.access(
        model_path, os.R_OK
    )
    if not model_path:
        st.error(
            "Local Qwen inference is not configured. Set "
            "DOCUQUERY_QWEN_MODEL_PATH to the path of a Qwen GGUF model "
            "before asking questions."
        )
    elif not model_available:
        st.error(
            "The configured Qwen model path is not a readable file. "
            "Update DOCUQUERY_QWEN_MODEL_PATH before asking questions."
        )

    show_processed_documents(st.session_state.processed_documents)

    st.subheader("Upload PDF documents")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Index documents", type="primary"):
        ingest_uploaded_files(uploaded_files)

    if st.session_state.processed_documents:
        st.subheader("Select documents")
        labels = document_display_labels(st.session_state.processed_documents)
        selected_document_ids = st.multiselect(
            "Select documents for questions",
            options=list(st.session_state.processed_documents),
            default=st.session_state.selected_document_ids,
            format_func=lambda document_id: labels[document_id],
        )
        st.session_state.selected_document_ids = selected_document_ids

    st.subheader("Ask a question")
    question = st.chat_input(
        "Ask a question about your indexed documents",
        disabled=not model_available,
    )
    if question is None:
        return

    if not question.strip():
        st.warning("Enter a question before submitting.")
        return

    if not st.session_state.processed_documents:
        st.info("Upload and index at least one PDF before asking a question.")
        return

    selected_document_ids = [
        document_id
        for document_id in st.session_state.selected_document_ids
        if document_id in st.session_state.processed_documents
    ]
    st.session_state.selected_document_ids = selected_document_ids
    if not selected_document_ids:
        st.info("Select at least one document before asking a question.")
        return

    try:
        orchestrator = get_orchestrator(model_path)
        result = orchestrator.answer(
            question,
            document_ids=selected_document_ids,
        )
    except GenerationError:
        st.error("The local Qwen model could not generate an answer.")
        return
    except Exception:
        st.error("The question could not be answered.")
        return

    st.subheader("Answer")
    st.write(result.answer_text)

    if not result.evidence:
        return

    st.subheader("Evidence")
    for evidence in result.evidence:
        page_label = (
            f"Page {evidence.page_number}"
            if evidence.page_number is not None
            else "Page unavailable"
        )
        with st.expander(f"{evidence.source} - {page_label}"):
            st.write(evidence.text)


if __name__ == "__main__":
    main()
