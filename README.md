# DocuQuery AI

> A local Retrieval-Augmented Generation (RAG) system for question answering over PDF documents.

DocuQuery AI lets a user upload one or more PDFs, select the documents to use,
and ask natural-language questions about their contents. The application
retrieves relevant document context for a local Qwen GGUF model and displays
the generated answer with source and page-aware evidence.

## Live Demo

[Open the temporary GitHub Codespaces demo](https://miniature-memory-5gjgj7rjqr65c5v-8501.app.github.dev)

The demo requires the Codespace to be running and is temporary; it is not
permanent production hosting.

The [GitHub repository](https://github.com/debajoyatibera/DocuQuery-AI) is the
source of truth for the project.

## Why this project?

Searching long PDFs manually is slow and difficult when the question is phrased
in natural language. This project demonstrates a local RAG workflow that turns
PDF text into searchable chunks, retrieves context relevant to a question, and
returns an answer alongside the retrieved source evidence.

## Key Features

- PDF ingestion and text extraction with page metadata
- Recursive text chunking
- Local SentenceTransformer embeddings
- Persistent ChromaDB vector search
- Document-scoped retrieval across one or more selected PDFs
- Local Qwen GGUF answer generation through `llama-cpp-python`
- Page-aware source evidence in the Streamlit interface
- API-key-independent local inference for the core runtime
- Missing-evidence handling when retrieval returns no usable context

## How It Works

### Ingestion

```text
PDF -> extraction -> page-aware chunks -> embeddings -> ChromaDB
```

Uploaded PDF content is assigned a deterministic document ID. Each indexed
chunk retains its source filename, page number, chunk index, and document ID.

### Question answering

```text
Question -> query embedding -> scoped retrieval -> context construction
	  -> local Qwen generation -> answer + evidence
```

The user selects one or more indexed PDFs before asking a question. Retrieval
is restricted to those selected document IDs rather than the entire collection.

## Architecture

```text
		    DOCUMENT INGESTION
			   |
PDF --> Loader --> Chunker --> Embeddings --> ChromaDB
					       |
					       |
Question --> Query Embedding --> Scoped Retrieval
				      |
				      v
			      Retrieved Context
				      |
				      v
			      Local Qwen GGUF
				      |
				      v
			       Answer + Evidence
```

## Why RAG?

RAG connects a language model to a specific document collection at question
time:

1. Documents are split into manageable chunks.
2. Chunks are represented with embedding vectors.
3. Relevant chunks are retrieved for the user question.
4. Retrieved context is provided to the local language model.
5. The application displays the answer and evidence associated with the retrieved context.

This approach makes the answer generation workflow inspectable without claiming
that retrieval completely eliminates model errors or hallucinations.

## Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Application and backend logic |
| Streamlit | User interface |
| pypdf | PDF text extraction |
| LangChain text splitters | Recursive text chunking |
| SentenceTransformers | Local embeddings |
| ChromaDB | Vector storage and retrieval |
| llama-cpp-python | Local GGUF inference |
| Qwen | Local answer generation |

## Project Structure

```text
app.py                         Streamlit application entry point
requirements.txt               Runtime dependencies
core/
  answer_generation.py         Answer-generator abstraction and local Qwen adapter
  chunking.py                  Page-aware recursive text chunking
  context.py                   Retrieved-context construction
  embeddings.py                Embedding-provider abstraction and implementation
  ingestion.py                 Document ingestion orchestration
  loaders.py                   PDF loading and page extraction
  orchestration.py             RAG workflow coordination
  retrieval.py                 Query embedding and vector retrieval
  vectorstore.py               Vector-store abstraction and Chroma implementation
data/                          Local persistent vector-store location
tests/                         Unit and integration tests
pytest.ini                     Pytest configuration
```

## Local Setup

The Qwen GGUF model is not included in this repository. Provide a readable
local model file separately.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Linux, macOS, or Codespaces

```bash
export DOCUQUERY_QWEN_MODEL_PATH=/path/to/qwen-model.gguf
```

### Windows PowerShell

```powershell
$env:DOCUQUERY_QWEN_MODEL_PATH = "C:\path\to\qwen-model.gguf"
```

Start Streamlit:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Usage

1. Start the Streamlit application.
2. Upload one or more PDF documents.
3. Index the uploaded documents.
4. Select one or more documents for retrieval.
5. Ask a question in natural language.
6. Review the generated answer.
7. Inspect the source filename, page number, and retrieved evidence text.

## Testing

The current verified test result is:

```text
87 passed, 1 skipped
```

Run the suite with:

```bash
pytest -q
```

## Limitations

- Authentication is not implemented.
- Multi-user document ownership is not implemented.
- Documents use persistent local ChromaDB storage.
- The local Qwen GGUF model must be provided separately.
- PDF extraction quality depends on the document's text layer.
- The GitHub Codespaces demo is temporary and depends on the running Codespace.
- Local language-model performance depends on available CPU and RAM.

## Future Improvements

The following are planned or possible improvements, not current features:

- Stronger provenance propagation through the evidence models
- Document management, deletion, and replacement workflows
- Retrieval evaluation and benchmark reporting
- Reranking for more precise retrieval
- Improved citation and evidence UX
- Authentication and multi-user isolation
- Persistent production storage
- Permanent production deployment
- Stronger local or hosted language models

## Learning / Engineering Highlights

- Modular RAG architecture with clear provider abstractions
- Local embedding and vector-retrieval pipeline
- Document-scoped retrieval across selected PDFs
- Deterministic document and chunk identity
- Page-aware provenance and evidence handling
- Dependency inversion between RAG components and infrastructure providers
- Automated unit and integration testing
- API-key-independent local inference

## Deployment Note

The current Codespaces link is a temporary portfolio demonstration, not
permanent hosting or a production deployment. A longer-lived deployment would
need managed runtime dependencies, model and storage provisioning, and an
appropriate access-control strategy.
