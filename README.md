# DocuQuery-AI

DocuQuery-AI is a local Streamlit application for asking questions about
uploaded PDF documents. It retrieves relevant document chunks, answers with a
local Qwen GGUF model, and shows the source evidence used for the response.

## Key features

- PDF text extraction and recursive chunking
- Local sentence-transformer embeddings
- Persistent Chroma vector search
- Document-scoped retrieval across one or more selected PDFs
- Local Qwen answer generation
- Page-aware evidence display

## Architecture

```text
PDF upload -> loader -> chunker -> embeddings -> Chroma
											  |
Question -> embeddings -> scoped retrieval -> context -> Qwen answer
```

Document IDs are derived from uploaded PDF content and stored with each
indexed chunk. Streamlit selections restrict retrieval to the chosen documents.

## Tech stack

- Python
- Streamlit
- pypdf
- LangChain text splitters
- SentenceTransformers
- ChromaDB
- llama-cpp-python

## Local setup

Install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure the local Qwen GGUF model for the current shell:

```bash
export DOCUQUERY_QWEN_MODEL_PATH=/path/to/qwen-model.gguf
```

Start the application:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Limitations

- The current application is designed for local or trusted environments.
- Authentication and multi-user document ownership are not implemented.
- Documents are stored in a persistent local Chroma collection.
- PDF extraction quality depends on the source document and its text layer.
- The local Qwen model must be provided separately.

## Deployment note

The project is prepared for a deployment or portfolio demonstration, but it is
not publicly deployed by this repository. A deployment should provide the
runtime dependencies, a readable local model path, persistent storage, and an
appropriate access-control strategy.
