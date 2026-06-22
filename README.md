# Company Policy Assistant

Traditional RAG-based Q&A chatbot that answers questions about company policies using ChromaDB for vector search and GitHub Models for LLM responses.

## Setup

```bash
uv sync
```

Create a `.env` file with your API key:

```
GITHUB_TOKEN=your_token_here
```

## Usage

**1. Ingest documents**

Place policy documents (`.txt` or `.pdf`) in the `documents/` directory, then run:

```bash
uv run python ingest.py
```

**2. Start the app**

```bash
uv run streamlit run app.py
```

## Stack

- **Streamlit** — chat UI
- **ChromaDB** — vector store
- **Sentence Transformers** — embeddings
- **OpenAI SDK (GitHub Models)** — LLM responses
