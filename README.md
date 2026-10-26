
# Engineering Knowledge Copilot

A simple RAG-based engineering assistant that allows users to:

- Upload PDFs, documentation, and code files
- Ask technical questions
- Retrieve grounded answers with page-level citations
- Ask follow-up questions in a chat with recent-turn memory

---

# Features

- OpenAI API integration
- Embeddings + Vector Search
- FAISS vector database
- Conversational RAG pipeline
- Streamlit UI
- PDF + code ingestion

---

# Architecture

Documents → Chunking → Embeddings → FAISS → Retriever → LLM → Grounded Answer

```
app.py               # Streamlit UI
copilot/
  config.py          # settings (API key, model, chunk size, k, history length)
  ingest.py          # load uploaded files and split them into chunks
  rag.py             # build the FAISS index, format cited context, answer questions
tests/               # unit tests (run offline with fake embeddings and LLM)
```

---

# Installation

## 1. Clone or unzip the project

## 2. Create virtual environment

### Linux / Mac
```bash
python -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

---

# Install dependencies

```bash
pip install -r requirements.txt
```

---

# Configure Environment

1. Copy the template:
```bash
cp .env.example .env
```

2. Add your OpenAI API key to `.env`:
```bash
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

3. Optionally choose a different chat model (default `gpt-4o-mini`):
```bash
CHAT_MODEL=gpt-4o-mini
```

---

# Run the application

```bash
streamlit run app.py
```

---

# Development

```bash
pip install -r requirements-dev.txt
pytest          # run the tests (no API key needed)
ruff check .    # lint
```

CI runs both on every push and pull request to `main`.

---

# Limitations

- The vector index lives in memory and is rebuilt per browser session; nothing is persisted.
- Retrieval uses only the current question, so vague follow-ups ("explain more") may retrieve less relevant chunks.
- Text and code files must be UTF-8 encoded.
- Scanned PDFs without a text layer are not supported (no OCR).

---

# Example Questions

- Explain how FMI state serialization works
- Compare UKF and EKF
- Summarize section 4 of this paper
- What does this C++ function do?
- Explain the covariance update step
