
# Engineering Knowledge Copilot

A simple RAG-based engineering assistant that allows users to:

- Upload PDFs, documentation, and code files
- Ask technical questions
- Retrieve grounded answers with citations
- Maintain conversational memory

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

---

# Run the application

```bash
streamlit run app.py
```

---

# Example Questions

- Explain how FMI state serialization works
- Compare UKF and EKF
- Summarize section 4 of this paper
- What does this C++ function do?
- Explain the covariance update step

---

# Future Improvements

- Multi-agent workflows
- OCR support
- Image embeddings
- Video understanding
- Hybrid retrieval
- Local LLM support
- LangGraph orchestration
- Persistent vector database
