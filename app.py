import os

import streamlit as st

from copilot import config
from copilot.ingest import load_file, split_documents
from copilot.rag import answer_question, build_vectorstore, unique_sources

st.set_page_config(page_title="Engineering Knowledge Copilot", layout="wide")

st.title("🧠 Engineering Knowledge Copilot")
st.markdown(
    '''
Upload engineering PDFs or code files and ask technical questions.
This project demonstrates:
- RAG (Retrieval-Augmented Generation)
- Embeddings + Vector Search
- OpenAI API Integration
- Stateful Chat Memory
- Document Grounding with Citations
'''
)

if not config.OPENAI_API_KEY:
    st.warning("Please set your OPENAI_API_KEY inside the .env file.")
    st.stop()

uploaded_files = st.file_uploader(
    "Upload PDFs or code/text files",
    type=config.SUPPORTED_EXTENSIONS,
    accept_multiple_files=True
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:

    docs = []
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        docs.extend(load_file(file_path))

    split_docs = split_documents(docs)

    if not split_docs:
        st.error("No document chunks were created. Check uploaded files and loaders.")
        st.stop()

    try:
        st.session_state.vectorstore = build_vectorstore(split_docs)
    except Exception as e:
        st.error("Failed to build the vector index. Check your OPENAI_API_KEY and network access.")
        st.exception(e)
        st.stop()

    st.success(f"Processed {len(split_docs)} document chunks.")

if st.session_state.vectorstore:

    user_question = st.text_input("Ask a technical question")

    if user_question:

        answer, retrieved_docs = answer_question(user_question, st.session_state.vectorstore)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")

        for doc in unique_sources(retrieved_docs):
            st.markdown(f"**{doc.metadata.get('source', 'Unknown Source')}**")
            st.code(doc.page_content[:700])

else:
    st.info("Upload files to begin.")
