import streamlit as st

from copilot import config
from copilot.ingest import load_bytes, split_documents
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
    st.session_state.index_key = None
    st.session_state.chunk_count = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit reruns this script on every interaction, so only re-embed when the uploads change
index_key = tuple(f.file_id for f in uploaded_files)

if index_key != st.session_state.index_key:
    st.session_state.vectorstore = None
    st.session_state.index_key = index_key

    if uploaded_files:
        docs = []
        for uploaded_file in uploaded_files:
            docs.extend(load_bytes(uploaded_file.name, uploaded_file.getvalue()))

        split_docs = split_documents(docs)

        if not split_docs:
            st.session_state.index_key = None
            st.error("No document chunks were created. Check uploaded files and loaders.")
            st.stop()

        try:
            with st.spinner("Indexing documents..."):
                st.session_state.vectorstore = build_vectorstore(split_docs)
        except Exception as e:
            st.session_state.index_key = None
            st.error("Failed to build the vector index. Check your OPENAI_API_KEY and network access.")
            st.exception(e)
            st.stop()

        st.session_state.chunk_count = len(split_docs)

if st.session_state.vectorstore:

    st.success(f"Processed {st.session_state.chunk_count} document chunks.")

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
