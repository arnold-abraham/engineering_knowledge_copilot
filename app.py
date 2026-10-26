import streamlit as st

from copilot import config
from copilot.ingest import load_bytes, split_documents
from copilot.rag import answer_question, build_vectorstore, source_label, unique_sources

st.set_page_config(page_title="Engineering Knowledge Copilot", layout="wide")


def render_sources(sources: list[tuple[str, str]]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for label, snippet in sources:
            st.markdown(f"**{label}**")
            st.code(snippet)


st.title("🧠 Engineering Knowledge Copilot")
st.markdown(
    '''
Upload engineering PDFs or code files and ask technical questions.
This project demonstrates:
- RAG (Retrieval-Augmented Generation)
- Embeddings + Vector Search
- OpenAI API Integration
- Chat Memory (recent turns)
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
    st.session_state.chat_history = []

# Streamlit reruns this script on every interaction, so only re-embed when the uploads change
index_key = tuple(f.file_id for f in uploaded_files)

if index_key != st.session_state.index_key:
    st.session_state.vectorstore = None
    st.session_state.index_key = index_key
    st.session_state.chat_history = []

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

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message["sources"])

    if question := st.chat_input("Ask a technical question"):

        with st.chat_message("user"):
            st.markdown(question)

        history = [(m["role"], m["content"]) for m in st.session_state.chat_history]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, retrieved_docs = answer_question(question, st.session_state.vectorstore, history)
            sources = [(source_label(doc), doc.page_content[:700]) for doc in unique_sources(retrieved_docs)]
            st.markdown(answer)
            render_sources(sources)

        st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})

else:
    st.info("Upload files to begin.")
