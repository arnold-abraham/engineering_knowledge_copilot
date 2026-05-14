
import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("Please set your OPENAI_API_KEY inside the .env file.")
    st.stop()

uploaded_files = st.file_uploader(
    "Upload PDFs or code/text files",
    type=["pdf", "py", "cpp", "c", "txt", "md"],
    accept_multiple_files=True
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:

    docs = []

    upload_dir = "uploaded_docs"
    os.makedirs(upload_dir, exist_ok=True)

    for uploaded_file in uploaded_files:

        file_path = os.path.join(upload_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

        else:
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    split_count = len(split_docs)

    if split_count == 0:
        st.error("No document chunks were created. Check uploaded files and loaders.")
        st.stop()

    embeddings = OpenAIEmbeddings()

    # Sanity-check embeddings creation to surface informative errors
    try:
        # test a single embedding to ensure the embedding service is working
        test_text = split_docs[0].page_content if split_docs else ""
        test_emb = embeddings.embed_documents([test_text])
    except Exception as e:
        st.error("Failed to create embeddings. Check your OPENAI_API_KEY and network access.")
        st.exception(e)
        st.stop()

    if not test_emb or len(test_emb) == 0 or len(test_emb[0]) == 0:
        st.error("Embeddings returned empty vectors. Verify the embedding backend and API key.")
        st.stop()

    try:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
    except IndexError as e:
        st.error("Internal error building FAISS index: received empty embeddings list.")
        st.exception(e)
        st.stop()
    except Exception as e:
        st.error("Unexpected error building vectorstore.")
        st.exception(e)
        st.stop()

    st.session_state.vectorstore = vectorstore

    st.success(f"Processed {split_count} document chunks.")

if st.session_state.vectorstore:

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    system_prompt = (
        "You are an expert engineering assistant. Use the provided context to answer technical questions. "
        "If you cannot answer based on the context, say so. Always cite your sources."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {input}")
    ])

    user_question = st.text_input("Ask a technical question")

    if user_question:

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(user_question)

        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Create the chain manually
        chain = prompt | llm
        result = chain.invoke({"context": context, "input": user_question})
        
        answer = result.content if hasattr(result, 'content') else str(result)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")

        seen = set()

        for doc in retrieved_docs:

            source_name = doc.metadata.get("source", "Unknown Source")

            if source_name in seen:
                continue

            seen.add(source_name)

            st.markdown(f"**{source_name}**")
            st.code(doc.page_content[:700])

else:
    st.info("Upload files to begin.")
