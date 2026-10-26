from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from copilot import config

SYSTEM_PROMPT = (
    "You are an expert engineering assistant. Use the provided context to answer technical questions. "
    "If you cannot answer based on the context, say so. "
    "Each context chunk starts with a bracketed source label, e.g. [paper.pdf, p. 3]. "
    "Cite the labels you used inline, exactly as written."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {input}"),
])


def build_vectorstore(chunks: list[Document], embeddings: Optional[Embeddings] = None) -> FAISS:
    """Embed chunks into a FAISS index."""
    return FAISS.from_documents(chunks, embeddings or OpenAIEmbeddings())


def source_label(doc: Document) -> str:
    """Human-readable citation, e.g. "paper.pdf, p. 3" (PDF pages are 0-indexed in metadata)."""
    source = doc.metadata.get("source", "Unknown Source")
    page = doc.metadata.get("page")
    return f"{source}, p. {page + 1}" if isinstance(page, int) else source


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{source_label(doc)}]\n{doc.page_content}" for doc in docs)


def unique_sources(docs: list[Document]) -> list[Document]:
    """Keep the first retrieved chunk per citation label (source + page)."""
    seen = set()
    unique = []
    for doc in docs:
        label = source_label(doc)
        if label not in seen:
            seen.add(label)
            unique.append(doc)
    return unique


def answer_question(
    question: str,
    vectorstore: FAISS,
    llm: Optional[BaseChatModel] = None,
) -> tuple[str, list[Document]]:
    """Retrieve relevant chunks and ask the LLM. Returns (answer, retrieved_docs)."""
    llm = llm or ChatOpenAI(model=config.CHAT_MODEL, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    retrieved_docs = retriever.invoke(question)
    result = (PROMPT | llm).invoke({"context": format_context(retrieved_docs), "input": question})

    answer = result.content if hasattr(result, "content") else str(result)
    return answer, retrieved_docs
