import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from copilot import config


def load_file(file_path: str) -> list[Document]:
    """Load a PDF or text/code file into LangChain documents."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_bytes(name: str, data: bytes) -> list[Document]:
    """Load an uploaded file's bytes via a temp file, tagging chunks with the original filename."""
    suffix = os.path.splitext(name)[1]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "upload" + suffix)
        with open(tmp_path, "wb") as f:
            f.write(data)
        docs = load_file(tmp_path)

    for doc in docs:
        doc.metadata["source"] = os.path.basename(name)
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)
