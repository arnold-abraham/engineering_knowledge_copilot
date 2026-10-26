from copilot import config
from copilot.ingest import load_bytes, split_documents


def test_load_bytes_text_uses_original_filename_as_source():
    docs = load_bytes("notes.md", b"# Notes\nKalman filters.")

    assert len(docs) == 1
    assert docs[0].page_content == "# Notes\nKalman filters."
    assert docs[0].metadata["source"] == "notes.md"


def test_load_bytes_strips_directories_from_filename():
    docs = load_bytes("../../etc/notes.txt", b"hello")

    assert docs[0].metadata["source"] == "notes.txt"


def test_load_bytes_pdf_returns_one_document_per_page(sample_pdf_bytes):
    docs = load_bytes("sample.pdf", sample_pdf_bytes)

    assert [doc.metadata["page"] for doc in docs] == [0, 1]
    assert "covariance update" in docs[0].page_content
    assert all(doc.metadata["source"] == "sample.pdf" for doc in docs)


def test_split_documents_respects_chunk_size():
    docs = load_bytes("long.txt", ("word " * 1000).encode())

    chunks = split_documents(docs)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= config.CHUNK_SIZE for chunk in chunks)
    assert all(chunk.metadata["source"] == "long.txt" for chunk in chunks)


def test_split_documents_empty_input():
    assert split_documents([]) == []
