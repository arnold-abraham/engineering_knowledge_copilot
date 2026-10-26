from langchain_core.documents import Document

from copilot import config
from copilot.rag import answer_question, format_context, source_label, unique_sources


def test_source_label_pdf_page_is_one_indexed():
    doc = Document(page_content="", metadata={"source": "paper.pdf", "page": 0})

    assert source_label(doc) == "paper.pdf, p. 1"


def test_source_label_without_page():
    assert source_label(Document(page_content="", metadata={"source": "main.cpp"})) == "main.cpp"
    assert source_label(Document(page_content="")) == "Unknown Source"


def test_format_context_prefixes_each_chunk_with_label():
    docs = [
        Document(page_content="first", metadata={"source": "a.pdf", "page": 2}),
        Document(page_content="second", metadata={"source": "b.py"}),
    ]

    assert format_context(docs) == "[a.pdf, p. 3]\nfirst\n\n[b.py]\nsecond"


def test_unique_sources_dedupes_per_page_and_keeps_order():
    docs = [
        Document(page_content="1", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="2", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="3", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="4", metadata={"source": "b.py"}),
    ]

    assert [doc.page_content for doc in unique_sources(docs)] == ["1", "3", "4"]


def test_answer_question_returns_answer_and_retrieved_docs(vectorstore, llm):
    answer, docs = answer_question("What is the covariance update?", vectorstore, llm=llm)

    assert answer == "fake answer"
    assert 0 < len(docs) <= config.RETRIEVER_K

    prompt_messages = llm.calls[0]
    assert prompt_messages[0].type == "system"
    assert prompt_messages[-1].content.endswith("Question: What is the covariance update?")
    assert "[kf.pdf, p. 1]" in prompt_messages[-1].content


def test_answer_question_sends_only_recent_history(vectorstore, llm, monkeypatch):
    monkeypatch.setattr(config, "HISTORY_TURNS", 1)
    history = [
        ("user", "old question"),
        ("assistant", "old answer"),
        ("user", "recent question"),
        ("assistant", "recent answer"),
    ]

    answer_question("follow-up", vectorstore, history=history, llm=llm)

    messages = llm.calls[0]
    assert [(m.type, m.content) for m in messages[1:-1]] == [
        ("human", "recent question"),
        ("ai", "recent answer"),
    ]
