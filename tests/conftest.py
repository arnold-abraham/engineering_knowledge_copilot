from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from copilot.rag import build_vectorstore

DATA_DIR = Path(__file__).parent / "data"


class RecordingChatModel(FakeListChatModel):
    """Fake chat model that records the messages it was called with."""

    calls: list = []

    def _call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        self.calls.append(messages)
        return super()._call(messages, *args, **kwargs)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return (DATA_DIR / "sample.pdf").read_bytes()


@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def llm() -> RecordingChatModel:
    return RecordingChatModel(responses=["fake answer"])


@pytest.fixture
def vectorstore(embeddings):
    docs = [
        Document(page_content="Kalman filter covariance update.", metadata={"source": "kf.pdf", "page": 0}),
        Document(page_content="Extended Kalman filter linearisation.", metadata={"source": "kf.pdf", "page": 1}),
        Document(page_content="def update(state): return state", metadata={"source": "filter.py"}),
    ]
    return build_vectorstore(docs, embeddings)
