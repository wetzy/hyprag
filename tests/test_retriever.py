"""
tests.test_retriever
~~~~~~~~~~~~~~~~~~~~
Regression tests for ``HypragRetriever``.

These do NOT load a real encoder — they exercise the chunk-construction and
expansion logic by patching the encoder with a deterministic stub.
"""

from __future__ import annotations

import numpy as np
import pytest

from hyprag.retriever import HypragRetriever


class _StubEncoder:
    """Deterministic encoder: returns identity-like vectors from text length."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def encode(self, texts, **_kwargs):  # noqa: ANN001 - matches ST signature
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, i % self._dim] = float(len(t))
        return out

    def get_sentence_embedding_dimension(self):
        return self._dim


@pytest.fixture
def stub_retriever() -> HypragRetriever:
    r = HypragRetriever.__new__(HypragRetriever)
    r._encoder = _StubEncoder()
    from hyprag.faiss_index import FaissIndex

    r._index = FaissIndex(dim=8)
    r._chunks = []
    r._chunker = None
    return r


# ---------------------------------------------------------------------------
# Regression: index_texts must not turn all chunks into siblings
# ---------------------------------------------------------------------------

def test_index_texts_chunks_have_no_parent(stub_retriever):
    """
    Earlier ``index_texts`` used node_path='text.0', 'text.1', ... which
    gave every chunk parent_path='text'. With ``include_siblings=True``
    (default), querying with k=3 returned ALL chunks via the sibling link.
    Fix: chunks must be root-level (parent_path='').
    """
    stub_retriever.index_texts(["a" * 20, "b" * 21, "c" * 22, "d" * 23, "e" * 24])
    for c in stub_retriever._chunks:
        assert c.parent_path == "", (
            f"chunk {c.node_path!r} has parent_path={c.parent_path!r} — "
            f"flat index_texts chunks must have no parent"
        )


def test_query_k_respected_for_flat_texts(stub_retriever):
    """
    With flat (no-hierarchy) chunks, query(k=N) must return at most N
    chunks even with default expand_subtree=True. Regression for the
    'k is being ignored' bug in v0.5.2.
    """
    stub_retriever.index_texts(["x" * (10 + i) for i in range(10)])
    results = stub_retriever.query("query", k=3)
    assert len(results) <= 3, (
        f"expected <=3 chunks for k=3 on flat texts, got {len(results)}"
    )


def test_index_texts_node_paths_unique(stub_retriever):
    stub_retriever.index_texts([f"item {i} text padding" for i in range(5)])
    paths = [c.node_path for c in stub_retriever._chunks]
    assert len(set(paths)) == len(paths)
    assert all("." not in p for p in paths), (
        "index_texts node_paths must not contain dots — that creates a "
        "fake parent and turns every chunk into a sibling"
    )


def test_index_texts_custom_root_slug(stub_retriever):
    stub_retriever.index_texts(["short text padding 1", "short text padding 2"], root_slug="doc")
    paths = [c.node_path for c in stub_retriever._chunks]
    assert paths == ["doc0", "doc1"]


def test_index_texts_returns_count(stub_retriever):
    n = stub_retriever.index_texts(["one text long enough", "two text long enough"])
    assert n == 2


def test_index_texts_empty_is_noop(stub_retriever):
    assert stub_retriever.index_texts([]) == 0
    assert stub_retriever._chunks == []
