"""
tests.test_retriever
~~~~~~~~~~~~~~~~~~~~
Regression + feature tests for ``HypragRetriever``.

These do NOT load a real encoder. The ``_StubEncoder`` maps text to a
deterministic vector by counting characters in a small alphabet, so
``encode("aaa") @ encode("aaab")`` is high and ``encode("aaa") @ encode("xyz")``
is zero. This is enough to exercise the FAISS path, subtree expansion,
score capture, and rescoring without downloading any model weights.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hyprag.chunker import Chunk
from hyprag.retriever import (
    HypragRetriever,
    RetrievalResult,
    _expand_with_metadata,
    subtree_expand,
)


# ---------------------------------------------------------------------------
# Encoder + retriever fixtures
# ---------------------------------------------------------------------------

class _StubEncoder:
    """
    Character-count encoder.

    Each text becomes an 8-d vector where ``v[i] = count of letter i in text``,
    using a small lowercase alphabet. Two strings with overlapping letter
    profiles produce vectors with high cosine similarity; disjoint strings
    produce orthogonal vectors. Sufficient for unit tests without weights.
    """

    _ALPHABET = "abcdefgh"

    def __init__(self, dim: int = 8) -> None:
        assert dim == len(self._ALPHABET), "stub uses an 8-letter alphabet"
        self._dim = dim

    def encode(self, texts, **_kwargs):  # noqa: ANN001 - matches ST signature
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            lower = t.lower()
            for j, ch in enumerate(self._ALPHABET):
                out[i, j] = float(lower.count(ch))
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


def _make_chunk(id_: int, node_path: str, text: str) -> Chunk:
    return Chunk(
        id=id_,
        text=text,
        depth=node_path.count("."),
        node_path=node_path,
        source_file="",
        start_line=0,
        end_line=0,
    )


# ---------------------------------------------------------------------------
# index_texts regression (kept from 0.5.3)
# ---------------------------------------------------------------------------

def test_index_texts_chunks_have_no_parent(stub_retriever):
    stub_retriever.index_texts(["a" * 20, "b" * 21, "c" * 22, "d" * 23, "e" * 24])
    for c in stub_retriever._chunks:
        assert c.parent_path == ""


def test_query_k_respected_for_flat_texts(stub_retriever):
    stub_retriever.index_texts(["x" * (10 + i) for i in range(10)])
    results = stub_retriever.query("query", k=3)
    assert len(results) <= 3


def test_index_texts_node_paths_unique(stub_retriever):
    stub_retriever.index_texts([f"item {i} text padding" for i in range(5)])
    paths = [c.node_path for c in stub_retriever._chunks]
    assert len(set(paths)) == len(paths)
    assert all("." not in p for p in paths)


def test_index_texts_custom_root_slug(stub_retriever):
    stub_retriever.index_texts(["short padding text 1", "short padding text 2"], root_slug="doc")
    paths = [c.node_path for c in stub_retriever._chunks]
    assert paths == ["doc0", "doc1"]


def test_index_texts_returns_count(stub_retriever):
    n = stub_retriever.index_texts(["one text long enough", "two text long enough"])
    assert n == 2


def test_index_texts_empty_is_noop(stub_retriever):
    assert stub_retriever.index_texts([]) == 0
    assert stub_retriever._chunks == []


# ---------------------------------------------------------------------------
# Backward compatibility: default query still returns list[Chunk]
# ---------------------------------------------------------------------------

def test_query_default_returns_list_of_chunks(stub_retriever):
    stub_retriever.index_texts(["aaa bbb", "ccc ddd", "eee fff"])
    results = stub_retriever.query("aaa", k=2)
    assert all(isinstance(r, Chunk) for r in results)


def test_query_empty_index_raises(stub_retriever):
    with pytest.raises(RuntimeError, match="Index is empty"):
        stub_retriever.query("anything")


# ---------------------------------------------------------------------------
# return_metadata=True → RetrievalResult with score/relation/seed_path
# ---------------------------------------------------------------------------

def test_return_metadata_yields_retrieval_results(stub_retriever):
    stub_retriever.index_texts(["aaa bbb", "ccc ddd"])
    results = stub_retriever.query("aaa", k=2, return_metadata=True)
    assert len(results) >= 1
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(hasattr(r, "score") for r in results)
    assert all(r.relation in ("seed", "parent", "child", "sibling") for r in results)


def test_seed_has_real_score_and_empty_seed_path(stub_retriever):
    stub_retriever.index_texts(["aaaa", "bbbb", "cccc"])
    results = stub_retriever.query("aaaa", k=1, return_metadata=True)
    seeds = [r for r in results if r.relation == "seed"]
    assert len(seeds) == 1
    seed = seeds[0]
    assert 0.0 <= seed.score <= 1.0 + 1e-6, f"score {seed.score} out of [0, 1]"
    # Perfect match: "aaaa" query against "aaaa" chunk → cosine ~ 1.0
    assert seed.score > 0.99
    assert seed.seed_path == ""


def test_expanded_chunks_have_nan_score_without_rescore(stub_retriever):
    """
    Without rescore, expanded chunks have no meaningful similarity to the
    query (they were pulled in structurally). Their score is NaN to make
    that explicit, not 0.0 (which would imply zero similarity).
    """
    parent = _make_chunk(0, "doc", "header aaaaa")
    seed = _make_chunk(1, "doc.child1", "aaaa bbbb")
    sibling = _make_chunk(2, "doc.child2", "cccc dddd")
    stub_retriever.index_chunks([parent, seed, sibling])

    results = stub_retriever.query("aaaa", k=1, return_metadata=True)
    for r in results:
        if r.relation != "seed":
            assert math.isnan(r.score), (
                f"expected NaN score for relation={r.relation!r}, got {r.score}"
            )


# ---------------------------------------------------------------------------
# Subtree expansion + relation tagging
# ---------------------------------------------------------------------------

def test_parent_relation_tagged_correctly(stub_retriever):
    parent = _make_chunk(0, "root", "parent header text")
    seed = _make_chunk(1, "root.child", "aaaa target text")
    stub_retriever.index_chunks([parent, seed])

    results = stub_retriever.query("aaaa", k=1, return_metadata=True)
    by_rel = {r.relation: r for r in results}
    assert "seed" in by_rel
    assert "parent" in by_rel
    assert by_rel["parent"].chunk.node_path == "root"
    assert by_rel["parent"].seed_path == "root.child"


def test_sibling_relation_tagged_correctly(stub_retriever):
    parent = _make_chunk(0, "root", "parent header")
    seed = _make_chunk(1, "root.a", "aaaa target")
    sibling = _make_chunk(2, "root.b", "bbbb other")
    stub_retriever.index_chunks([parent, seed, sibling])

    results = stub_retriever.query("aaaa", k=1, return_metadata=True)
    by_path = {r.chunk.node_path: r for r in results}
    assert by_path["root.b"].relation == "sibling"
    assert by_path["root.b"].seed_path == "root.a"


def test_child_relation_tagged_correctly(stub_retriever):
    seed = _make_chunk(0, "root", "aaaa parent text")
    child = _make_chunk(1, "root.kid", "bbbb child text")
    stub_retriever.index_chunks([seed, child])

    results = stub_retriever.query("aaaa", k=1, return_metadata=True)
    by_path = {r.chunk.node_path: r for r in results}
    assert by_path["root.kid"].relation == "child"
    assert by_path["root.kid"].seed_path == "root"


def test_expand_subtree_false_disables_expansion(stub_retriever):
    parent = _make_chunk(0, "root", "parent header")
    seed = _make_chunk(1, "root.a", "aaaa target")
    sibling = _make_chunk(2, "root.b", "bbbb other")
    stub_retriever.index_chunks([parent, seed, sibling])

    results = stub_retriever.query("aaaa", k=1, expand_subtree=False, return_metadata=True)
    assert len(results) == 1
    assert results[0].relation == "seed"


# ---------------------------------------------------------------------------
# rescore_after_expand: reorders by semantic similarity
# ---------------------------------------------------------------------------

def test_rescore_promotes_relevant_sibling_above_irrelevant_parent(stub_retriever):
    """
    Wikipedia €20M scenario in miniature.

    The seed is the FAISS top-1 by definition — rescore can never demote it.
    What rescore DOES do is promote the second-most-relevant chunk (often
    a sibling with the actual answer) above structural-but-irrelevant
    chunks (like a parent heading with no query overlap).

    Without rescore the order is fixed: [seed, parent, children, siblings]
    — so a low-content parent ends up at position 1 even when a content-rich
    sibling that answers the query sits at position N.
    """
    seed = _make_chunk(0, "root.a", "aaaa bb")          # FAISS top-1
    parent = _make_chunk(1, "root", "ffff gg hh")       # irrelevant heading
    sibling = _make_chunk(2, "root.b", "aaa cc dd")     # contains answer
    stub_retriever.index_chunks([seed, parent, sibling])

    # Without rescore: structural order — parent before sibling
    plain = stub_retriever.query("aaaa", k=1, return_metadata=True)
    plain_paths = [r.chunk.node_path for r in plain]
    assert plain_paths.index("root") < plain_paths.index("root.b"), (
        f"structural order broken: {plain_paths}"
    )

    # With rescore: sibling (contains a's) outranks parent (no a's)
    rescored = stub_retriever.query(
        "aaaa", k=1, return_metadata=True, rescore_after_expand=True
    )
    rescored_paths = [r.chunk.node_path for r in rescored]
    assert rescored_paths.index("root.b") < rescored_paths.index("root"), (
        f"rescore failed to promote sibling above irrelevant parent: "
        f"{rescored_paths}"
    )
    # Seed stays at top — it's by definition the highest-cosine chunk
    assert rescored[0].chunk.node_path == "root.a"


def test_rescore_sorts_descending_by_score(stub_retriever):
    parent = _make_chunk(0, "root", "header aaa zzz")
    seed = _make_chunk(1, "root.a", "aaaa main")
    sibling1 = _make_chunk(2, "root.b", "aa other")
    sibling2 = _make_chunk(3, "root.c", "bb other")
    stub_retriever.index_chunks([parent, seed, sibling1, sibling2])

    results = stub_retriever.query(
        "aaaa", k=1, return_metadata=True, rescore_after_expand=True
    )
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"results not sorted by score descending: {scores}"
    )


def test_rescore_assigns_score_to_every_chunk(stub_retriever):
    parent = _make_chunk(0, "root", "header aaa")
    seed = _make_chunk(1, "root.a", "aaaa bb")
    sibling = _make_chunk(2, "root.b", "cccc dd")
    stub_retriever.index_chunks([parent, seed, sibling])

    results = stub_retriever.query(
        "aaaa", k=1, return_metadata=True, rescore_after_expand=True
    )
    for r in results:
        assert not math.isnan(r.score), (
            f"chunk {r.chunk.node_path!r} has NaN score after rescore"
        )
        assert 0.0 <= r.score <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# subtree_expand backward compatibility
# ---------------------------------------------------------------------------

def test_subtree_expand_returns_list_of_chunks():
    parent = _make_chunk(0, "root", "p")
    seed = _make_chunk(1, "root.a", "s")
    sibling = _make_chunk(2, "root.b", "x")
    out = subtree_expand([seed], [parent, seed, sibling])
    assert all(isinstance(c, Chunk) for c in out)
    paths = {c.node_path for c in out}
    assert paths == {"root", "root.a", "root.b"}


def test_expand_with_metadata_returns_tuples():
    parent = _make_chunk(0, "root", "p")
    seed = _make_chunk(1, "root.a", "s")
    sibling = _make_chunk(2, "root.b", "x")
    out = _expand_with_metadata(
        [seed],
        [parent, seed, sibling],
        include_parents=True,
        include_children=True,
        include_siblings=True,
        max_expand=50,
    )
    by_rel = {rel: (c, sp) for c, rel, sp in out}
    assert by_rel["seed"][0].node_path == "root.a"
    assert by_rel["parent"][0].node_path == "root"
    assert by_rel["parent"][1] == "root.a"
    assert by_rel["sibling"][0].node_path == "root.b"
    assert by_rel["sibling"][1] == "root.a"
