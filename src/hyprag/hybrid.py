"""
hyprag.hybrid
~~~~~~~~~~~~~
HybridRetriever: BM25 lexical + HypRAG hyperbolic semantic, merged via
Reciprocal Rank Fusion (RRF), then subtree-expanded on the Poincaré ball.

Why this works
--------------
HypRAG's geometric expansion multiplies recall 2-13× *when the encoder lands
near the right cluster*.  The failure mode is zero-recall queries where the
query tokens have no lexical overlap with the code (e.g. "schedule callbacks"
vs. ``call_soon``).  BM25 finds those exact matches instantly.  RRF merges
both ranked lists without score normalisation, so neither retriever dominates.

Architecture
------------
    BM25Index (lexical) ──┐
                          ├── RRF merge ──► top-k ──► subtree_expand ──► results
    PoincareBallIndex ────┘
    (semantic, via HypragRetriever internals)

Drop-in API
-----------
    HybridRetriever has the same .index_path() / .query() surface as
    HypragRetriever.  The API server can swap them without touching call sites.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hyprag.bm25 import BM25Index
from hyprag.chunker import Chunk
from hyprag.retriever import HypragRetriever, subtree_expand

__all__ = ["HybridRetriever", "reciprocal_rank_fusion"]


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    *,
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Merge ranked lists of corpus indices via Reciprocal Rank Fusion.

    Each list contributes 1/(k + rank + 1) to each document's score.
    Documents not present in a list contribute 0 from that list.

    Parameters
    ----------
    ranked_lists : list of lists
        Each inner list is a ranked sequence of corpus indices (best first).
    k : int
        RRF constant. Default 60 per Cormack, Clarke & Buettcher (2009).

    Returns
    -------
    list of (corpus_index, rrf_score) sorted descending by score.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    BM25 + hyperbolic semantic retrieval with RRF fusion and subtree expansion.

    Parameters
    ----------
    encoder_model : str
        sentence-transformers model name.  Default ``"all-MiniLM-L6-v2"``.
    rrf_k : int
        RRF constant.  Default 60.
    bm25_candidates : int
        Number of BM25 results to fetch per query before fusion.
        Default ``max(k * 4, 20)``.
    semantic_candidates : int
        Number of semantic results to fetch per query before fusion.
        Default ``max(k * 4, 20)``.

    All remaining kwargs are forwarded to HypragRetriever (curvature,
    max_depth, ball_scale, min_norm, chunker_kwargs, device).
    """

    def __init__(
        self,
        encoder_model: str = "all-MiniLM-L6-v2",
        *,
        curvature: float = 1.0,
        max_depth: int = 2,
        ball_scale: float = 0.9,
        min_norm: float = 0.05,
        chunker_kwargs: dict | None = None,
        device: str | None = None,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        bm25_candidates: int | None = None,
        semantic_candidates: int | None = None,
    ) -> None:
        self._hyprag = HypragRetriever(
            encoder_model,
            curvature=curvature,
            max_depth=max_depth,
            ball_scale=ball_scale,
            min_norm=min_norm,
            chunker_kwargs=chunker_kwargs,
            device=device,
        )
        self._bm25 = BM25Index(k1=bm25_k1, b=bm25_b)
        self._rrf_k = rrf_k
        self._bm25_candidates = bm25_candidates
        self._semantic_candidates = semantic_candidates

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_path(self, path: str | Path) -> int:
        """
        Chunk and index a Python file or directory.

        Builds the hyperbolic index via HypragRetriever, then rebuilds the
        BM25 index over all chunk texts.  Incremental calls append to both.
        """
        n = self._hyprag.index_path(path)
        # BM25 rebuild is cheap (pure Python, ~0.5s for 16k chunks)
        self._bm25.build([c.text for c in self._hyprag.chunks])
        return n

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        k: int = 10,
        *,
        expand_subtree: bool = True,
        include_parents: bool = True,
        include_children: bool = True,
        include_siblings: bool = True,
        max_expand: int = 50,
        use_hybrid: bool = True,
    ) -> list[Chunk]:
        """
        Retrieve the most relevant chunks for *text*.

        Parameters
        ----------
        text : str
            Natural-language or code query.
        k : int
            Number of chunks to return after RRF (before expansion).
        expand_subtree : bool
            Apply subtree expansion after fusion.
        use_hybrid : bool
            When *True* (default), run BM25 + semantic RRF fusion.
            When *False*, fall back to pure HypRAG semantic retrieval
            (useful for A/B comparison without re-indexing).
        include_parents / include_children / include_siblings : bool
            Expansion controls, forwarded to subtree_expand.
        max_expand : int
            Hard cap on total chunks returned after expansion.

        Returns
        -------
        list[Chunk]
            Retrieved (and optionally expanded) chunks, deduplicated.
        """
        chunks = self._hyprag.chunks
        if not chunks:
            raise RuntimeError("Index is empty — call .index_path() first.")

        if not use_hybrid:
            return self._hyprag.query(
                text, k,
                expand_subtree=expand_subtree,
                include_parents=include_parents,
                include_children=include_children,
                include_siblings=include_siblings,
                max_expand=max_expand,
            )

        n_candidates = max(k * 4, 20)
        bm25_n = self._bm25_candidates or n_candidates
        semantic_n = self._semantic_candidates or n_candidates

        # --- Lexical branch ---
        _, bm25_ids = self._bm25.search(text, bm25_n)
        bm25_ranked: list[int] = list(bm25_ids)

        # --- Semantic branch (raw k-NN, no expansion yet) ---
        q_vec: np.ndarray = self._hyprag._encoder.encode(  # type: ignore[attr-defined]
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        _, sem_ids = self._hyprag._index.search(q_vec, semantic_n)  # type: ignore[attr-defined]
        semantic_ranked: list[int] = [idx for idx in sem_ids[0] if idx != -1]

        # --- RRF merge ---
        fused = reciprocal_rank_fusion(
            [semantic_ranked, bm25_ranked], k=self._rrf_k
        )

        # Top-k fused corpus indices → initial result set
        top_ids = [doc_id for doc_id, _ in fused[:k]]
        results = [chunks[i] for i in top_ids]

        # --- Subtree expansion on the fused set ---
        if expand_subtree:
            results = subtree_expand(
                results,
                chunks,
                include_parents=include_parents,
                include_children=include_children,
                include_siblings=include_siblings,
                max_expand=max_expand,
            )

        return results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        """Total indexed chunks."""
        return self._hyprag.ntotal

    @property
    def chunks(self) -> list[Chunk]:
        """Read-only view of the full corpus."""
        return self._hyprag.chunks

    def __repr__(self) -> str:
        return (
            f"HybridRetriever("
            f"ntotal={self.ntotal}, "
            f"rrf_k={self._rrf_k})"
        )
