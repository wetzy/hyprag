"""
hyprag.hybrid
~~~~~~~~~~~~~
HybridRetriever: BM25 lexical + dense semantic retrieval, merged via
Reciprocal Rank Fusion (RRF), then subtree-expanded over the chunk hierarchy.

When to use
-----------
BM25 catches exact-token matches that a dense encoder may miss — common in
code corpora where a query like "schedule callbacks" must find a method
named ``call_soon``. On natural-language corpora with uniform vocabulary
(e.g. GDPR's pervasive "data", "processing", "controller"), BM25 tends to
hurt rather than help, because exact-token signals are too noisy to break
ties between semantically similar articles.

Empirically:
- CPython stdlib (K=5, BGE): hybrid +5% Recall vs FAISS+expand alone.
- GDPR (K=5, BGE-base): hybrid −7% Recall vs FAISS+expand alone.

Default to ``use_hybrid=False`` and turn it on only when the corpus has
informative lexical signal.

Architecture
------------
    BM25Index (lexical)  ──┐
                           ├── RRF merge ──► top-k ──► subtree_expand ──► results
    FaissIndex (semantic) ─┘
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

    Each list contributes ``1 / (k + rank + 1)`` to each document's score.
    Documents not present in a list contribute 0 from that list.

    Returns ``[(corpus_index, rrf_score)]`` sorted descending by score.
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
    BM25 + dense semantic retrieval with RRF fusion and subtree expansion.

    Parameters
    ----------
    encoder_model : str
        sentence-transformers model name. Default ``"BAAI/bge-base-en-v1.5"``.
    rrf_k : int
        RRF constant. Default 60.
    bm25_candidates : int
        Number of BM25 results to fetch per query before fusion.
        Default ``max(k * 4, 20)``.
    semantic_candidates : int
        Number of semantic results to fetch per query before fusion.
        Default ``max(k * 4, 20)``.

    Remaining kwargs (max_depth, chunker_kwargs) are forwarded to
    HypragRetriever.
    """

    def __init__(
        self,
        encoder_model: str = "BAAI/bge-base-en-v1.5",
        *,
        max_depth: int = 2,
        chunker_kwargs: dict | None = None,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        bm25_candidates: int | None = None,
        semantic_candidates: int | None = None,
    ) -> None:
        self._hyprag = HypragRetriever(
            encoder_model,
            max_depth=max_depth,
            chunker_kwargs=chunker_kwargs,
        )
        self._bm25 = BM25Index(k1=bm25_k1, b=bm25_b)
        self._rrf_k = rrf_k
        self._bm25_candidates = bm25_candidates
        self._semantic_candidates = semantic_candidates

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_path(self, path: str | Path) -> int:
        n = self._hyprag.index_path(path)
        self._bm25.build([c.text for c in self._hyprag.chunks])
        return n

    def index_chunks(self, chunks: list[Chunk]) -> int:
        n = self._hyprag.index_chunks(chunks)
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

        Set ``use_hybrid=False`` to short-circuit to pure semantic retrieval
        (useful for A/B comparison without re-indexing).
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

        top_ids = [doc_id for doc_id, _ in fused[:k]]
        results = [chunks[i] for i in top_ids]

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
        return self._hyprag.ntotal

    @property
    def chunks(self) -> list[Chunk]:
        return self._hyprag.chunks

    def __repr__(self) -> str:
        return (
            f"HybridRetriever("
            f"ntotal={self.ntotal}, "
            f"rrf_k={self._rrf_k})"
        )
