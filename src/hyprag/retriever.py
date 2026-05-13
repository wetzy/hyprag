"""
hyprag.retriever
~~~~~~~~~~~~~~~~
High-level API: chunker + encoder + hyperbolic index + subtree expansion.

    retriever = HypragRetriever()
    retriever.index_path("./myproject")

    results = retriever.query("how does the parser handle edge cases?", k=5)
    for chunk in results:
        print(chunk.node_path, chunk.start_line)

Core product insight
--------------------
After retrieving k chunks by geodesic distance, ``subtree_expand`` pulls every
*sibling* and *parent* of each hit.  Because the Poincaré ball organises
nodes radially by depth, hits tend to cluster by subtree — so expansion is
cheap (few extra nodes) but high-recall (you rarely miss a relevant method).

This "pull the whole subtree" behaviour is impossible for flat L2 retrieval:
there is no geometry that simultaneously captures semantic similarity *and*
the parent/child relationship.  The Poincaré ball encodes both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from hyprag.chunker import Chunk, HierarchicalChunker
from hyprag.index import PoincareBallIndex

__all__ = ["HypragRetriever", "subtree_expand"]


# ---------------------------------------------------------------------------
# Subtree expansion
# ---------------------------------------------------------------------------

def subtree_expand(
    results: list[Chunk],
    corpus: list[Chunk],
    *,
    include_parents: bool = True,
    include_children: bool = True,
    include_siblings: bool = True,
    max_expand: int = 50,
) -> list[Chunk]:
    """
    Expand a set of retrieved chunks with structurally related nodes.

    For each chunk in *results*, this function inspects the full *corpus* and
    adds chunks that are:

    - **Children** — chunks whose ``parent_path`` matches a retrieved
      ``node_path``.  This "pulls the subtree down".
    - **Parents** — the chunk whose ``node_path`` matches a retrieved
      ``parent_path``.  Provides context for the hit.
    - **Siblings** — chunks sharing the same ``parent_path`` as a retrieved
      chunk.  Useful when one method of a class is relevant and you want the
      whole class API surface.

    All three behaviours are enabled by default.  Each can be disabled
    independently.

    Parameters
    ----------
    results : list[Chunk]
        Initial retrieval results (k-nearest by geodesic distance).
    corpus : list[Chunk]
        The full indexed corpus to expand from.
    include_parents : bool
        Pull the direct parent of each retrieved chunk.
    include_children : bool
        Pull all direct children of each retrieved chunk.
    include_siblings : bool
        Pull all nodes sharing the same parent as each retrieved chunk.
    max_expand : int
        Hard cap on the total number of chunks returned.

    Returns
    -------
    list[Chunk]
        Deduplicated, expanded list.  Order: original results first, then
        parents, then children/siblings (corpus-order within each group).
    """
    if not results:
        return []

    retrieved_paths: set[str] = {c.node_path for c in results}
    retrieved_parents: set[str] = {c.parent_path for c in results if c.parent_path}

    seen_ids: set[int] = {c.id for c in results}
    expanded: list[Chunk] = list(results)

    # Single pass over corpus to classify every candidate
    parents_buf: list[Chunk] = []
    children_buf: list[Chunk] = []
    siblings_buf: list[Chunk] = []

    for chunk in corpus:
        if chunk.id in seen_ids:
            continue

        is_parent = include_parents and chunk.node_path in retrieved_parents
        is_child = include_children and chunk.parent_path in retrieved_paths
        is_sibling = (
            include_siblings
            and chunk.parent_path in retrieved_parents
            and not is_parent   # a parent of retrieved is not its own sibling
        )

        if is_parent:
            parents_buf.append(chunk)
        elif is_child:
            children_buf.append(chunk)
        elif is_sibling:
            siblings_buf.append(chunk)

    # Merge in priority order: parents first, then children, then siblings
    for buf in (parents_buf, children_buf, siblings_buf):
        for chunk in buf:
            if len(expanded) >= max_expand:
                return expanded
            if chunk.id not in seen_ids:
                expanded.append(chunk)
                seen_ids.add(chunk.id)

    return expanded


# ---------------------------------------------------------------------------
# High-level retriever
# ---------------------------------------------------------------------------

class HypragRetriever:
    """
    End-to-end hyperbolic code/document retriever.

    Combines:
    - ``HierarchicalChunker`` — AST-based chunking with depth + path metadata.
    - ``SentenceTransformer`` encoder — flat embeddings from any HF model.
    - ``PoincareBallIndex`` — depth-weighted projection onto the Poincaré ball.
    - ``subtree_expand`` — structural expansion after geodesic retrieval.

    Parameters
    ----------
    encoder_model : str
        Any ``sentence-transformers`` model name or local path.
        Default ``"all-MiniLM-L6-v2"`` (384-d, fast, solid quality).
    curvature : float
        Curvature of the Poincaré ball.  Default 1.0.
    max_depth : int
        Maximum hierarchy depth passed to both the chunker and the index.
        Default 2 (module → class → method).
    ball_scale : float
        Maximum radial norm for leaf nodes.  Default 0.9.
    min_norm : float
        Radial norm for root (depth-0) nodes.  Default 0.05.
    chunker_kwargs : dict, optional
        Extra keyword arguments forwarded to ``HierarchicalChunker``.
    device : str, optional
        ``"cpu"`` or ``"cuda"``.  Auto-detected when omitted.
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
    ) -> None:
        # Lazy import so that sentence-transformers is optional for users who
        # bring their own embeddings and use PoincareBallIndex directly.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HypragRetriever requires sentence-transformers: "
                "pip install sentence-transformers"
            ) from exc

        self._encoder = SentenceTransformer(encoder_model)
        dim: int = self._encoder.get_sentence_embedding_dimension()  # type: ignore[assignment]

        self._index = PoincareBallIndex(
            dim,
            curvature=curvature,
            ball_scale=ball_scale,
            max_depth=max_depth,
            min_norm=min_norm,
            device=device,
        )
        self._chunker = HierarchicalChunker(
            max_depth=max_depth, **(chunker_kwargs or {})
        )
        self._max_depth = max_depth
        self._chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_path(self, path: str | Path) -> int:
        """
        Chunk and index a Python file or directory tree.

        Can be called multiple times to incrementally grow the index.

        Parameters
        ----------
        path : str or Path
            A ``.py`` file or directory root.

        Returns
        -------
        int
            Number of chunks added in this call.
        """
        path = Path(path)
        chunks = (
            self._chunker.chunk_directory(path)
            if path.is_dir()
            else self._chunker.chunk_file(path)
        )
        if not chunks:
            return 0

        # Re-number IDs globally
        id_offset = len(self._chunks)
        for c in chunks:
            c.id += id_offset

        texts = [c.text for c in chunks]
        depths = [c.depth for c in chunks]

        vecs: np.ndarray = self._encoder.encode(  # type: ignore[assignment]
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        self._index.add(vecs, depths=depths)
        self._chunks.extend(chunks)
        return len(chunks)

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
    ) -> list[Chunk]:
        """
        Retrieve the most relevant chunks for *text*.

        Parameters
        ----------
        text : str
            Natural-language query.
        k : int
            Number of geodesic nearest neighbours to fetch before expansion.
        expand_subtree : bool
            When *True* (default), apply ``subtree_expand`` to the k results.
        include_parents / include_children / include_siblings : bool
            Fine-grained control over which relations are expanded.
            Only used when ``expand_subtree=True``.
        max_expand : int
            Hard cap on the total chunks returned after expansion.

        Returns
        -------
        list[Chunk]
            Retrieved (and optionally expanded) chunks, deduplicated.
        """
        if not self._chunks:
            raise RuntimeError("Index is empty — call .index_path() first.")

        q_vec: np.ndarray = self._encoder.encode(  # type: ignore[assignment]
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        dists, ids = self._index.search(q_vec, k)

        results: list[Chunk] = [
            self._chunks[idx]
            for idx in ids[0]
            if idx != -1
        ]

        if expand_subtree:
            results = subtree_expand(
                results,
                self._chunks,
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
        """Total number of indexed chunks."""
        return self._index.ntotal

    @property
    def chunks(self) -> list[Chunk]:
        """Read-only view of the full indexed corpus."""
        return self._chunks

    def __repr__(self) -> str:
        return (
            f"HypragRetriever("
            f"ntotal={self.ntotal}, "
            f"max_depth={self._max_depth}, "
            f"index={self._index!r})"
        )
