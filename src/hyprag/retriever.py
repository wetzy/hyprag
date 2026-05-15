"""
hyprag.retriever
~~~~~~~~~~~~~~~~
High-level API: chunker + encoder + FAISS index + subtree expansion.

    retriever = HypragRetriever()
    retriever.index_path("./myproject")

    results = retriever.query("how does the parser handle edge cases?", k=5)
    for chunk in results:
        print(chunk.node_path, chunk.start_line)

Core product insight
--------------------
After retrieving *k* nearest neighbours by cosine similarity,
``subtree_expand`` walks the chunk hierarchy to pull every *parent*,
*sibling*, and *child* of each hit. The flat encoder finds the right region
of the document; the hierarchy walker fills in the surrounding context.

On the GDPR corpus (672 chunks, 20 hand-labeled queries, K=5, BGE-base),
this lifts Recall@5 from 0.530 (FAISS alone) to **0.866** (FAISS + subtree
expansion). On the CPython stdlib corpus (16k chunks) the same expansion
lifts Recall@5 from 0.092 to 0.203. Earlier experiments using a
Poincaré-ball backend produced numerically identical results at ~13× the
latency; the geometry has been retired.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hyprag.chunker import Chunk, HierarchicalChunker
from hyprag.faiss_index import FaissIndex

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
        Initial retrieval results (k-nearest by cosine similarity).
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
            and not is_parent
        )

        if is_parent:
            parents_buf.append(chunk)
        elif is_child:
            children_buf.append(chunk)
        elif is_sibling:
            siblings_buf.append(chunk)

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
    End-to-end hierarchical retriever.

    Combines:
    - ``HierarchicalChunker`` — AST-based chunking with depth + path metadata.
    - ``SentenceTransformer`` encoder — flat embeddings from any HF model.
    - ``FaissIndex`` — cosine-similarity nearest-neighbour search.
    - ``subtree_expand`` — structural expansion after initial retrieval.

    Parameters
    ----------
    encoder_model : str
        Any ``sentence-transformers`` model name or local path.
        Default ``"BAAI/bge-base-en-v1.5"`` (768-d, the encoder used in the
        published benchmarks).
    max_depth : int
        Maximum hierarchy depth passed to the AST chunker. Default 2
        (module → class → method). Pre-built chunks (e.g. from
        ``GDPRChunker``) are not affected.
    chunker_kwargs : dict, optional
        Extra keyword arguments forwarded to ``HierarchicalChunker``.
    """

    def __init__(
        self,
        encoder_model: str = "BAAI/bge-base-en-v1.5",
        *,
        max_depth: int = 2,
        chunker_kwargs: dict | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HypragRetriever requires sentence-transformers: "
                "pip install sentence-transformers"
            ) from exc

        self._encoder = SentenceTransformer(encoder_model)
        dim: int = self._encoder.get_sentence_embedding_dimension()  # type: ignore[assignment]

        self._index = FaissIndex(dim)
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

        id_offset = len(self._chunks)
        for c in chunks:
            c.id += id_offset

        return self._add_chunks(chunks)

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """
        Index a list of pre-built ``Chunk`` objects (e.g. from ``GDPRChunker``).

        Returns
        -------
        int
            Number of chunks added in this call.
        """
        if not chunks:
            return 0
        id_offset = len(self._chunks)
        for c in chunks:
            c.id = id_offset + c.id if c.id is not None else id_offset
            id_offset += 0  # ids should already be unique within the list
        return self._add_chunks(chunks)

    def _add_chunks(self, chunks: list[Chunk]) -> int:
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
            Number of nearest neighbours to fetch before expansion.
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

        _, ids = self._index.search(q_vec, k)

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
        return self._index.ntotal

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def __repr__(self) -> str:
        return (
            f"HypragRetriever("
            f"ntotal={self.ntotal}, "
            f"max_depth={self._max_depth}, "
            f"index={self._index!r})"
        )
