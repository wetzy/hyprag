"""
hyprag.retriever
~~~~~~~~~~~~~~~~
High-level API: chunker + encoder + FAISS index + subtree expansion.

    retriever = HypragRetriever()
    retriever.index_path("./myproject")

    results = retriever.query("how does the parser handle edge cases?", k=5)
    for chunk in results:
        print(chunk.node_path, chunk.start_line)

For richer downstream control, request metadata:

    results = retriever.query(text, k=5, return_metadata=True)
    for r in results:
        print(r.score, r.relation, r.chunk.node_path)

Core product insight
--------------------
After retrieving *k* nearest neighbours by cosine similarity,
``subtree_expand`` walks the chunk hierarchy to pull every *parent*,
*sibling*, and *child* of each hit. The flat encoder finds the right region
of the document; the hierarchy walker fills in the surrounding context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hyprag.chunker import Chunk, HierarchicalChunker
from hyprag.faiss_index import FaissIndex

__all__ = [
    "HypragRetriever",
    "RetrievalResult",
    "subtree_expand",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    A single chunk returned from ``HypragRetriever.query(..., return_metadata=True)``.

    Attributes
    ----------
    chunk : Chunk
        The retrieved chunk.
    score : float
        Cosine similarity to the query in ``[0.0, 1.0]``. For seeds this is
        always the FAISS similarity. For expanded chunks (parents, siblings,
        children) it is the FAISS similarity if ``rescore_after_expand=True``
        was passed, else ``float('nan')``.
    relation : str
        One of ``"seed"``, ``"parent"``, ``"sibling"``, ``"child"``.
        ``"seed"`` means the chunk was returned by the initial FAISS search.
    seed_path : str
        ``node_path`` of the seed chunk that pulled this chunk in via
        ``subtree_expand``. Empty for seeds themselves.
    """

    chunk: Chunk
    score: float
    relation: str
    seed_path: str


# ---------------------------------------------------------------------------
# Subtree expansion
# ---------------------------------------------------------------------------

def _expand_with_metadata(
    seeds: list[Chunk],
    corpus: list[Chunk],
    *,
    include_parents: bool,
    include_children: bool,
    include_siblings: bool,
    max_expand: int,
) -> list[tuple[Chunk, str, str]]:
    """
    Internal: expand seeds with the full corpus and tag each expanded chunk
    with its relation (``"parent"``/``"child"``/``"sibling"``) and the
    ``node_path`` of the seed that pulled it in.

    Returns a list of ``(chunk, relation, seed_path)`` tuples. The seeds
    themselves come first, tagged ``"seed"`` with empty ``seed_path``.
    Order after seeds: parents, children, siblings (corpus order within
    each group).
    """
    if not seeds:
        return []

    # Maps from node_path/parent_path → seed.node_path so we can attribute
    # each expanded chunk back to a specific seed. If a chunk matches
    # multiple seeds, the first seed wins.
    parent_of_seed: dict[str, str] = {}   # seed.parent_path → seed.node_path
    seeds_by_path: dict[str, str] = {}    # seed.node_path → seed.node_path
    siblings_by_parent: dict[str, str] = {}  # seed.parent_path → seed.node_path

    for s in seeds:
        seeds_by_path.setdefault(s.node_path, s.node_path)
        if s.parent_path:
            parent_of_seed.setdefault(s.parent_path, s.node_path)
            siblings_by_parent.setdefault(s.parent_path, s.node_path)

    seen_ids: set[int] = {c.id for c in seeds}
    out: list[tuple[Chunk, str, str]] = [(c, "seed", "") for c in seeds]

    parents_buf: list[tuple[Chunk, str, str]] = []
    children_buf: list[tuple[Chunk, str, str]] = []
    siblings_buf: list[tuple[Chunk, str, str]] = []

    for chunk in corpus:
        if chunk.id in seen_ids:
            continue

        if include_parents and chunk.node_path in parent_of_seed:
            parents_buf.append((chunk, "parent", parent_of_seed[chunk.node_path]))
            continue

        if include_children and chunk.parent_path in seeds_by_path:
            children_buf.append(
                (chunk, "child", seeds_by_path[chunk.parent_path])
            )
            continue

        if (
            include_siblings
            and chunk.parent_path
            and chunk.parent_path in siblings_by_parent
        ):
            siblings_buf.append(
                (chunk, "sibling", siblings_by_parent[chunk.parent_path])
            )

    for buf in (parents_buf, children_buf, siblings_buf):
        for item in buf:
            if len(out) >= max_expand:
                return out
            if item[0].id not in seen_ids:
                out.append(item)
                seen_ids.add(item[0].id)

    return out


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
      ``node_path``. This "pulls the subtree down".
    - **Parents** — the chunk whose ``node_path`` matches a retrieved
      ``parent_path``. Provides context for the hit.
    - **Siblings** — chunks sharing the same ``parent_path`` as a retrieved
      chunk. Useful when one method of a class is relevant and you want the
      whole class API surface.

    All three behaviours are enabled by default. Each can be disabled
    independently.

    Parameters
    ----------
    results : list[Chunk]
        Initial retrieval results (k-nearest by cosine similarity).
    corpus : list[Chunk]
        The full indexed corpus to expand from.
    include_parents / include_children / include_siblings : bool
        Toggle which structural relations to pull in.
    max_expand : int
        Hard cap on the total number of chunks returned.

    Returns
    -------
    list[Chunk]
        Deduplicated, expanded list. Order: original results first, then
        parents, then children, then siblings (corpus-order within each group).
    """
    return [
        c
        for c, _, _ in _expand_with_metadata(
            results,
            corpus,
            include_parents=include_parents,
            include_children=include_children,
            include_siblings=include_siblings,
            max_expand=max_expand,
        )
    ]


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
        dim: int = self._encoder.get_embedding_dimension()  # type: ignore[assignment]

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

    def index_texts(self, texts: list[str], root_slug: str = "text") -> int:
        """
        Index a plain list of strings as flat, root-level chunks.

        Convenience method for quick experimentation. Each string becomes its
        own chunk at depth 0 with a unique root-level ``node_path``
        (``text0``, ``text1``, …) and **no parent**. Because they share no
        common parent, ``subtree_expand`` will not link them as siblings —
        a flat list of strings is treated as a flat list of strings.

        For real hierarchical indexing, use ``index_path`` (Python source)
        or ``index_chunks`` with ``Chunk`` objects from a chunker.

        Returns
        -------
        int
            Number of chunks added.
        """
        if not texts:
            return 0
        id_offset = len(self._chunks)
        chunks = [
            Chunk(
                id=id_offset + i,
                text=t,
                depth=0,
                node_path=f"{root_slug}{i}",
                source_file="",
                start_line=0,
                end_line=0,
            )
            for i, t in enumerate(texts)
        ]
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
        rescore_after_expand: bool = False,
        return_metadata: bool = False,
    ) -> list[Chunk] | list[RetrievalResult]:
        """
        Retrieve the most relevant chunks for *text*.

        Parameters
        ----------
        text : str
            Natural-language query.
        k : int
            Number of nearest neighbour **seeds** to fetch before expansion.
            If ``expand_subtree=True`` (the default) the final list may be
            longer — every seed's parent/siblings/children are pulled in
            structurally. Pass ``expand_subtree=False`` if you want strictly
            ``k`` chunks back.
        expand_subtree : bool
            When *True* (default), apply ``subtree_expand`` to the k results.
        include_parents / include_children / include_siblings : bool
            Fine-grained control over which relations are expanded.
            Only used when ``expand_subtree=True``.
        max_expand : int
            Hard cap on the total chunks returned after expansion.
        rescore_after_expand : bool
            When *True*, re-encode every expanded chunk against the query and
            sort the final list by cosine similarity descending. Adds one
            extra forward pass through the encoder (~1 ms for ~20 chunks).

            Why this matters: structural expansion adds parents/siblings/
            children in document order, not semantic order. If the answer
            lives in a *sibling* of the top FAISS hit, it can end up at the
            bottom of the list. Rescoring promotes it to the top.
        return_metadata : bool
            When *True*, return ``list[RetrievalResult]`` carrying per-chunk
            ``score``, ``relation`` (``"seed"``/``"parent"``/``"sibling"``/
            ``"child"``), and ``seed_path`` (the seed that pulled it in).
            When *False* (default), return ``list[Chunk]`` for backward
            compatibility.

        Returns
        -------
        list[Chunk] | list[RetrievalResult]
        """
        if not self._chunks:
            raise RuntimeError("Index is empty — call .index_path() first.")

        q_vec: np.ndarray = self._encoder.encode(  # type: ignore[assignment]
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        distances, ids = self._index.search(q_vec, k)
        seed_similarities: dict[int, float] = {}
        seeds: list[Chunk] = []
        for dist, idx in zip(distances[0], ids[0]):
            if idx == -1:
                continue
            seeds.append(self._chunks[idx])
            seed_similarities[self._chunks[idx].id] = float(1.0 - dist)

        if not seeds:
            return []

        if expand_subtree:
            tagged = _expand_with_metadata(
                seeds,
                self._chunks,
                include_parents=include_parents,
                include_children=include_children,
                include_siblings=include_siblings,
                max_expand=max_expand,
            )
        else:
            tagged = [(c, "seed", "") for c in seeds]

        # Build RetrievalResult objects with per-chunk scores
        results = [
            RetrievalResult(
                chunk=chunk,
                score=seed_similarities.get(chunk.id, float("nan")),
                relation=relation,
                seed_path=seed_path,
            )
            for chunk, relation, seed_path in tagged
        ]

        if rescore_after_expand and results:
            chunk_vecs: np.ndarray = self._encoder.encode(  # type: ignore[assignment]
                [r.chunk.text for r in results],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            sims = _cosine_similarity(q_vec[0], chunk_vecs)
            for r, s in zip(results, sims):
                r.score = float(s)
            results.sort(key=lambda r: r.score, reverse=True)

        if return_metadata:
            return results
        return [r.chunk for r in results]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        return self._index.ntotal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Cosine similarity of *query* against each row of *corpus*."""
    q = query / max(float(np.linalg.norm(query)), 1e-12)
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    c = corpus / norms
    return c @ q
