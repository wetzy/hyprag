"""
hyprag — Hierarchical RAG: FAISS retrieval + subtree expansion over a
parent/child chunk hierarchy.

Quick start
-----------
The simplest possible flow — point ``index()`` at anything and ask::

    from hyprag import HypragRetriever

    r = HypragRetriever()
    r.index("https://en.wikipedia.org/wiki/General_Data_Protection_Regulation")
    results = r.query("What is the maximum fine for a severe violation?", k=3)

``index()`` auto-detects what you gave it:

    r.index("./contract.pdf")            # PDF
    r.index("./notes.md")                # markdown
    r.index("./codebase/")               # directory of source code
    r.index("plain text content here")   # raw string
    r.index(["doc 1", "doc 2"])          # list of strings

For richer downstream control, request metadata + rescoring::

    results = r.query(
        "...",
        k=5,
        return_metadata=True,
        rescore_after_expand=True,
        min_score=0.55,
    )
    for res in results:
        print(res.chunk.node_path, res.score, res.relation)

Explicit chunkers remain available when you want direct control:
``HTMLChunker``, ``MarkdownChunker``, ``PDFChunker``, ``TextChunker``,
``GDPRChunker``.

Low-level FAISS index (drop-in)
-------------------------------
    from hyprag import FaissIndex
    import numpy as np

    index = FaissIndex(dim=768)
    index.add(vectors)
    distances, ids = index.search(query_vec, k=10)
"""

from hyprag.bm25 import BM25Index
from hyprag.chunker import Chunk, HierarchicalChunker
from hyprag.chunkers import (
    GDPRChunker,
    HTMLChunker,
    MarkdownChunker,
    PDFChunker,
    TextChunker,
)
from hyprag.faiss_index import FaissIndex
from hyprag.hybrid import HybridRetriever, reciprocal_rank_fusion
from hyprag.retriever import HypragRetriever, RetrievalResult, subtree_expand
from hyprag.sources import chunks_from_source, fetch_url, sniff_kind
from hyprag.summarize import ChunkSummarizer, apply_summaries, load_summaries

__all__ = [
    "Chunk",
    "HierarchicalChunker",
    "FaissIndex",
    "HypragRetriever",
    "RetrievalResult",
    "subtree_expand",
    "BM25Index",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "ChunkSummarizer",
    "apply_summaries",
    "load_summaries",
    "GDPRChunker",
    "HTMLChunker",
    "MarkdownChunker",
    "PDFChunker",
    "TextChunker",
    "chunks_from_source",
    "fetch_url",
    "sniff_kind",
]
__version__ = "0.7.0"
