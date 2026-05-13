"""
hyprag — Hyperbolic RAG: nearest-neighbour retrieval on the Poincaré ball.

Quick start
-----------
    from hyprag import HypragRetriever

    retriever = HypragRetriever()
    retriever.index_path("./myproject")
    results = retriever.query("how is the parser initialised?", k=5)

Low-level API (FAISS drop-in)
------------------------------
    from hyprag import PoincareBallIndex
    import numpy as np

    index = PoincareBallIndex(dim=384)
    index.add(vectors, depths=[0, 1, 1, 2, 2, 2])
    distances, ids = index.search(query_vec, k=10)
"""

from hyprag.chunker import Chunk, HierarchicalChunker
from hyprag.index import PoincareBallIndex
from hyprag.retriever import HypragRetriever, subtree_expand
from hyprag.bm25 import BM25Index
from hyprag.hybrid import HybridRetriever, reciprocal_rank_fusion
from hyprag.summarize import ChunkSummarizer, apply_summaries, load_summaries

__all__ = [
    "Chunk",
    "HierarchicalChunker",
    "PoincareBallIndex",
    "HypragRetriever",
    "subtree_expand",
    "BM25Index",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "ChunkSummarizer",
    "apply_summaries",
    "load_summaries",
]
__version__ = "0.4.0"
