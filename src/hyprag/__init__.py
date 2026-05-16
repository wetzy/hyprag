"""
hyprag — Hierarchical RAG: FAISS retrieval + subtree expansion over a
parent/child chunk hierarchy.

Quick start
-----------
    from hyprag import HypragRetriever

    retriever = HypragRetriever()                  # BGE-base-en-v1.5
    retriever.index_path("./myproject")
    results = retriever.query("how is the parser initialised?", k=5)

Pre-chunked documents (e.g. GDPR via ``GDPRChunker``)
-----------------------------------------------------
    from hyprag import HypragRetriever
    from hyprag.chunkers import GDPRChunker

    chunks = GDPRChunker().load(html_path="gdpr_corpus.html")
    retriever = HypragRetriever()
    retriever.index_chunks(chunks)
    results = retriever.query(
        "what rights do individuals have to access their personal data", k=5
    )

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
from hyprag.faiss_index import FaissIndex
from hyprag.hybrid import HybridRetriever, reciprocal_rank_fusion
from hyprag.retriever import HypragRetriever, subtree_expand
from hyprag.summarize import ChunkSummarizer, apply_summaries, load_summaries

__all__ = [
    "Chunk",
    "HierarchicalChunker",
    "FaissIndex",
    "HypragRetriever",
    "subtree_expand",
    "BM25Index",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "ChunkSummarizer",
    "apply_summaries",
    "load_summaries",
]
__version__ = "0.5.1"
