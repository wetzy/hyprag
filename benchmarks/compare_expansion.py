"""
benchmarks.compare_expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Measure the contribution of subtree expansion on a Python codebase.

Three conditions on identical embeddings and corpus:
    1. FAISS              – flat cosine, no expansion          (baseline)
    2. FAISS + expand     – flat cosine, with subtree_expand   ← product
    3. Hybrid + expand    – BM25 + FAISS via RRF + expand

Geometry context
----------------
An earlier version of this comparison included a Poincaré-ball arm. Across
two corpora (CPython stdlib and GDPR legal text) the ball produced
numerically identical results at ~13× the latency. That arm has been
removed; the geometry verdict is settled.

Usage
-----
    python -m benchmarks.compare_expansion --cpython-lib cpython/Lib
    python -m benchmarks.compare_expansion --cpython-lib cpython/Lib \\
        --encoder BAAI/bge-base-en-v1.5 --k 10
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from hyprag.bm25 import BM25Index
from hyprag.chunker import Chunk, HierarchicalChunker
from hyprag.hybrid import reciprocal_rank_fusion
from hyprag.retriever import subtree_expand

from benchmarks.queries import QUERIES, is_relevant

EXCLUDE_DIRS = {"test", "tests", "idlelib", "turtledemo", "__pycache__"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    recall: float
    precision: float
    avg_result_size: float
    avg_latency_ms: float


@dataclass
class ComparisonReport:
    encoder_model: str
    k: int
    n_queries: int
    corpus_chunks: int
    faiss:        ConditionResult
    faiss_expand: ConditionResult
    hybrid_expand: ConditionResult
    per_query: list[dict]

    def verdict(self) -> str:
        lift = (self.faiss_expand.recall - self.faiss.recall) / max(self.faiss.recall, 1e-9)
        hybrid_gap = self.hybrid_expand.recall - self.faiss_expand.recall
        bm25_note = (
            "BM25 helps" if hybrid_gap > 0.02
            else "BM25 hurts" if hybrid_gap < -0.02
            else "BM25 neutral"
        )
        return (
            f"Expansion lift: {lift:+.1%} (FAISS → FAISS+expand). "
            f"Hybrid Δ vs FAISS+expand: {hybrid_gap:+.3f} — {bm25_note}."
        )


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(lib_path: Path) -> list[Chunk]:
    chunker = HierarchicalChunker()
    chunks: list[Chunk] = []
    for py_file in sorted(lib_path.rglob("*.py")):
        if any(part in EXCLUDE_DIRS for part in py_file.parts):
            continue
        file_chunks = chunker.chunk_file(py_file)
        for c in file_chunks:
            c.id += len(chunks)
        chunks.extend(file_chunks)
    return chunks


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def run_comparison(
    chunks: list[Chunk],
    encoder_model: str,
    k: int,
) -> ComparisonReport:
    from sentence_transformers import SentenceTransformer

    print(f"Encoding {len(chunks):,} chunks with {encoder_model}...")
    model = SentenceTransformer(encoder_model, trust_remote_code=True)
    texts = [c.text for c in chunks]
    vecs: np.ndarray = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    dim = vecs.shape[1]

    # Cosine similarity → normalise then use inner product
    vecs = vecs.astype(np.float32)
    faiss.normalize_L2(vecs)

    print("Building FAISS index...")
    fi = faiss.IndexFlatIP(dim)
    fi.add(vecs)

    print("Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(texts)

    n_cand = max(k * 4, 20)

    per_query: list[dict] = []
    sums: dict[str, float] = {
        "r_f": 0.0, "p_f": 0.0, "sz_f": 0.0, "t_f": 0.0,
        "r_fe": 0.0, "p_fe": 0.0, "sz_fe": 0.0, "t_fe": 0.0,
        "r_hybe": 0.0, "p_hybe": 0.0, "sz_hybe": 0.0, "t_hybe": 0.0,
    }

    print(f"\nRunning {len(QUERIES)} queries at K={k}...\n")

    for q in QUERIES:
        q_vec = model.encode([q.text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_vec)

        n_relevant = max(
            sum(1 for c in chunks if is_relevant(c.node_path, q.ground_truth_prefixes)),
            1,
        )

        # ── Condition 1: FAISS, no expansion ──────────────────────────────
        t0 = time.perf_counter()
        _, fids = fi.search(q_vec, k)
        f_chunks = [chunks[i] for i in fids[0] if i >= 0]
        t_f = (time.perf_counter() - t0) * 1000
        f_hit = sum(1 for c in f_chunks if is_relevant(c.node_path, q.ground_truth_prefixes))

        # ── Condition 2: FAISS + subtree expansion ─────────────────────────
        t0 = time.perf_counter()
        _, fids2 = fi.search(q_vec, k)
        fe_seeds = [chunks[i] for i in fids2[0] if i >= 0]
        fe_chunks = subtree_expand(fe_seeds, chunks, max_expand=n_cand)
        t_fe = (time.perf_counter() - t0) * 1000
        fe_hit = sum(1 for c in fe_chunks if is_relevant(c.node_path, q.ground_truth_prefixes))

        # ── Condition 3: Hybrid (BM25 + FAISS via RRF) + expansion ────────
        t0 = time.perf_counter()
        _, bm25_ids = bm25.search(q.text, n_cand)
        _, sem_ids = fi.search(q_vec, n_cand)
        sem_ranked = [idx for idx in sem_ids[0] if idx >= 0]
        fused = reciprocal_rank_fusion([sem_ranked, list(bm25_ids)])
        hyb_seeds = [chunks[doc_id] for doc_id, _ in fused[:k]]
        hyb_chunks = subtree_expand(hyb_seeds, chunks, max_expand=n_cand)
        t_hybe = (time.perf_counter() - t0) * 1000
        hyb_hit = sum(1 for c in hyb_chunks if is_relevant(c.node_path, q.ground_truth_prefixes))

        per_query.append({
            "query": q.text,
            "n_relevant": n_relevant,
            "faiss":         {"recall": round(f_hit / n_relevant, 3),   "precision": round(f_hit / max(len(f_chunks), 1), 3),   "n_results": len(f_chunks),   "latency_ms": round(t_f, 2)},
            "faiss_expand":  {"recall": round(fe_hit / n_relevant, 3),  "precision": round(fe_hit / max(len(fe_chunks), 1), 3),  "n_results": len(fe_chunks),  "latency_ms": round(t_fe, 2)},
            "hybrid_expand": {"recall": round(hyb_hit / n_relevant, 3), "precision": round(hyb_hit / max(len(hyb_chunks), 1), 3), "n_results": len(hyb_chunks), "latency_ms": round(t_hybe, 2)},
        })

        sums["r_f"]    += f_hit / n_relevant;    sums["p_f"]    += f_hit / max(len(f_chunks), 1);    sums["sz_f"]    += len(f_chunks);    sums["t_f"]    += t_f
        sums["r_fe"]   += fe_hit / n_relevant;   sums["p_fe"]   += fe_hit / max(len(fe_chunks), 1);  sums["sz_fe"]   += len(fe_chunks);   sums["t_fe"]   += t_fe
        sums["r_hybe"] += hyb_hit / n_relevant;  sums["p_hybe"] += hyb_hit / max(len(hyb_chunks), 1); sums["sz_hybe"] += len(hyb_chunks); sums["t_hybe"] += t_hybe

    n = len(QUERIES)
    def cr(rk, pk, szk, tk) -> ConditionResult:
        return ConditionResult(
            recall=round(sums[rk] / n, 3),
            precision=round(sums[pk] / n, 3),
            avg_result_size=round(sums[szk] / n, 1),
            avg_latency_ms=round(sums[tk] / n, 2),
        )

    return ComparisonReport(
        encoder_model=encoder_model,
        k=k,
        n_queries=n,
        corpus_chunks=len(chunks),
        faiss=         cr("r_f",    "p_f",    "sz_f",    "t_f"),
        faiss_expand=  cr("r_fe",   "p_fe",   "sz_fe",   "t_fe"),
        hybrid_expand= cr("r_hybe", "p_hybe", "sz_hybe", "t_hybe"),
        per_query=per_query,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(r: ComparisonReport) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  Expansion Comparison  |  K={r.k}  |  {r.n_queries} queries  |  {r.corpus_chunks:,} chunks")
    print(f"  Encoder: {r.encoder_model}")
    print(sep)
    print(f"  {'Condition':<28} {'Recall':>8} {'Precision':>10} {'Results':>8} {'Latency':>10}")
    print(f"  {'─'*28} {'─'*8} {'─'*10} {'─'*8} {'─'*10}")

    def row(label, cr: ConditionResult, *, highlight=False):
        tag = " ◀" if highlight else ""
        print(
            f"  {label:<28} {cr.recall:>8.3f} {cr.precision:>10.3f}"
            f" {cr.avg_result_size:>8.1f} {cr.avg_latency_ms:>9.1f}ms{tag}"
        )

    row("FAISS (baseline)", r.faiss)
    row("FAISS + expand",   r.faiss_expand, highlight=True)
    row("Hybrid (BM25+FAISS)+exp", r.hybrid_expand)
    print(sep)

    print(f"\n  VERDICT: {r.verdict()}")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure subtree-expansion contribution on a Python codebase."
    )
    p.add_argument("--cpython-lib", type=Path, required=True,
                   help="Path to cpython/Lib directory")
    p.add_argument("--encoder", default="BAAI/bge-base-en-v1.5",
                   help="sentence-transformers model (default: BAAI/bge-base-en-v1.5)")
    p.add_argument("--k", type=int, default=5,
                   help="Number of seed chunks before expansion (default: 5)")
    p.add_argument("--out", type=Path, default=Path("benchmarks/results/comparison.json"),
                   help="Output JSON path")
    args = p.parse_args()

    if not args.cpython_lib.exists():
        raise SystemExit(f"Path not found: {args.cpython_lib}")

    print(f"Loading corpus from {args.cpython_lib}...")
    chunks = load_corpus(args.cpython_lib)
    print(f"  → {len(chunks):,} chunks loaded")

    report = run_comparison(chunks, args.encoder, args.k)
    print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"Results saved to {args.out}\n")


if __name__ == "__main__":
    main()
