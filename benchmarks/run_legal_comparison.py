"""
benchmarks.run_legal_comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Three-condition expansion study on the GDPR legal corpus:

    1. FAISS              – flat cosine, no expansion          (baseline)
    2. FAISS + expand     – flat cosine, with subtree_expand   ← product
    3. Hybrid + expand    – BM25 + FAISS via RRF + expand

History: an earlier version included a Poincaré-ball arm. It produced
numerically identical results to FAISS at ~13× the latency on GDPR (delta
= 0.000 across all 20 queries). That arm has been removed; see
``benchmarks/results/legal_comparison.json`` for the historical record.

Usage
-----
    python -m benchmarks.run_legal_comparison
    python -m benchmarks.run_legal_comparison --html-path /path/to/gdpr.html
    python -m benchmarks.run_legal_comparison --encoder BAAI/bge-base-en-v1.5 --k 5
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
from hyprag.chunker import Chunk
from hyprag.chunkers.legal import GDPRChunker
from hyprag.hybrid import reciprocal_rank_fusion
from hyprag.retriever import subtree_expand

from benchmarks.gdpr_queries import GDPR_QUERIES, is_relevant


@dataclass
class ConditionResult:
    recall: float
    precision: float
    avg_result_size: float
    avg_latency_ms: float


@dataclass
class LegalComparisonReport:
    corpus: str
    encoder_model: str
    k: int
    n_queries: int
    corpus_chunks: int
    faiss:         ConditionResult
    faiss_expand:  ConditionResult
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
            f"Expansion lift: {lift:+.1%}. "
            f"Hybrid Δ vs FAISS+expand: {hybrid_gap:+.3f} — {bm25_note}."
        )


def run_comparison(
    chunks: list[Chunk],
    encoder_model: str,
    k: int,
) -> LegalComparisonReport:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Encoding {len(chunks):,} chunks on {device.upper()}...")

    model = SentenceTransformer(encoder_model, trust_remote_code=True, device=device)
    texts = [c.text for c in chunks]
    vecs: np.ndarray = model.encode(
        texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True
    )
    vecs = vecs.astype(np.float32)
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]

    print("  Building FAISS index...")
    fi = faiss.IndexFlatIP(dim)
    fi.add(vecs)

    print("  Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(texts)

    n_cand = max(k * 4, 20)
    per_query: list[dict] = []
    sums: dict[str, float] = {key: 0.0 for key in [
        "r_f","p_f","sz_f","t_f",
        "r_fe","p_fe","sz_fe","t_fe",
        "r_hybe","p_hybe","sz_hybe","t_hybe",
    ]}

    print(f"\n  Running {len(GDPR_QUERIES)} queries at K={k}...\n")

    for q in GDPR_QUERIES:
        q_vec = model.encode([q.text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_vec)

        n_rel = max(
            sum(1 for c in chunks if is_relevant(c.node_path, q.ground_truth_prefixes)),
            1,
        )

        # 1. FAISS — no expansion
        t0 = time.perf_counter()
        _, fids = fi.search(q_vec, k)
        f_chunks = [chunks[i] for i in fids[0] if i >= 0]
        t_f = (time.perf_counter() - t0) * 1000
        f_hit = sum(1 for c in f_chunks if is_relevant(c.node_path, q.ground_truth_prefixes))

        # 2. FAISS + subtree expansion
        t0 = time.perf_counter()
        _, fids2 = fi.search(q_vec, k)
        fe_seeds = [chunks[i] for i in fids2[0] if i >= 0]
        fe_chunks = subtree_expand(fe_seeds, chunks, max_expand=n_cand)
        t_fe = (time.perf_counter() - t0) * 1000
        fe_hit = sum(1 for c in fe_chunks if is_relevant(c.node_path, q.ground_truth_prefixes))

        # 3. Hybrid (BM25 + FAISS via RRF) + expansion
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
            "ground_truth": q.ground_truth_prefixes,
            "n_relevant": n_rel,
            "faiss":         {"recall": round(f_hit/n_rel,3),   "precision": round(f_hit/max(len(f_chunks),1),3),   "n_results": len(f_chunks),   "latency_ms": round(t_f,2)},
            "faiss_expand":  {"recall": round(fe_hit/n_rel,3),  "precision": round(fe_hit/max(len(fe_chunks),1),3),  "n_results": len(fe_chunks),  "latency_ms": round(t_fe,2)},
            "hybrid_expand": {"recall": round(hyb_hit/n_rel,3), "precision": round(hyb_hit/max(len(hyb_chunks),1),3), "n_results": len(hyb_chunks), "latency_ms": round(t_hybe,2)},
        })

        sums["r_f"]    += f_hit/n_rel;    sums["p_f"]    += f_hit/max(len(f_chunks),1);    sums["sz_f"]    += len(f_chunks);    sums["t_f"]    += t_f
        sums["r_fe"]   += fe_hit/n_rel;   sums["p_fe"]   += fe_hit/max(len(fe_chunks),1);  sums["sz_fe"]   += len(fe_chunks);   sums["t_fe"]   += t_fe
        sums["r_hybe"] += hyb_hit/n_rel;  sums["p_hybe"] += hyb_hit/max(len(hyb_chunks),1); sums["sz_hybe"] += len(hyb_chunks); sums["t_hybe"] += t_hybe

    n = len(GDPR_QUERIES)
    def cr(r,p,sz,t):
        return ConditionResult(round(sums[r]/n,3), round(sums[p]/n,3), round(sums[sz]/n,1), round(sums[t]/n,2))

    return LegalComparisonReport(
        corpus="GDPR (EU 2016/679)",
        encoder_model=encoder_model,
        k=k,
        n_queries=n,
        corpus_chunks=len(chunks),
        faiss=         cr("r_f","p_f","sz_f","t_f"),
        faiss_expand=  cr("r_fe","p_fe","sz_fe","t_fe"),
        hybrid_expand= cr("r_hybe","p_hybe","sz_hybe","t_hybe"),
        per_query=per_query,
    )


def print_report(r: LegalComparisonReport) -> None:
    sep = "─" * 76
    print(f"\n{sep}")
    print(f"  Legal Document Comparison — {r.corpus}")
    print(f"  K={r.k}  |  {r.n_queries} queries  |  {r.corpus_chunks:,} chunks  |  {r.encoder_model}")
    print(sep)
    print(f"  {'Condition':<30} {'Recall':>8} {'Precision':>10} {'Results':>8} {'Latency':>10}")
    print(f"  {'─'*30} {'─'*8} {'─'*10} {'─'*8} {'─'*10}")

    def row(label, c, mark=""):
        print(f"  {label:<30} {c.recall:>8.3f} {c.precision:>10.3f} {c.avg_result_size:>8.1f} {c.avg_latency_ms:>9.1f}ms{mark}")

    row("FAISS (baseline)",            r.faiss)
    row("FAISS + expand",              r.faiss_expand, "  ◀ best")
    row("Hybrid (BM25+FAISS)+expand",  r.hybrid_expand)
    print(sep)

    print(f"\n  VERDICT: {r.verdict()}")
    print(sep)

    print(f"\n  Per-query breakdown:")
    print(f"  {'Query':<45} {'GT articles':<20} {'FAISS':>6} {'F+exp':>6} {'Hyb+e':>6}")
    print(f"  {'─'*45} {'─'*20} {'─'*6} {'─'*6} {'─'*6}")
    for row_data in r.per_query:
        q_short = row_data["query"][:43] + ".." if len(row_data["query"]) > 43 else row_data["query"]
        gt = ", ".join(p.split(".")[-1] for p in row_data["ground_truth"])
        print(
            f"  {q_short:<45} {gt:<20}"
            f" {row_data['faiss']['recall']:>6.2f}"
            f" {row_data['faiss_expand']['recall']:>6.2f}"
            f" {row_data['hybrid_expand']['recall']:>6.2f}"
        )
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare retrieval strategies on GDPR legal text."
    )
    p.add_argument("--html-path", type=Path, default=None,
                   help="Local GDPR HTML file (downloads from EUR-Lex if omitted)")
    p.add_argument("--encoder", default="BAAI/bge-base-en-v1.5",
                   help="sentence-transformers model")
    p.add_argument("--k", type=int, default=5,
                   help="Seed chunks before expansion (default: 5)")
    p.add_argument("--out", type=Path,
                   default=Path("benchmarks/results/legal_comparison.json"))
    args = p.parse_args()

    print("Loading GDPR corpus...")
    chunker = GDPRChunker()
    chunks = chunker.load(html_path=args.html_path)
    print(f"  → {len(chunks):,} chunks at depths: "
          + str({d: sum(1 for c in chunks if c.depth == d) for d in sorted({c.depth for c in chunks})}))

    report = run_comparison(chunks, args.encoder, args.k)
    print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"Saved → {args.out}\n")


if __name__ == "__main__":
    main()
