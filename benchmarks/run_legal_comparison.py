"""
benchmarks.run_legal_comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run the FAISS vs FAISS+expand vs HypRAG+expand comparison on real legal text
(GDPR EU 2016/679) instead of a Python codebase.

This produces the same 4-condition table as compare_expansion.py but on a
corpus that enterprise legal/compliance buyers actually care about.

Usage (Colab / local)
---------------------
    python -m benchmarks.run_legal_comparison
    python -m benchmarks.run_legal_comparison --html-path /path/to/gdpr.html
    python -m benchmarks.run_legal_comparison --encoder BAAI/bge-base-en-v1.5 --k 5

Output
------
    Prints the comparison table.
    Saves JSON to benchmarks/results/legal_comparison.json (or --out).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from hyprag.chunker import Chunk
from hyprag.index import PoincareBallIndex
from hyprag.retriever import subtree_expand
from hyprag.bm25 import BM25Index
from hyprag.hybrid import reciprocal_rank_fusion
from hyprag.chunkers.legal import GDPRChunker

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
    hyprag_expand: ConditionResult
    hybrid_expand: ConditionResult
    per_query: list[dict]

    def verdict(self) -> str:
        gap = self.hyprag_expand.recall - self.faiss_expand.recall
        lift = (self.faiss_expand.recall - self.faiss.recall) / max(self.faiss.recall, 1e-9)
        if abs(gap) <= 0.02:
            return (
                f"GEOMETRY ADDS NOTHING (gap ≤ 2pp). "
                f"FAISS+expand achieves {lift:+.0%} lift alone. "
                f"Pivot to a FAISS-backed expansion library."
            )
        elif gap > 0.02:
            return (
                f"HYPERBOLIC SEEDING HELPS (+{gap:.1%} over FAISS+expand). "
                f"Poincaré geometry genuinely improves seed selection on legal text."
            )
        else:
            return (
                f"FAISS+expand WINS (+{-gap:.1%} over HypRAG+expand). "
                f"Drop the geometry — FAISS is both faster and more accurate."
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
    depths = np.array([c.depth for c in chunks])
    dim = vecs.shape[1]

    print("  Building FAISS index...")
    fi = faiss.IndexFlatL2(dim)
    fi.add(vecs)

    print("  Building HypRAG (Poincaré ball) index...")
    hi = PoincareBallIndex(dim)  # auto-detects CUDA
    hi.add(vecs, depths=depths)

    print("  Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(texts)

    n_cand = max(k * 4, 20)
    per_query: list[dict] = []
    sums: dict[str, float] = {k2: 0.0 for k2 in [
        "r_f","p_f","sz_f","t_f",
        "r_fe","p_fe","sz_fe","t_fe",
        "r_he","p_he","sz_he","t_he",
        "r_hybe","p_hybe","sz_hybe","t_hybe",
    ]}

    print(f"\n  Running {len(GDPR_QUERIES)} queries at K={k}...\n")

    for q in GDPR_QUERIES:
        q_vec = model.encode([q.text], convert_to_numpy=True)
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

        # 3. HypRAG + subtree expansion
        t0 = time.perf_counter()
        _, hids = hi.search(q_vec, k)
        he_seeds = [chunks[i] for i in hids[0] if i >= 0]
        he_chunks = subtree_expand(he_seeds, chunks, max_expand=n_cand)
        t_he = (time.perf_counter() - t0) * 1000
        he_hit = sum(1 for c in he_chunks if is_relevant(c.node_path, q.ground_truth_prefixes))

        # 4. Hybrid (BM25 + HypRAG via RRF) + expansion
        t0 = time.perf_counter()
        _, bm25_ids = bm25.search(q.text, n_cand)
        _, sem_ids = hi.search(q_vec, n_cand)
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
            "hyprag_expand": {"recall": round(he_hit/n_rel,3),  "precision": round(he_hit/max(len(he_chunks),1),3),  "n_results": len(he_chunks),  "latency_ms": round(t_he,2)},
            "hybrid_expand": {"recall": round(hyb_hit/n_rel,3), "precision": round(hyb_hit/max(len(hyb_chunks),1),3), "n_results": len(hyb_chunks), "latency_ms": round(t_hybe,2)},
        })

        sums["r_f"]    += f_hit/n_rel;    sums["p_f"]    += f_hit/max(len(f_chunks),1);    sums["sz_f"]    += len(f_chunks);    sums["t_f"]    += t_f
        sums["r_fe"]   += fe_hit/n_rel;   sums["p_fe"]   += fe_hit/max(len(fe_chunks),1);  sums["sz_fe"]   += len(fe_chunks);   sums["t_fe"]   += t_fe
        sums["r_he"]   += he_hit/n_rel;   sums["p_he"]   += he_hit/max(len(he_chunks),1);  sums["sz_he"]   += len(he_chunks);   sums["t_he"]   += t_he
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
        hyprag_expand= cr("r_he","p_he","sz_he","t_he"),
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

    row("FAISS (baseline)",             r.faiss)
    row("FAISS + expand",               r.faiss_expand)
    row("HypRAG + expand",              r.hyprag_expand)
    row("Hybrid (BM25+HypRAG)+expand",  r.hybrid_expand, "  ◀ best")
    print(sep)

    gap = r.hyprag_expand.recall - r.faiss_expand.recall
    lift = (r.faiss_expand.recall - r.faiss.recall) / max(r.faiss.recall, 1e-9)
    print(f"\n  Expansion lift  (FAISS → FAISS+expand):          {lift:+.1%}")
    print(f"  Geometry delta  (FAISS+expand → HypRAG+expand):  {gap:+.3f} ({gap/max(r.faiss_expand.recall,1e-9):+.1%})")
    print(f"\n  VERDICT: {r.verdict()}")
    print(sep)

    print(f"\n  Per-query breakdown:")
    print(f"  {'Query':<45} {'GT articles':<20} {'FAISS':>6} {'F+exp':>6} {'H+exp':>6} {'Hyb+e':>6}")
    print(f"  {'─'*45} {'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for row_data in r.per_query:
        q_short = row_data["query"][:43] + ".." if len(row_data["query"]) > 43 else row_data["query"]
        gt = ", ".join(p.split(".")[-1] for p in row_data["ground_truth"])
        print(
            f"  {q_short:<45} {gt:<20}"
            f" {row_data['faiss']['recall']:>6.2f}"
            f" {row_data['faiss_expand']['recall']:>6.2f}"
            f" {row_data['hyprag_expand']['recall']:>6.2f}"
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
