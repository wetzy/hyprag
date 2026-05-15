"""
benchmarks.specificity_ablation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Path C experiment: does the Poincaré ball's radial dimension carry useful
information when we deliberately encode hierarchy depth into the radius and
match it to a query-specificity heuristic?

Four conditions on the same corpus and candidate pool (FAISS top-N):

  C0  FAISS top-K → expand                      (baseline)
  C1  FAISS top-N → depth-blind rerank → top-K → expand
        (sanity check: re-ranking a wider pool by raw distance only)
  C2  FAISS top-N → classical depth-match rerank → top-K → expand
        (is depth-matching useful at all, by any combination?)
  C3  FAISS top-N → hyperbolic geodesic rerank → top-K → expand
        (does the curved-space combination beat the linear one?)

Verdict guide
-------------
  C2 ≈ C0:  depth signal has no value on this corpus.  Path C fails.
  C2 > C0:  depth is a useful feature; check whether C3 ≥ C2.
  C3 > C2:  the geometry adds something the linear combo can't reach —
            first real evidence the Poincaré ball pays rent.

Usage
-----
    python -m benchmarks.specificity_ablation \\
        --cpython-lib /content/cpython/Lib \\
        --encoder BAAI/bge-base-en-v1.5 \\
        --k 5 --n-cand 20 --alpha 0.3
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from hyprag.chunker import HierarchicalChunker, Chunk
from hyprag.retriever import subtree_expand
from hyprag.specificity import (
    infer_query_specificity, classical_rerank, hyperbolic_rerank,
)

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
class AblationReport:
    encoder_model: str
    k: int
    n_cand: int
    alpha: float
    ball_scale: float
    min_norm: float
    max_depth: int
    n_queries: int
    corpus_chunks: int
    c0_baseline:     ConditionResult
    c1_blind_rerank: ConditionResult
    c2_classical:    ConditionResult
    c3_hyperbolic:   ConditionResult
    per_query: list[dict]

    def verdict(self) -> str:
        c2_lift = self.c2_classical.recall - self.c0_baseline.recall
        c3_lift = self.c3_hyperbolic.recall - self.c2_classical.recall
        bits = []
        if c2_lift > 0.01:
            bits.append(f"Depth feature: +{c2_lift:.3f} over baseline ({c2_lift / max(self.c0_baseline.recall, 1e-9):+.1%}).")
        elif c2_lift < -0.01:
            bits.append(f"Depth feature: −{-c2_lift:.3f} over baseline — hurts.")
        else:
            bits.append("Depth feature: neutral on this corpus.")
        if c3_lift > 0.01:
            bits.append(f"Geometry over classical: +{c3_lift:.3f} ({c3_lift / max(self.c2_classical.recall, 1e-9):+.1%}). Curvature carries unique signal.")
        elif c3_lift < -0.01:
            bits.append(f"Geometry over classical: −{-c3_lift:.3f}. Linear combo is better.")
        else:
            bits.append("Geometry over classical: neutral. No advantage over a linear combo.")
        return "  " + "\n  ".join(bits)


# ---------------------------------------------------------------------------
# Corpus loading (same as compare_expansion.py)
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
# Core ablation
# ---------------------------------------------------------------------------

def run_ablation(
    chunks: list[Chunk],
    encoder_model: str,
    k: int,
    n_cand: int,
    alpha: float,
    ball_scale: float,
    min_norm: float,
    max_depth: int,
) -> AblationReport:
    from sentence_transformers import SentenceTransformer

    print(f"Encoding {len(chunks):,} chunks with {encoder_model}...")
    model = SentenceTransformer(encoder_model, trust_remote_code=True)
    texts = [c.text for c in chunks]
    vecs: np.ndarray = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)
    depths = np.array([c.depth for c in chunks], dtype=np.int32)
    dim = vecs.shape[1]

    print("Building FAISS index...")
    fi = faiss.IndexFlatL2(dim)
    fi.add(vecs)

    per_query: list[dict] = []
    sums = {f"r_{c}": 0.0 for c in ("c0", "c1", "c2", "c3")}
    sums.update({f"p_{c}": 0.0 for c in ("c0", "c1", "c2", "c3")})
    sums.update({f"sz_{c}": 0.0 for c in ("c0", "c1", "c2", "c3")})
    sums.update({f"t_{c}": 0.0 for c in ("c0", "c1", "c2", "c3")})

    print(f"\nRunning {len(QUERIES)} queries  |  K={k}  |  N_cand={n_cand}  |  α={alpha}\n")

    for q in QUERIES:
        q_vec = model.encode([q.text], convert_to_numpy=True).astype(np.float32)
        target = infer_query_specificity(q.text, max_depth=max_depth)

        n_relevant = max(
            sum(1 for c in chunks if is_relevant(c.node_path, q.ground_truth_prefixes)),
            1,
        )

        # Single wide FAISS retrieval shared by all conditions
        cand_dists, cand_ids = fi.search(q_vec, n_cand)
        cand_ids = [int(i) for i in cand_ids[0] if i >= 0]
        cand_dists_list = list(cand_dists[0][: len(cand_ids)])
        cand_depths = [int(depths[i]) for i in cand_ids]
        cand_vecs = vecs[cand_ids]

        # ── C0: top-K (first k of the wide retrieval) → expand ─────────
        t0 = time.perf_counter()
        c0_seeds = [chunks[i] for i in cand_ids[:k]]
        c0_chunks = subtree_expand(c0_seeds, chunks, max_expand=n_cand)
        t_c0 = (time.perf_counter() - t0) * 1000

        # ── C1: re-sort by distance only (sanity — must equal C0) ──────
        t0 = time.perf_counter()
        c1_order = [cid for _, cid in sorted(zip(cand_dists_list, cand_ids))][:k]
        c1_seeds = [chunks[i] for i in c1_order]
        c1_chunks = subtree_expand(c1_seeds, chunks, max_expand=n_cand)
        t_c1 = (time.perf_counter() - t0) * 1000

        # ── C2: classical depth-match rerank → top-K → expand ──────────
        t0 = time.perf_counter()
        c2_order = classical_rerank(
            cand_ids, cand_dists_list, cand_depths,
            target_specificity=target, max_depth=max_depth, alpha=alpha,
        )[:k]
        c2_seeds = [chunks[i] for i in c2_order]
        c2_chunks = subtree_expand(c2_seeds, chunks, max_expand=n_cand)
        t_c2 = (time.perf_counter() - t0) * 1000

        # ── C3: hyperbolic geodesic rerank → top-K → expand ────────────
        t0 = time.perf_counter()
        c3_order = hyperbolic_rerank(
            q_vec[0], target,
            cand_ids, cand_vecs, cand_depths,
            ball_scale=ball_scale, min_norm=min_norm,
            max_depth=max_depth, curvature=1.0, device="cpu",
        )[:k]
        c3_seeds = [chunks[i] for i in c3_order]
        c3_chunks = subtree_expand(c3_seeds, chunks, max_expand=n_cand)
        t_c3 = (time.perf_counter() - t0) * 1000

        def hit(result_chunks):
            return sum(
                1 for c in result_chunks
                if is_relevant(c.node_path, q.ground_truth_prefixes)
            )

        h0, h1, h2, h3 = (hit(c) for c in (c0_chunks, c1_chunks, c2_chunks, c3_chunks))
        per_query.append({
            "query": q.text,
            "target_specificity": round(target, 2),
            "n_relevant": n_relevant,
            "c0": {"recall": round(h0 / n_relevant, 3), "n_results": len(c0_chunks)},
            "c1": {"recall": round(h1 / n_relevant, 3), "n_results": len(c1_chunks)},
            "c2": {"recall": round(h2 / n_relevant, 3), "n_results": len(c2_chunks)},
            "c3": {"recall": round(h3 / n_relevant, 3), "n_results": len(c3_chunks)},
            "ground_truth_depths_in_corpus": sorted({
                int(depths[i]) for i, c in enumerate(chunks)
                if is_relevant(c.node_path, q.ground_truth_prefixes)
            }),
        })

        for name, h, sz, latency in [
            ("c0", h0, len(c0_chunks), t_c0),
            ("c1", h1, len(c1_chunks), t_c1),
            ("c2", h2, len(c2_chunks), t_c2),
            ("c3", h3, len(c3_chunks), t_c3),
        ]:
            sums[f"r_{name}"]  += h / n_relevant
            sums[f"p_{name}"]  += h / max(sz, 1)
            sums[f"sz_{name}"] += sz
            sums[f"t_{name}"]  += latency

    n = len(QUERIES)
    def cr(prefix):
        return ConditionResult(
            recall=round(sums[f"r_{prefix}"] / n, 3),
            precision=round(sums[f"p_{prefix}"] / n, 3),
            avg_result_size=round(sums[f"sz_{prefix}"] / n, 1),
            avg_latency_ms=round(sums[f"t_{prefix}"] / n, 2),
        )

    return AblationReport(
        encoder_model=encoder_model,
        k=k, n_cand=n_cand, alpha=alpha,
        ball_scale=ball_scale, min_norm=min_norm, max_depth=max_depth,
        n_queries=n,
        corpus_chunks=len(chunks),
        c0_baseline=cr("c0"),
        c1_blind_rerank=cr("c1"),
        c2_classical=cr("c2"),
        c3_hyperbolic=cr("c3"),
        per_query=per_query,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(r: AblationReport) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  Specificity Ablation  |  K={r.k}  N_cand={r.n_cand}  α={r.alpha}  |  "
          f"{r.n_queries} queries  |  {r.corpus_chunks:,} chunks")
    print(f"  Encoder: {r.encoder_model}")
    print(f"  Ball:  scale={r.ball_scale}  min_norm={r.min_norm}  max_depth={r.max_depth}")
    print(sep)
    print(f"  {'Condition':<36} {'Recall':>8} {'Precision':>10} {'Results':>8} {'Latency':>10}")
    print(f"  {'─'*36} {'─'*8} {'─'*10} {'─'*8} {'─'*10}")

    def row(label, cr):
        print(
            f"  {label:<36} {cr.recall:>8.3f} {cr.precision:>10.3f}"
            f" {cr.avg_result_size:>8.1f} {cr.avg_latency_ms:>9.1f}ms"
        )

    row("C0  FAISS top-K → expand",          r.c0_baseline)
    row("C1  Wide → distance-only → expand", r.c1_blind_rerank)
    row("C2  Wide → classical depth-match",  r.c2_classical)
    row("C3  Wide → hyperbolic rerank",      r.c3_hyperbolic)

    print(sep)
    print(f"\n  Deltas:")
    print(f"    C1 − C0  =  {r.c1_blind_rerank.recall - r.c0_baseline.recall:+.3f}  (should be ≈ 0)")
    print(f"    C2 − C0  =  {r.c2_classical.recall   - r.c0_baseline.recall:+.3f}  (depth feature)")
    print(f"    C3 − C2  =  {r.c3_hyperbolic.recall  - r.c2_classical.recall:+.3f}  (geometry over linear)")
    print(f"    C3 − C0  =  {r.c3_hyperbolic.recall  - r.c0_baseline.recall:+.3f}  (full pipeline lift)")
    print(f"\n  VERDICT:\n{r.verdict()}")
    print(sep)

    print(f"\n  Per-query detail:")
    print(f"  {'Query':<45} {'Spec':>5} {'C0':>5} {'C1':>5} {'C2':>5} {'C3':>5} {'GT depths':>10}")
    print(f"  {'─'*45} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*10}")
    for row_data in r.per_query:
        q_short = row_data["query"][:43] + ".." if len(row_data["query"]) > 43 else row_data["query"]
        gt = ",".join(str(d) for d in row_data["ground_truth_depths_in_corpus"])
        print(
            f"  {q_short:<45}"
            f" {row_data['target_specificity']:>5.2f}"
            f" {row_data['c0']['recall']:>5.2f}"
            f" {row_data['c1']['recall']:>5.2f}"
            f" {row_data['c2']['recall']:>5.2f}"
            f" {row_data['c3']['recall']:>5.2f}"
            f" {gt:>10}"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Specificity-aware re-ranking ablation for Path C."
    )
    p.add_argument("--cpython-lib", type=Path, required=True,
                   help="Path to cpython/Lib directory")
    p.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--k", type=int, default=5,
                   help="Number of seeds before expansion (default: 5)")
    p.add_argument("--n-cand", type=int, default=20,
                   help="Width of the candidate pool before rerank (default: 20)")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Weight on depth-match in classical rerank (default: 0.3)")
    p.add_argument("--ball-scale", type=float, default=0.9)
    p.add_argument("--min-norm", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--out", type=Path,
                   default=Path("benchmarks/results/specificity_ablation.json"))
    args = p.parse_args()

    if not args.cpython_lib.exists():
        raise SystemExit(f"Path not found: {args.cpython_lib}")

    print(f"Loading corpus from {args.cpython_lib}...")
    chunks = load_corpus(args.cpython_lib)
    print(f"  → {len(chunks):,} chunks")

    report = run_ablation(
        chunks,
        encoder_model=args.encoder, k=args.k, n_cand=args.n_cand,
        alpha=args.alpha,
        ball_scale=args.ball_scale, min_norm=args.min_norm,
        max_depth=args.max_depth,
    )
    print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"Results saved to {args.out}\n")


if __name__ == "__main__":
    main()
