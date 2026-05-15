"""
benchmarks.compare_chunkers
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Apples-to-apples comparison of two chunkers on the GDPR corpus:

    1. GDPRChunker  — domain-specific. Knows the URL pattern, the
                       entry-content div, that <ol><li> = paragraph,
                       that nested <li> = lettered point, and the
                       article→chapter map.
    2. HTMLChunker  — domain-agnostic. Uses only <h1>–<h6> heading
                       levels for hierarchy. No GDPR knowledge.

Both feed the same retrieval pipeline (FAISS + subtree_expand) and the
same 20 hand-labeled queries. The question: does the algorithm's
expansion lift survive when the chunker has zero source-specific
knowledge?

Why this exists
---------------
The 0.687 Recall@5 on GDPRChunker (gdpr-info.eu) is suspicious if you
read it as "of course it works, the chunker was tuned to the source."
This benchmark tests the steelman: run the algorithm against a chunker
that cannot have been tuned.

Usage
-----
    python -m benchmarks.compare_chunkers --html-path gdpr_corpus.html

The corpus file must be the same concatenated gdpr-info.eu HTML used by
``run_legal_comparison.py`` so the comparison is on identical input.

Mapping ground truth to HTMLChunker paths
-----------------------------------------
HTMLChunker produces paths like ``doc.art-15-gdpr.right-of-access...``;
the GDPR_QUERIES ground truth uses ``gdpr.ch3.art15`` style. We map a
chunk to "relevant" if its node_path contains ``art-15`` (the heading
slug derived from "Art. 15 GDPR" titles). This keeps the comparison
honest — the mapping uses information available in the heading text,
which the chunker had no special access to.
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
from hyprag.chunkers.html_generic import HTMLChunker
from hyprag.chunkers.legal import GDPRChunker
from hyprag.retriever import subtree_expand

from benchmarks.gdpr_queries import GDPR_QUERIES


@dataclass
class ChunkerResult:
    chunker: str
    corpus_chunks: int
    depth_dist: dict[int, int]
    recall_faiss: float
    recall_faiss_expand: float
    expansion_lift_pct: float
    avg_latency_ms: float


def _is_relevant_for_path_substring(node_path: str, gt_prefixes: list[str]) -> bool:
    """Match if the chunk path contains any ground-truth article slug.

    GDPRChunker emits gdpr.chN.artM.* ; HTMLChunker emits
    doc.<slug>... where the slug derives from the heading text. We
    translate ground-truth like ``gdpr.ch3.art15`` to the substring
    ``art15`` OR ``art-15`` and check both.
    """
    for prefix in gt_prefixes:
        # extract artN
        for token in prefix.split("."):
            if token.startswith("art") and token[3:].isdigit():
                n = token[3:]
                if f"art{n}" in node_path or f"art-{n}" in node_path:
                    return True
    return False


def _run_pipeline(chunks: list[Chunk], encoder_model: str, k: int) -> tuple[float, float, float]:
    """Encode, index, query, return (recall_faiss, recall_expand, avg_latency_ms)."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(encoder_model, trust_remote_code=True, device=device)
    texts = [c.text for c in chunks]
    vecs: np.ndarray = model.encode(
        texts, batch_size=128, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float32)
    faiss.normalize_L2(vecs)

    fi = faiss.IndexFlatIP(vecs.shape[1])
    fi.add(vecs)

    n_cand = max(k * 4, 20)
    recalls_f: list[float] = []
    recalls_fe: list[float] = []
    lats: list[float] = []

    for q in GDPR_QUERIES:
        q_vec = model.encode([q.text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_vec)

        n_rel = max(
            sum(1 for c in chunks if _is_relevant_for_path_substring(c.node_path, q.ground_truth_prefixes)),
            1,
        )

        t0 = time.perf_counter()
        _, fids = fi.search(q_vec, k)
        f_chunks = [chunks[i] for i in fids[0] if i >= 0]
        f_hit = sum(1 for c in f_chunks if _is_relevant_for_path_substring(c.node_path, q.ground_truth_prefixes))
        recalls_f.append(f_hit / n_rel)

        fe_chunks = subtree_expand(f_chunks, chunks, max_expand=n_cand)
        t_fe = (time.perf_counter() - t0) * 1000
        fe_hit = sum(1 for c in fe_chunks if _is_relevant_for_path_substring(c.node_path, q.ground_truth_prefixes))
        recalls_fe.append(fe_hit / n_rel)
        lats.append(t_fe)

    return (
        sum(recalls_f) / len(recalls_f),
        sum(recalls_fe) / len(recalls_fe),
        sum(lats) / len(lats),
    )


def _evaluate(label: str, chunks: list[Chunk], encoder_model: str, k: int) -> ChunkerResult:
    dist: dict[int, int] = {}
    for c in chunks:
        dist[c.depth] = dist.get(c.depth, 0) + 1
    print(f"\n{label}: {len(chunks)} chunks, depths {dict(sorted(dist.items()))}")

    r_f, r_fe, lat = _run_pipeline(chunks, encoder_model, k)
    lift = (r_fe - r_f) / max(r_f, 1e-9) * 100
    print(f"  FAISS         : Recall@{k} = {r_f:.3f}")
    print(f"  FAISS + expand: Recall@{k} = {r_fe:.3f}   (+{lift:.1f}%)")
    print(f"  Avg latency   : {lat:.2f} ms")

    return ChunkerResult(
        chunker=label,
        corpus_chunks=len(chunks),
        depth_dist=dict(sorted(dist.items())),
        recall_faiss=round(r_f, 3),
        recall_faiss_expand=round(r_fe, 3),
        expansion_lift_pct=round(lift, 1),
        avg_latency_ms=round(lat, 2),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--html-path", type=Path, required=True,
                   help="Concatenated GDPR HTML (from gdpr-info.eu fetch)")
    p.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", type=Path,
                   default=Path("benchmarks/results/chunker_comparison.json"))
    args = p.parse_args()

    print("Loading raw HTML...")
    html = args.html_path.read_text(encoding="utf-8", errors="replace")

    # Run both chunkers on the SAME input bytes
    print("\nChunking with GDPRChunker (domain-specific)...")
    gdpr_chunks = GDPRChunker().load(html_string=html)

    print("Chunking with HTMLChunker (heading-level only)...")
    html_chunks = HTMLChunker().chunk_html(html)

    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  Chunker Comparison on GDPR  |  K={args.k}  |  {len(GDPR_QUERIES)} queries")
    print(f"  Encoder: {args.encoder}")
    print(sep)

    gdpr_result = _evaluate("GDPRChunker (domain-specific)", gdpr_chunks, args.encoder, args.k)
    html_result = _evaluate("HTMLChunker (heading-level)",   html_chunks, args.encoder, args.k)

    print(f"\n{sep}")
    print("  VERDICT")
    print(sep)
    if html_result.expansion_lift_pct > 10:
        print(
            f"  Generic HTMLChunker still shows +{html_result.expansion_lift_pct:.0f}% "
            "expansion lift on heading-only hierarchy.\n"
            "  → The algorithm generalises beyond hand-crafted chunkers."
        )
    elif html_result.expansion_lift_pct > 0:
        print(
            f"  Generic HTMLChunker shows only +{html_result.expansion_lift_pct:.0f}% lift.\n"
            "  → Expansion benefit is real but depends on chunker quality."
        )
    else:
        print(
            f"  Generic HTMLChunker shows {html_result.expansion_lift_pct:+.0f}% lift.\n"
            "  → Heading-only hierarchy is too coarse to benefit from expansion on this corpus."
        )

    payload = {
        "encoder": args.encoder,
        "k": args.k,
        "n_queries": len(GDPR_QUERIES),
        "results": [asdict(gdpr_result), asdict(html_result)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved → {args.out}\n")


if __name__ == "__main__":
    main()
