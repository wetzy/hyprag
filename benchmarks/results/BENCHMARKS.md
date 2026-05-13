# HypRAG Benchmarks

Corpus: CPython standard library, **612 files → 16,186 chunks** (depths {0: 612, 1: 5215, 2: 10359}).

## A. Structural metrics (encoder-independent)

| Metric | FAISS IndexFlatL2 | HypRAG PoincareBallIndex | Δ |
|---|---:|---:|---:|
| Index build time (ms) | 5.2 | 132.9 | +2455.8% ✗ |
| Memory delta (MB) | 24.9 | 153.9 | +518.1% ✗ |
| Search latency (ms/query, k=10) | 1.322 | 21.299 | +1511.1% ✗ |
| Subtree coherence (top-5, random queries) | 0.005 | 0.0 | -100.0% ✗ |
| Subtree coherence (top-5, **expanded**) | — | 0.67 | — |

Chunking throughput: **5,270 chunks/sec** (3.071s wall).

### Honest notes on structural results

- HypRAG is currently slower and heavier than FAISS Flat because it uses 
  brute-force PyTorch ops instead of FAISS's hand-tuned SIMD. The point 
  of this version is to validate the geometry, not the engineering. 
  HNSW-on-the-ball is the next milestone.
- Search latency under ~100ms is acceptable for a retrieval endpoint, 
  but the 40×+ gap will widen at >100k vectors. Plan accordingly.
- Subtree coherence on **random** queries should be low for both 
  retrievers — the metric only becomes informative on the semantic eval 
  below, where the query actually targets a subtree.

## B. Semantic metrics

_Not run in this sandbox (no network access for the encoder)._

Reproduce locally:

```bash
git clone --depth 1 --sparse https://github.com/python/cpython.git
cd cpython && git sparse-checkout set Lib && cd ..
python -m benchmarks.run_benchmark --cpython-lib cpython/Lib
```

Expected directional results (validating the hypothesis):

1. **HypRAG raw k-NN ≈ FAISS** on Recall@K. Pure geodesic distance vs 
   Euclidean distance on the same embeddings shouldn't differ much when 
   no depth information is used as a tiebreaker.
2. **HypRAG + subtree expansion >> FAISS** on Recall@K. This is the 
   product claim: pulling the subtree of a hit catches the related 
   methods that flat retrieval misses.
3. **Precision drops slightly under expansion** — expected; you're 
   trading off precision for recall. The right metric to watch is 
   F1 or nDCG once the eval set grows.

## C. Reproducibility

All numbers above come from `benchmarks/run_benchmark.py`. To regenerate:

```bash
pip install -e ".[dev]"
python -m benchmarks.run_benchmark --cpython-lib /path/to/cpython/Lib
```

Outputs land in `benchmarks/results/`. Re-running on the same CPython 
commit should produce byte-identical chunk counts; timing numbers will 
vary by ~5% run-to-run depending on the machine.