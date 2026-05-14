# HypRAG Benchmarks

Corpus: CPython standard library, **612 files → 16,186 chunks** (depths {0: 612, 1: 5215, 2: 10359}).

## A. Structural metrics (encoder-independent)

| Metric | FAISS IndexFlatL2 | HypRAG PoincareBallIndex | Δ |
|---|---:|---:|---:|
| Index build time (ms) | 6.0 | 114.7 | +1811.7% ✗ |
| Memory delta (MB) | 24.9 | 153.9 | +518.1% ✗ |
| Search latency (ms/query, k=10) | 1.836 | 29.6 | +1512.2% ✗ |
| Subtree coherence (top-5, random queries) | 0.005 | 0.0 | -100.0% ✗ |
| Subtree coherence (top-5, **expanded**) | — | 0.67 | — |

Chunking throughput: **7,044 chunks/sec** (2.298s wall).

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

## B. Semantic metrics (Recall@K, Precision@K)

Encoder: `all-MiniLM-L6-v2`. K = 5. Queries: **20** hand-labeled. Ground truth = hand-curated subtree prefixes (see `benchmarks/queries.py`).

| Metric | FAISS | HypRAG (k-NN) | HypRAG + expand | Hybrid (RRF) | Hybrid + expand | Δ Hybrid+expand vs FAISS |
|---|---:|---:|---:|---:|---:|---:|
| Recall@5 | 0.102 | 0.038 | 0.195 | 0.058 | **0.188** | +84.3% ✓ |
| Precision@5 | 0.28 | 0.17 | 0.175 | 0.15 | 0.178 | -36.4% ✗ |

### Per-query breakdown

<details><summary>Click to expand</summary>

| Query | FAISS R@K | HypRAG+expand R@K | Hybrid+expand R@K |
|---|---:|---:|---:|
| how does asyncio schedule callbacks | 0.00 | 0.00 | 0.00 |
| where is SSL certificate validation | 0.04 | 0.10 | 0.19 |
| what does the csv DictWriter do | 0.83 | 1.00 | 1.00 |
| how is os path join implemented across platforms | 0.50 | 0.50 | 0.50 |
| where are HTTP status codes defined | 0.00 | 0.00 | 0.00 |
| how does threading Lock work internally | 0.12 | 0.31 | 0.46 |
| how does pickle handle custom classes | 0.03 | 0.02 | 0.01 |
| where is JSON parsing logic | 0.00 | 0.00 | 0.00 |
| how does logging configure handlers | 0.00 | 0.00 | 0.00 |
| what does collections OrderedDict do | 0.00 | 0.00 | 0.00 |
| how does urllib parse URLs | 0.00 | 0.00 | 0.00 |
| how does subprocess Popen launch processes | 0.05 | 0.10 | 0.14 |
| how does datetime parse strings | 0.01 | 0.00 | 0.00 |
| how does argparse handle subparsers | 0.00 | 0.00 | 0.00 |
| where is base64 encoding implemented | 0.19 | 0.77 | 0.35 |
| how does sqlite3 connect to a database | 0.00 | 0.00 | 0.00 |
| how does heapq maintain the heap invariant | 0.28 | 1.00 | 1.00 |
| what does the abc module provide | 0.00 | 0.12 | 0.12 |
| how does xml etree parse documents | 0.00 | 0.00 | 0.00 |
| how does multiprocessing share memory between processes | 0.00 | 0.00 | 0.00 |

</details>

## C. Reproducibility

All numbers above come from `benchmarks/run_benchmark.py`. To regenerate:

```bash
pip install -e ".[dev]"
python -m benchmarks.run_benchmark --cpython-lib /path/to/cpython/Lib
```

Outputs land in `benchmarks/results/`. Re-running on the same CPython 
commit should produce byte-identical chunk counts; timing numbers will 
vary by ~5% run-to-run depending on the machine.