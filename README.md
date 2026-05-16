# HypRAG

**Hierarchical retrieval for structured documents.** FAISS cosine k-NN seeds a result set, then `subtree_expand` walks the chunk parent/child graph to pull every parent, sibling, and child of each hit. The flat encoder finds the right region of the document; the hierarchy walker fills in the surrounding context.

```
+154% Recall@5 on GDPR  ·  +120% on CPython stdlib  ·  <1 ms/query CPU  ·  no GPU
```

## What it does

Most RAG pipelines treat documents as flat bags of chunks. When the right answer lives in paragraph 15(1)(c) of a regulation, flat retrieval returns the chunk for 15(1)(c) — but loses the article header, the surrounding paragraphs, and the chapter context that make the answer interpretable.

HypRAG keeps that structure. Each chunk carries a `node_path` (e.g. `gdpr.ch3.art15.p1.pa`) and a depth tag. After the FAISS lookup, `subtree_expand` returns the parent, the siblings, and the children of every hit. Same recall as flat FAISS at the seed step, but a much higher hit rate after expansion — the answer arrives with its scaffolding intact.

## Benchmarks

### GDPR (EU 2016/679) — 821 chunks, 20 hand-labeled queries, BGE-base, K=5

| Condition                       | Recall@5 |   Precision | Latency |
| ------------------------------- | -------: | ----------: | ------: |
| FAISS (flat)                    |    0.286 |       0.590 |  0.1 ms |
| **FAISS + subtree_expand**      |**0.727** |       0.441 |  0.6 ms |
| Hybrid (BM25+FAISS) + expand    |    0.683 |       0.408 |  1.8 ms |

Expansion lift: **+154.2 %**. BM25 hybrid hurts on regulatory text (uniform vocabulary).

### Chunker generalisation — same GDPR corpus, different chunkers

| Chunker                              | Chunks |  FAISS | FAISS+expand |   Lift |
| ------------------------------------ | -----: | -----: | -----------: | -----: |
| `GDPRChunker` (domain-specific)      |    821 |  0.221 |        0.549 | +148 % |
| `HTMLChunker` (generic, no domain)   |    896 |  0.256 |    **0.564** | +120 % |

The expansion lift is algorithm-driven, not chunker-biased. A source-agnostic chunker that only uses HTML heading levels and `<ol>/<ul>` nesting reaches essentially the same post-expansion recall as a hand-crafted GDPR parser.

### CPython stdlib — 16k chunks, K=5

| Condition                       | Recall@5 |
| ------------------------------- | -------: |
| FAISS (flat)                    |    0.092 |
| **FAISS + subtree_expand**      |**0.203** |

Expansion lift: **+120 %**.

Reproducing the GDPR numbers:

```bash
python -m benchmarks.run_legal_comparison --html-path gdpr_corpus.html
python -m benchmarks.compare_chunkers      --html-path gdpr_corpus.html
```

## Install

```bash
pip install hyprag                       # core (faiss, sentence-transformers, numpy)
pip install hyprag[legal]                # adds beautifulsoup4 for HTML chunkers
pip install hyprag[api]                  # adds fastapi + uvicorn for the HTTP server
pip install hyprag[dev]                  # pytest, ruff, mypy
```

## Quick start — Python codebase

```python
from hyprag.retriever import HypragRetriever

r = HypragRetriever()              # default encoder: BAAI/bge-base-en-v1.5
r.index_path("./myproject")        # AST-based chunker, module → class → method

for chunk in r.query("how does the parser handle escape sequences?", k=5):
    print(chunk.depth, chunk.node_path, chunk.start_line)
```

## Quick start — GDPR (or any hierarchical HTML)

```python
from hyprag.chunkers import GDPRChunker     # domain-specific, +154% lift
from hyprag.chunkers import HTMLChunker     # generic, +120% lift, zero domain knowledge
from hyprag.retriever import HypragRetriever

# Fetch the corpus once (per-article from gdpr-info.eu; takes ~5 min)
chunks = GDPRChunker().load()              # or .load(html_path=Path("..."))

r = HypragRetriever()
r.index_chunks(chunks)

for chunk in r.query("when must a data breach be reported?", k=5):
    print(chunk.depth, chunk.node_path)
    print(chunk.text[:200])
```

`HTMLChunker` works on any HTML document — Wikipedia, documentation, statutes — using only `<h1>`–`<h6>` levels and `<ol>/<ul>/<li>` nesting as hierarchy signals.

## HTTP API

```bash
uvicorn api.main:app --reload
```

`POST /index/gdpr`, `POST /index/codebase`, `POST /index/texts` build indexes. `POST /search` queries them. Each request is authenticated via `X-API-Key`; tiering (free / paid) caps vectors, queries/day, and TTL — see `api/auth.py`.

Every `IndexResponse` returns `depth_distribution` and `warnings`, so callers can verify the chunker recovered the hierarchy as expected without inspecting internals.

## Subtree expansion

`subtree_expand(results, corpus)` is the core algorithm. Given any list of seed chunks and the full corpus, it returns the seeds plus every chunk that is:

- a **parent** — `chunk.node_path` matches a seed's `parent_path`
- a **child** — `chunk.parent_path` matches a seed's `node_path`
- a **sibling** — same `parent_path` as a seed

All three are toggleable; `max_expand` caps the result size. The walk is O(N) per query — cheap enough to run on every search.

## What's deliberately not here

- **No geometry.** Earlier versions used a Poincaré-ball backend for hyperbolic embeddings. Four experiments across two corpora produced numerically identical results to FAISS at up to 257× the latency. Removed in v0.5.0; the git history preserves the code.
- **No LLM summaries.** Tested; recall regressed. Not coming back.
- **No cross-encoder reranking by default.** `bge-reranker-base` hurt on code (Recall@5 0.349 → 0.080). Plug your own in if you have a domain-tuned one.
- **No BM25 by default.** Hurts on legal text (uniform vocabulary). Opt-in per-request via `HybridRetriever` for code corpora where identifiers carry signal.

## Status

v0.5.x. The algorithm is stable. The API is stable. The chunkers are tested against real corpora. What's missing is a hosted demo and packaging polish.

## License

MIT.
