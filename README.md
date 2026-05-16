# HypRAG

**Hierarchical retrieval for structured documents.** FAISS cosine k-NN seeds a result set, then `subtree_expand` walks the chunk parent/child graph to pull every parent, sibling, and child of each hit. The flat encoder finds the right region of the document; the hierarchy walker fills in the surrounding context.

```
+154% Recall@5 on GDPR  ·  +120% on CPython stdlib  ·  <1 ms/query CPU  ·  no GPU
```

## What it does

Most RAG pipelines treat documents as flat bags of chunks. When the right answer lives in paragraph 15(1)(c) of a regulation, flat retrieval returns the chunk for 15(1)(c) — but loses the article header, the surrounding paragraphs, and the chapter context that make the answer interpretable.

HypRAG keeps that structure. Each chunk carries a `node_path` (e.g. `gdpr.ch3.art15.p1.pa`) and a depth tag. After the FAISS lookup, `subtree_expand` returns the parent, the siblings, and the children of every hit. Same recall as flat FAISS at the seed step, but a much higher hit rate after expansion — the answer arrives with its scaffolding intact.

## Install

```bash
pip install hyprag                       # core: faiss, sentence-transformers, numpy
pip install "hyprag[pdf]"                # + pypdf for PDF chunking
pip install "hyprag[pdf-plumber]"        # + pdfplumber for clean text on legal/EU PDFs
pip install "hyprag[html]"               # + beautifulsoup4 for HTML/markdown chunkers
pip install "hyprag[all]"                # everything optional
```

## Quick start

One method handles every input format:

```python
from hyprag import HypragRetriever

r = HypragRetriever()
r.index("https://en.wikipedia.org/wiki/General_Data_Protection_Regulation")

results = r.query(
    "What is the maximum fine for a severe violation?",
    k=1,
    return_metadata=True,
    rescore_after_expand=True,
    min_score=0.55,
)
for res in results:
    print(f"[{res.chunk.node_path}] score={res.score:.3f} ({res.relation})")
    print(res.chunk.text[:200], "\n")
```

`r.index()` dispatches on what you pass:

| Input                                       | Routed to                       |
| ------------------------------------------- | ------------------------------- |
| `"https://..."`                             | URL fetch → HTML / PDF chunker  |
| `"./contract.pdf"`                          | `PDFChunker`                    |
| `"./notes.md"`                              | `MarkdownChunker`               |
| `"./doc.html"`                              | `HTMLChunker`                   |
| `"./codebase/"` (directory)                 | extension dispatch per file     |
| `"./script.py"`                             | `HierarchicalChunker` (AST)     |
| `"<html><body>..."`                         | `HTMLChunker`                   |
| `"# Title\n..."`                            | `MarkdownChunker`               |
| Anything else (raw string)                  | `TextChunker`                   |
| `["doc 1", "doc 2"]`                        | flat list of root-level chunks  |

Use `chunker_kwargs` to forward options (e.g. PDF backend choice):

```python
r.index("./boe-gdpr-es.pdf", chunker_kwargs={"backend": "pdfplumber"})
```

## Why structural expansion matters

```python
results = r.query("...", k=5, return_metadata=True, rescore_after_expand=True)
for res in results:
    print(res.score, res.relation, res.chunk.node_path)
```

Each `RetrievalResult` carries:
- `chunk` — the actual `Chunk`.
- `score` — cosine similarity in `[0, 1]`.
- `relation` — `"seed"`, `"parent"`, `"sibling"`, or `"child"`.
- `seed_path` — which seed pulled this chunk in.

The seed is the FAISS top hit. Its parent, siblings, and children come along because they're often where the actual answer lives. `rescore_after_expand=True` re-encodes everything against the query so siblings with high overlap end up at the top; `min_score=` drops low-signal expansions (e.g. the entire References section of a Wikipedia article when one ref happened to match).

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

## Built-in chunkers

| Chunker               | Hierarchy signal                                           |
| --------------------- | ---------------------------------------------------------- |
| `HierarchicalChunker` | Python AST — module → class → method                       |
| `HTMLChunker`         | `<h1>`–`<h6>` levels + `<ol>/<ul>/<li>` nesting            |
| `MarkdownChunker`     | ATX (`#`–`######`) + setext + list nesting                 |
| `PDFChunker`          | Numbered headings (`1.`, `2.1.3`), word headings (English + Spanish: `Artículo`, `Capítulo`, `Sección`, …), ALL-CAPS lines, page fallback |
| `TextChunker`         | Paragraph (blank-line) split with sentence overflow        |
| `GDPRChunker`         | DOM-driven, fetches gdpr-info.eu per article               |

All produce `Chunk` objects with a `node_path` that subtree expansion walks.

## Subtree expansion

`subtree_expand(results, corpus)` is the core algorithm. Given any list of seed chunks and the full corpus, it returns the seeds plus every chunk that is:

- a **parent** — `chunk.node_path` matches a seed's `parent_path`
- a **child** — `chunk.parent_path` matches a seed's `node_path`
- a **sibling** — same `parent_path` as a seed

All three are toggleable; `max_expand` caps the result size. The walk is O(N) per query — cheap enough to run on every search.

## Multilingual

The default encoder (`BAAI/bge-base-en-v1.5`) is English-only. For Spanish, French, German, or any mixed-language corpus, swap in a multilingual model:

```python
r = HypragRetriever(
    encoder_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
```

For Spanish BOE / EUR-Lex PDFs, combine the multilingual encoder with `backend="pdfplumber"` — pypdf mangles tracked typography (`"Artículo"` → `"Ar tículo"`), defeating heading detection.

## HTTP API

```bash
pip install "hyprag[api]"
uvicorn api.main:app --reload
```

`POST /index/url`, `POST /index/codebase`, `POST /index/texts` build indexes. `POST /search` queries them. Each request is authenticated via `X-API-Key`; tiering (free / paid) caps vectors, queries/day, and TTL — see `api/auth.py`.

## What's deliberately not here

- **No geometry.** Earlier versions used a Poincaré-ball backend for hyperbolic embeddings. Four experiments across two corpora produced numerically identical results to FAISS at up to 257× the latency. Removed in v0.5.0; the git history preserves the code.
- **No LLM summaries.** Tested; recall regressed. Not coming back.
- **No cross-encoder reranking by default.** `bge-reranker-base` hurt on code (Recall@5 0.349 → 0.080). Plug your own in if you have a domain-tuned one.
- **No BM25 by default.** Hurts on legal text (uniform vocabulary). Opt-in per-request via `HybridRetriever` for code corpora where identifiers carry signal.

## Status

v0.7.x. The algorithm is stable. The unified `r.index()` API is the recommended entry point. PDF support is best-effort — `pdfplumber` is needed for legal/EU PDFs with tracked typography.

## License

MIT.
