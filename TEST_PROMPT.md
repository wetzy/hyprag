# HypRAG Test Prompt

I just found this Python library called **hyprag** on GitHub — claims +154% Recall@5 on legal document retrieval using hierarchical chunk expansion, and one method that indexes anything (URL, PDF, markdown, HTML, plain text). Want to test if it actually works.

## Install

```
pip install "hyprag[all]"
```

> Note: this pulls in `beautifulsoup4`, `pypdf`, `pdfplumber`, `lxml`, and `requests`. On Python 3.14 pip may show a warning about version pins — install succeeds regardless.

## Test 1 — URL fetch, ask a real question

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

Expected: the top result mentions the **€20M / 4% of global turnover** maximum fine from GDPR Article 83(5).

## Test 2 — Plain-text fallback

```python
from hyprag import HypragRetriever

r = HypragRetriever()
r.index([
    "The right to erasure allows individuals to request deletion of personal data.",
    "Controllers must respond to erasure requests within one month.",
    "Erasure does not apply when data is needed for legal claims.",
    "Data subjects have the right to access their personal data held by controllers.",
    "Personal data must be kept accurate and up to date at all times.",
])

for chunk in r.query("when can someone request their data be deleted?", k=3):
    print(chunk.node_path, chunk.text[:80])
```

## Test 3 — Markdown file dispatch

```python
from pathlib import Path
from hyprag import HypragRetriever

Path("notes.md").write_text("""
# Project Notes

## Background

We migrated to Postgres in Q3 because the MongoDB cluster had recurring write conflicts.

## Decisions

- Switch ORMs to SQLAlchemy for type safety
- Drop the read replica until traffic justifies it
- Use connection pooling in production
""")

r = HypragRetriever()
r.index("notes.md")
for chunk in r.query("why did we move off mongodb?", k=2):
    print(chunk.node_path, chunk.text[:100])
```

## What to verify

1. Does `r.index(url)` work without errors and produce >100 chunks for the Wikipedia GDPR page?
2. Does the top GDPR result actually mention the €20M / 4% fine?
3. Does `r.index([...])` (list of strings) return chunks at depth 0 with no parent?
4. Does `r.index("notes.md")` produce a hierarchy (`doc`, `doc.project-notes`, `doc.project-notes.background`, …)?
5. Does subtree expansion pull in siblings — e.g. querying for "mongodb" returns the Background section AND the Decisions section?

Report exactly what gets printed for each test, including any errors or warnings.
