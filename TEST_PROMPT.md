# HypRAG Test Prompt

I just found this Python library called **hyprag** on GitHub — claims +154% Recall@5 on legal document retrieval using hierarchical chunk expansion. Want to test if it actually works.

## Install

```
pip install hyprag
pip install "hyprag[legal]"
```

> Note: pip may show a harmless warning about `lxml` or `requests` version pins on Python 3.14 — the install succeeds regardless.

## Test 1 — basic retrieval on plain text chunks

```python
from hyprag.retriever import HypragRetriever

r = HypragRetriever()
r.index_texts([
    "The right to erasure allows individuals to request deletion of personal data.",
    "Controllers must respond to erasure requests within one month.",
    "Erasure does not apply when data is needed for legal claims.",
    "Data subjects have the right to access their personal data held by controllers.",
    "Personal data must be kept accurate and up to date at all times.",
])

results = r.query("when can someone request their data be deleted?", k=3)
for chunk in results:
    print(chunk.node_path, chunk.text[:100])
```

## Test 2 — HTML chunker on a simple structured document

```python
from hyprag.chunkers.html_generic import HTMLChunker

html = """
<html><body>
  <h1>Chapter 1: Data Rights</h1>
  <p>Individuals have several rights regarding their personal data.</p>
  <h2>Article 1: Right of Access</h2>
  <ol>
    <li>Any person may request a copy of their data held by a controller.</li>
    <li>The controller must respond within 30 days of receiving the request.</li>
  </ol>
  <h2>Article 2: Right to Erasure</h2>
  <ol>
    <li>Individuals may request deletion of their data under certain conditions.</li>
    <li>Erasure may be refused if the data is required for legal proceedings.</li>
  </ol>
</body></html>
"""

chunks = HTMLChunker().chunk_html(html)
for c in chunks:
    print(f"depth={c.depth} path={c.node_path}")
    print(f"  {c.text[:80]}")
    print()
```

## What to verify

1. Does `index_texts` work without errors?
2. Does the query return relevant results?
3. Does `HTMLChunker` produce a proper hierarchy — depth 0 for the root, depth 1 for Chapter, depth 2 for Articles, depth 3 for the numbered list items?
4. Do `node_path` and `parent_path` relationships make sense (e.g. `doc.chapter-1-data-rights.article-2-right-to-erasure.li1`)?

Report exactly what gets printed for both tests, including any errors.
