"""
hyprag.sources
~~~~~~~~~~~~~~
Source dispatcher: turn any input — URL, file path, directory, raw
string — into a list of ``Chunk`` objects by picking the right chunker.

The retriever's ``index()`` method delegates to ``chunks_from_source``
so that:

    r.index("https://en.wikipedia.org/wiki/GDPR")   # URL → HTML
    r.index("./notes.md")                           # markdown
    r.index("./contract.pdf")                       # PDF
    r.index("./codebase/")                          # Python directory
    r.index("plain text content here")              # plain text
    r.index(["doc 1", "doc 2"])                     # list of strings

The dispatcher is intentionally heuristic; explicit chunkers
(``HTMLChunker``, ``MarkdownChunker``, …) remain available when the
caller wants direct control.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from hyprag.chunker import Chunk

__all__ = [
    "chunks_from_source",
    "fetch_url",
    "sniff_kind",
]

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_HTML_SNIFF_RE = re.compile(
    r"<(?:html|body|head|h[1-6]|div|article|section|main)\b",
    re.IGNORECASE,
)
_MARKDOWN_SNIFF_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def sniff_kind(s: str) -> str:
    """
    Best-effort detection of what *s* is.

    Returns one of: ``"url"``, ``"path"``, ``"html"``, ``"markdown"``,
    ``"text"``. Does NOT touch the filesystem unless the string is short
    enough to plausibly be a path (no newlines, < 260 chars).
    """
    if _URL_RE.match(s):
        return "url"

    is_path_like = (
        "\n" not in s
        and len(s) < 260
        and not s.lstrip().startswith(("<", "#"))
    )
    if is_path_like:
        try:
            if Path(s).expanduser().exists():
                return "path"
        except (OSError, ValueError):
            pass

    head = s[:1024]
    if _HTML_SNIFF_RE.search(head):
        return "html"
    if _MARKDOWN_SNIFF_RE.search(head):
        return "markdown"
    return "text"


# ---------------------------------------------------------------------------
# URL fetching
# ---------------------------------------------------------------------------

def fetch_url(url: str, *, timeout: float = 20.0) -> tuple[str, str]:
    """
    Fetch *url* and return (body_text, content_type).

    Uses ``urllib.request`` (stdlib, no extra deps). Sends a browser
    User-Agent so sites that block default Python clients work
    (Wikipedia, gov sites, etc.). Follows http→http redirects but
    rejects redirects that change scheme or hostname to a private IP
    target — basic SSRF guard for the dev-tool use case.
    """
    import urllib.error
    import urllib.request

    if not _URL_RE.match(url):
        raise ValueError(f"not an http(s) URL: {url!r}")

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; hyprag/0.7) "
                "Python-urllib"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/pdf,"
                "text/markdown,text/plain;q=0.9,*/*;q=0.8"
            ),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = (resp.headers.get("Content-Type") or "").lower()
        raw = resp.read()

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        return raw, content_type or "application/pdf"  # type: ignore[return-value]

    # Decode text-like content
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
    try:
        body = raw.decode(charset, errors="replace")
    except LookupError:
        body = raw.decode("utf-8", errors="replace")
    return body, content_type


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def chunks_from_source(
    source: str | Path | Iterable[str],
    *,
    chunker_kwargs: dict | None = None,
) -> list[Chunk]:
    """
    Turn *source* into a list of ``Chunk`` objects.

    Dispatch rules (first match wins):
    1. ``Path`` or path-like string pointing at an existing file/dir →
       file/directory dispatch by extension.
    2. ``str`` starting with ``http://``/``https://`` → fetch, then
       dispatch by Content-Type.
    3. ``list`` / ``tuple`` of strings → treat each as a plain-text
       document with sequential ids.
    4. ``str`` content sniffed as HTML / markdown / plain text → run
       the corresponding chunker.
    """
    kw = chunker_kwargs or {}

    # 3. List of strings
    if isinstance(source, (list, tuple)):
        return _chunks_from_text_list(list(source), **kw)

    # 1 & 4. Single string-or-Path
    if isinstance(source, Path):
        return _chunks_from_path(source, **kw)

    if not isinstance(source, str):
        raise TypeError(
            f"unsupported source type {type(source).__name__!r} — "
            "pass a path, URL, string, or list of strings"
        )

    kind = sniff_kind(source)
    if kind == "url":
        return _chunks_from_url(source, **kw)
    if kind == "path":
        return _chunks_from_path(Path(source).expanduser(), **kw)
    if kind == "html":
        from hyprag.chunkers.html_generic import HTMLChunker
        return HTMLChunker(**kw).chunk_html(source)
    if kind == "markdown":
        from hyprag.chunkers.markdown import MarkdownChunker
        return MarkdownChunker(**kw).chunk_markdown(source)
    from hyprag.chunkers.text import TextChunker
    return TextChunker(**kw).chunk_text(source)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunks_from_path(path: Path, **kw) -> list[Chunk]:
    if path.is_dir():
        return _chunks_from_directory(path, **kw)

    suffix = path.suffix.lower()
    if suffix in (".html", ".htm", ".xhtml"):
        from hyprag.chunkers.html_generic import HTMLChunker
        html = path.read_text(encoding="utf-8", errors="replace")
        return HTMLChunker(**kw).chunk_html(html, doc_title=path.stem)
    if suffix in (".md", ".markdown"):
        from hyprag.chunkers.markdown import MarkdownChunker
        text = path.read_text(encoding="utf-8", errors="replace")
        return MarkdownChunker(**kw).chunk_markdown(text, doc_title=path.stem)
    if suffix == ".pdf":
        from hyprag.chunkers.pdf import PDFChunker
        return PDFChunker(**kw).chunk_pdf(path)
    if suffix == ".py":
        from hyprag.chunker import HierarchicalChunker
        return HierarchicalChunker(**kw).chunk_file(path)
    # .txt, .rst, .log, anything else — treat as plain text
    from hyprag.chunkers.text import TextChunker
    return TextChunker(**kw).chunk_file(path)


def _chunks_from_directory(root: Path, **kw) -> list[Chunk]:
    """
    Walk a directory and dispatch each supported file. Chunk ids are
    re-numbered sequentially across all files so the combined corpus is
    consistent.
    """
    supported = (".html", ".htm", ".xhtml", ".md", ".markdown",
                 ".pdf", ".py", ".txt", ".rst")
    all_chunks: list[Chunk] = []
    next_id = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported:
            continue
        try:
            file_chunks = _chunks_from_path(path, **kw)
        except (OSError, ImportError):
            continue
        for c in file_chunks:
            c.id = next_id
            next_id += 1
        all_chunks.extend(file_chunks)
    return all_chunks


def _chunks_from_url(url: str, **kw) -> list[Chunk]:
    body, content_type = fetch_url(url)
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        from hyprag.chunkers.pdf import PDFChunker
        return PDFChunker(**kw).chunk_pdf_bytes(body, source=url)
    if "markdown" in content_type or url.lower().endswith((".md", ".markdown")):
        from hyprag.chunkers.markdown import MarkdownChunker
        return MarkdownChunker(**kw).chunk_markdown(body, doc_title=url)
    if "text/plain" in content_type and not _HTML_SNIFF_RE.search(body[:1024]):
        from hyprag.chunkers.text import TextChunker
        return TextChunker(**kw).chunk_text(body, doc_title=url, source=url)
    # Default: HTML (text/html, application/xhtml+xml, anything else)
    from hyprag.chunkers.html_generic import HTMLChunker
    return HTMLChunker(**kw).chunk_html(body, doc_title=url)


def _chunks_from_text_list(texts: list[str], **_kw) -> list[Chunk]:
    """
    Treat each string as a flat depth-0 chunk with no parent. Mirrors
    ``HypragRetriever.index_texts`` so list-of-strings input behaves the
    same regardless of which entry point you use.
    """
    chunks: list[Chunk] = []
    for i, t in enumerate(texts):
        chunks.append(Chunk(
            id=i, text=t, depth=0,
            node_path=f"text{i}",
            source_file="", start_line=0, end_line=0,
        ))
    return chunks
