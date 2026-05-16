"""
hyprag.chunkers.text
~~~~~~~~~~~~~~~~~~~~
Plain-text chunker.

Splits a text document into a two-level hierarchy:

    depth 0  document root  (always emitted)
    depth 1  paragraph      (separated by blank lines)

If paragraphs are too long, they are split on sentence boundaries to
keep individual chunks under ``max_chunk_chars``. If a single paragraph
is short enough as-is, it is kept intact.

This is the fallback chunker used by the dispatcher when the input is
not recognised as HTML, Markdown, PDF, or source code.
"""

from __future__ import annotations

import re
from pathlib import Path

from hyprag.chunker import Chunk

__all__ = ["TextChunker"]


_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class TextChunker:
    """
    Chunk plain text into a flat sequence of paragraph chunks.

    Parameters
    ----------
    root_slug : str
        Slug for the depth-0 root chunk. Default ``"doc"``.
    min_chunk_chars : int
        Paragraphs shorter than this are merged with the next paragraph.
        Default 40.
    max_chunk_chars : int
        Paragraphs longer than this are split on sentence boundaries.
        Default 1200.
    """

    def __init__(
        self,
        *,
        root_slug: str = "doc",
        min_chunk_chars: int = 40,
        max_chunk_chars: int = 1200,
    ) -> None:
        self.root_slug = root_slug
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    def chunk_text(
        self, text: str, *, doc_title: str | None = None, source: str = "<text>"
    ) -> list[Chunk]:
        title = doc_title or "Document"
        chunks: list[Chunk] = []
        idx = 0
        chunks.append(Chunk(
            id=idx, depth=0, node_path=self.root_slug,
            text=f"{self.root_slug}\n{title}",
            source_file=source, start_line=1, end_line=1,
        ))
        idx += 1

        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
        if not paragraphs:
            return chunks

        # Merge short paragraphs forward
        merged: list[str] = []
        buf = ""
        for p in paragraphs:
            if buf:
                buf = f"{buf}\n\n{p}"
            else:
                buf = p
            if len(buf) >= self.min_chunk_chars:
                merged.append(buf)
                buf = ""
        if buf:
            if merged:
                merged[-1] = f"{merged[-1]}\n\n{buf}"
            else:
                merged.append(buf)

        # Split long paragraphs on sentence boundaries
        split_paragraphs: list[str] = []
        for p in merged:
            if len(p) <= self.max_chunk_chars:
                split_paragraphs.append(p)
                continue
            sentences = _SENTENCE_END_RE.split(p)
            buf = ""
            for s in sentences:
                if buf and len(buf) + len(s) + 1 > self.max_chunk_chars:
                    split_paragraphs.append(buf)
                    buf = s
                else:
                    buf = f"{buf} {s}".strip() if buf else s
            if buf:
                split_paragraphs.append(buf)

        for i, p in enumerate(split_paragraphs, 1):
            path = f"{self.root_slug}.p{i}"
            chunks.append(Chunk(
                id=idx, depth=1, node_path=path,
                text=f"{path}\n{title} (paragraph {i})\n\n{p}",
                source_file=source, start_line=1, end_line=1,
            ))
            idx += 1

        return chunks

    def chunk_file(self, path: str | Path) -> list[Chunk]:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.chunk_text(text, doc_title=path.stem, source=str(path))
