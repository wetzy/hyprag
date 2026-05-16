"""
hyprag.chunkers.pdf
~~~~~~~~~~~~~~~~~~~
Best-effort PDF chunker.

PDFs don't carry structure — you get a stream of text positioned on
pages. This chunker extracts text with ``pypdf`` and applies a small
set of heuristics to recover headings:

1. **Numbered prefixes** — lines starting with ``1.``, ``2.1.``,
   ``Chapter 3``, ``Article 17``, ``Section IV`` are treated as
   headings. The depth comes from the dot-count
   (``2.1.3`` → depth 3) or the prefix word (``Chapter`` → depth 1,
   ``Article``/``Section`` → depth 2).
2. **Short all-caps lines** — short standalone lines in ALL CAPS that
   sit between paragraphs are treated as section breaks at depth 2.
3. **Otherwise** — pages become depth-1 chunks. Better than nothing
   for unstructured PDFs.

This won't give you perfect hierarchy on every PDF — for that you'd
need font-size information, which requires a heavier extractor like
``pdfplumber``. But it's enough to make retrieval work on the common
case (regulations, contracts, legal/technical docs with numbered
sections).

Scanned PDFs without OCR text layer return one big chunk and a
warning — there is nothing to chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from hyprag.chunker import Chunk

__all__ = ["PDFChunker"]


_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Numbered heading prefixes: "1.", "1.2.", "1.2.3" — with optional
# trailing space and title text.
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$")

# Word-prefixed headings: "Chapter 3 - Title", "Article 17", "Section IV"
_WORD_HEADING_RE = re.compile(
    r"^(chapter|article|section|part|title)\s+([IVXLCDM\d]+)\b[\s\-:.]*(.*)$",
    re.IGNORECASE,
)
_WORD_HEADING_DEPTH = {
    "title": 1,
    "part": 1,
    "chapter": 1,
    "section": 2,
    "article": 2,
}

# Short all-caps line as section break heuristic.
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z0-9 ,\-:'/]{2,79}$")


def _slug(text: str, max_len: int = 32) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return (s[:max_len] or "section").rstrip("-")


@dataclass
class _Section:
    level: int
    title: str
    body: list[str] = field(default_factory=list)
    children: list["_Section"] = field(default_factory=list)


class PDFChunker:
    """
    Chunk a PDF into a hierarchy using heuristic heading detection.

    Parameters
    ----------
    root_slug : str
        Slug for the depth-0 root chunk. Default ``"doc"``.
    min_chunk_chars : int
        Sections shorter than this are still emitted, but very short
        leaf-only headings are dropped. Default 40.
    page_fallback : bool
        When no headings are detected, fall back to one chunk per page
        at depth 1. Default *True*. Set to *False* to get a single
        depth-0 chunk containing the whole document.
    """

    def __init__(
        self,
        *,
        root_slug: str = "doc",
        min_chunk_chars: int = 40,
        page_fallback: bool = True,
    ) -> None:
        self.root_slug = root_slug
        self.min_chunk_chars = min_chunk_chars
        self.page_fallback = page_fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_pdf(
        self, path: str | Path, *, doc_title: str | None = None
    ) -> list[Chunk]:
        path = Path(path)
        pages = self._extract_pages(path)
        return self._chunks_from_pages(
            pages, source=str(path), doc_title=doc_title
        )

    def chunk_pdf_bytes(
        self, data: bytes, *, source: str = "<pdf>", doc_title: str | None = None
    ) -> list[Chunk]:
        import io
        pages = self._extract_pages_from_stream(io.BytesIO(data))
        return self._chunks_from_pages(
            pages, source=source, doc_title=doc_title
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_pages(self, path: Path) -> list[str]:
        try:
            import pypdf
        except ImportError as exc:
            raise ImportError(
                "pip install pypdf  (needed by PDFChunker)"
            ) from exc

        with open(path, "rb") as fh:
            reader = pypdf.PdfReader(fh)
            return [page.extract_text() or "" for page in reader.pages]

    def _extract_pages_from_stream(self, stream) -> list[str]:
        try:
            import pypdf
        except ImportError as exc:
            raise ImportError(
                "pip install pypdf  (needed by PDFChunker)"
            ) from exc

        reader = pypdf.PdfReader(stream)
        return [page.extract_text() or "" for page in reader.pages]

    # ------------------------------------------------------------------
    # Heading detection + tree building
    # ------------------------------------------------------------------

    def _classify_line(self, line: str) -> tuple[int, str] | None:
        """
        Return (depth, title) if *line* looks like a heading, else None.
        """
        stripped = line.strip()
        if not stripped:
            return None

        m_word = _WORD_HEADING_RE.match(stripped)
        if m_word:
            kind = m_word.group(1).lower()
            number = m_word.group(2)
            tail = m_word.group(3).strip()
            title = f"{kind.title()} {number}"
            if tail:
                title = f"{title} - {tail}"
            return _WORD_HEADING_DEPTH[kind], title

        m_num = _NUMBERED_RE.match(stripped)
        if m_num:
            number = m_num.group(1)
            tail = m_num.group(2).strip()
            # Filter: numbered list items vs section numbers. Section
            # numbers have tails that look like titles (capitalised, no
            # trailing punctuation). List items usually have lowercase
            # continuations of an enclosing paragraph.
            if not tail or tail[0].islower() or len(tail) > 120:
                return None
            depth = number.count(".") + 1
            depth = max(1, min(depth, 5))
            return depth, f"{number} {tail}"

        # All-caps short standalone line → section break.
        if (
            _ALLCAPS_RE.match(stripped)
            and len(stripped) <= 80
            and len(stripped.split()) <= 12
        ):
            return 2, stripped.title()

        return None

    def _build_tree(self, pages: list[str]) -> tuple[list[_Section], bool]:
        """
        Walk all pages line by line, emit a section tree.

        Returns (roots, any_heading_found).
        """
        roots: list[_Section] = []
        stack: list[_Section] = []
        any_heading = False

        for page_text in pages:
            for raw in page_text.splitlines():
                cls = self._classify_line(raw)
                if cls is not None:
                    any_heading = True
                    level, title = cls
                    while stack and stack[-1].level >= level:
                        stack.pop()
                    section = _Section(level=level, title=title)
                    if stack:
                        stack[-1].children.append(section)
                    else:
                        roots.append(section)
                    stack.append(section)
                    continue

                if stack and raw.strip():
                    stack[-1].body.append(raw.strip())

        return roots, any_heading

    # ------------------------------------------------------------------
    # Chunk emission
    # ------------------------------------------------------------------

    def _chunks_from_pages(
        self,
        pages: list[str],
        source: str,
        doc_title: str | None,
    ) -> list[Chunk]:
        title = doc_title or Path(source).stem or "Document"
        chunks: list[Chunk] = []
        idx = 0

        chunks.append(Chunk(
            id=idx, depth=0, node_path=self.root_slug,
            text=f"{self.root_slug}\n{title}",
            source_file=source, start_line=1, end_line=len(pages) or 1,
        ))
        idx += 1

        # Empty / scanned PDF — single chunk, return immediately.
        if not any(p.strip() for p in pages):
            chunks[0] = Chunk(
                id=0, depth=0, node_path=self.root_slug,
                text=(
                    f"{self.root_slug}\n{title}\n\n"
                    "[no extractable text — PDF may be scanned without OCR]"
                ),
                source_file=source, start_line=1, end_line=len(pages) or 1,
            )
            return chunks

        sections, any_heading = self._build_tree(pages)

        if any_heading and sections:
            idx = self._emit_sections(
                sections, parent_path=self.root_slug,
                parent_depth=0, chunks=chunks, next_id=idx, source=source,
            )
        elif self.page_fallback:
            for page_num, page_text in enumerate(pages, 1):
                body = page_text.strip()
                if not body:
                    continue
                path = f"{self.root_slug}.page{page_num}"
                chunks.append(Chunk(
                    id=idx, depth=1, node_path=path,
                    text=f"{path}\nPage {page_num}\n\n{body}",
                    source_file=source,
                    start_line=page_num, end_line=page_num,
                ))
                idx += 1
        else:
            full = "\n\n".join(p.strip() for p in pages if p.strip())
            chunks[0] = Chunk(
                id=0, depth=0, node_path=self.root_slug,
                text=f"{self.root_slug}\n{title}\n\n{full}",
                source_file=source, start_line=1, end_line=len(pages),
            )

        return chunks

    def _emit_sections(
        self,
        sections: list[_Section],
        parent_path: str,
        parent_depth: int,
        chunks: list[Chunk],
        next_id: int,
        source: str,
    ) -> int:
        for sec in sections:
            depth = parent_depth + 1
            slug = _slug(sec.title)
            path = f"{parent_path}.{slug}"

            body = " ".join(sec.body).strip()
            text_parts = [path, sec.title]
            if body:
                text_parts.append("")
                text_parts.append(body)

            # Drop empty leaf sections with short titles (likely OCR noise)
            if not body and not sec.children and len(sec.title) < self.min_chunk_chars:
                continue

            chunks.append(Chunk(
                id=next_id, depth=depth, node_path=path,
                text="\n".join(text_parts),
                source_file=source, start_line=1, end_line=1,
            ))
            next_id += 1

            next_id = self._emit_sections(
                sec.children, parent_path=path,
                parent_depth=depth, chunks=chunks, next_id=next_id, source=source,
            )

        return next_id
