"""
hyprag.chunkers.html_generic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Source-agnostic HTML chunker.

Uses only the universal structural signal in HTML — heading levels
``<h1>`` through ``<h6>`` and ordered/unordered list nesting — to build
a hierarchy. No knowledge of any specific website, schema, or document
format is encoded. The same code chunks Wikipedia, a regulation page,
documentation sites, blog posts, anything.

Hierarchy
---------
    depth 0  document root         (always emitted)
    depth 1  <h1>                  top-level sections
    depth 2  <h2>                  sub-sections
    depth 3  <h3>
    depth 4  <h4>
    depth 5  list items (<li>) under the deepest heading, if any

Why this matters
----------------
The HypRAG benchmark with a hand-crafted GDPR chunker shows +149%
expansion lift. The reasonable suspicion is "of course it works, you
hard-coded the hierarchy." This chunker rules that out by using only
the universal heading-level signal that every web page carries. If
expansion still produces lift here, the algorithm generalises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hyprag.chunker import Chunk

__all__ = ["HTMLChunker"]


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, max_len: int = 32) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return (s[:max_len] or "section").rstrip("-")


@dataclass
class _Section:
    """A heading and everything that belongs to it before the next heading
    of equal-or-shallower level."""
    level: int                       # 1–6 (h1–h6)
    title: str
    body_text: str                   # paragraphs etc. that live directly under it
    list_items: list[str]            # top-level <li> text under this heading
    children: list["_Section"] = field(default_factory=list)


class HTMLChunker:
    """
    Chunk any HTML document into a hierarchy driven only by heading levels.

    Parameters
    ----------
    root_slug : str
        Slug to use for the depth-0 root chunk. Default ``"doc"``.
    min_chunk_chars : int
        Sections with less than this much body text are still emitted, but
        the threshold suppresses depth-5 list-item chunks for short items.
        Default 40.
    include_list_items : bool
        When *True* (default), top-level ``<li>`` elements under the deepest
        heading become depth-N+1 chunks. Set *False* to keep the hierarchy
        purely heading-driven.
    """

    def __init__(
        self,
        *,
        root_slug: str = "doc",
        min_chunk_chars: int = 40,
        include_list_items: bool = True,
    ) -> None:
        self.root_slug = root_slug
        self.min_chunk_chars = min_chunk_chars
        self.include_list_items = include_list_items

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_html(self, html: str, *, doc_title: str | None = None) -> list[Chunk]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "pip install beautifulsoup4  (needed by HTMLChunker)"
            ) from exc

        # ``html.parser`` again — see legal.py for the lxml gotcha
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # Walk the WHOLE soup, not soup.body. ``soup.body`` returns only the
        # first <body> tag; concatenated multi-document HTML (e.g. the GDPR
        # corpus of 99 stacked <html><body>...</body></html> pages) has 99
        # body tags and we need to traverse them all. ``soup.descendants``
        # covers everything.
        sections = self._build_section_tree(soup)

        title = (
            doc_title
            or (soup.title.get_text(strip=True) if soup.title else None)
            or "Document"
        )

        chunks: list[Chunk] = []
        idx = 0
        # depth 0 — root
        chunks.append(Chunk(
            id=idx, depth=0, node_path=self.root_slug,
            text=f"{self.root_slug}\n{title}",
            source_file="<html>", start_line=1, end_line=1,
        )); idx += 1

        idx = self._emit_sections(
            sections, parent_path=self.root_slug,
            parent_depth=0, chunks=chunks, next_id=idx,
        )
        return chunks

    # ------------------------------------------------------------------
    # Section tree construction
    # ------------------------------------------------------------------

    def _build_section_tree(self, body) -> list[_Section]:
        """
        Walk the body in document order. Each heading opens a new section at
        its level; everything until the next heading of equal-or-shallower
        level is that section's content.

        We track a stack of currently-open sections so nesting falls out
        naturally: an h3 attaches to the most recent unclosed h2, an h2 to
        the most recent unclosed h1, etc.
        """
        roots: list[_Section] = []
        stack: list[_Section] = []   # currently-open sections, deepest last

        for el in body.descendants:
            name = getattr(el, "name", None)
            if not name:
                continue

            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(name[1])
                # Pop until we find a parent of strictly shallower level
                while stack and stack[-1].level >= level:
                    stack.pop()
                section = _Section(
                    level=level,
                    title=el.get_text(" ", strip=True),
                    body_text="",
                    list_items=[],
                )
                if stack:
                    stack[-1].children.append(section)
                else:
                    roots.append(section)
                stack.append(section)
                continue

            if not stack:
                # Content before the first heading is discarded
                continue

            current = stack[-1]

            if name == "p":
                txt = el.get_text(" ", strip=True)
                if txt:
                    current.body_text = (
                        f"{current.body_text}\n{txt}" if current.body_text else txt
                    )

            elif name == "li":
                # Only count list items that aren't nested inside another <li>
                # already handled by a previous iteration.
                if el.find_parent("li") is None and self.include_list_items:
                    txt = el.get_text(" ", strip=True)
                    if txt:
                        current.list_items.append(txt)

        return roots

    # ------------------------------------------------------------------
    # Chunk emission
    # ------------------------------------------------------------------

    def _emit_sections(
        self,
        sections: list[_Section],
        parent_path: str,
        parent_depth: int,
        chunks: list[Chunk],
        next_id: int,
    ) -> int:
        for sec in sections:
            depth = parent_depth + 1
            slug = _slug(sec.title)
            path = f"{parent_path}.{slug}"

            text_parts = [path, sec.title]
            if sec.body_text:
                text_parts.append("")
                text_parts.append(sec.body_text)

            chunks.append(Chunk(
                id=next_id, depth=depth, node_path=path,
                text="\n".join(text_parts),
                source_file="<html>", start_line=1, end_line=1,
            ))
            next_id += 1

            # Recurse into nested headings
            next_id = self._emit_sections(
                sec.children, parent_path=path,
                parent_depth=depth, chunks=chunks, next_id=next_id,
            )

            # Emit list-item chunks ONE LEVEL DEEPER, but only if this
            # section has no sub-headings (otherwise list items would compete
            # with sub-sections for the same depth and ordering becomes weird).
            if sec.list_items and not sec.children and self.include_list_items:
                for i, item in enumerate(sec.list_items, 1):
                    if len(item) < self.min_chunk_chars:
                        continue
                    item_path = f"{path}.li{i}"
                    chunks.append(Chunk(
                        id=next_id, depth=depth + 1, node_path=item_path,
                        text=f"{item_path}\n{sec.title} (item {i})\n\n{item}",
                        source_file="<html>", start_line=1, end_line=1,
                    ))
                    next_id += 1

        return next_id
