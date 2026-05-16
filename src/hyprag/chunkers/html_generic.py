"""
hyprag.chunkers.html_generic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Source-agnostic HTML chunker.

Uses TWO universal structural signals — heading levels ``<h1>``–``<h6>``
AND ``<ol>``/``<ul>``/``<li>`` nesting — to build a hierarchy. No
knowledge of any specific website, schema, or document format is
encoded. The same code chunks Wikipedia, a regulation page, a
documentation site, or a blog post.

Hierarchy
---------
    depth 0   document root          (always emitted)
    depth 1   <h1>                   top-level sections
    depth 2   <h2>                   sub-sections — and top-level <li>
                                     items under a depth-1 heading
    depth 3   <h3>                   sub-sub-sections — and nested <li>
                                     items (e.g. lettered points)
    depth N   <hN>                   sections, plus list nesting one
                                     level deeper than the deepest
                                     heading above

Headings and list items can coexist at the same depth — they're treated
as parallel children of their parent section. Subtree expansion walks
them all.

Why list nesting matters
------------------------
Many regulatory documents (GDPR, EU regs, statutes) encode their
paragraph and point hierarchy in ``<ol><li>`` nesting, not in heading
levels. A chunker that only looks at headings sees the article title
and misses the paragraph structure entirely — paragraphs and lettered
points never become chunks. Adding list nesting as a second hierarchy
signal recovers that without any source-specific knowledge.

The previous implementation suppressed list items whenever a section
had child headings, on the theory that "ordering becomes weird." It
does — slightly — but the cost is catastrophic for documents like GDPR
where every article has a sibling "Suitable Recitals" heading that
killed all paragraph emission. List items and child headings now
coexist at the same depth, which subtree expansion handles fine.
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
class _ListItem:
    """One ``<li>`` parsed into direct text + nested-list children.

    ``text`` is the text of the <li> with nested ``<ol>``/``<ul>``
    excluded. ``children`` is the recursive parse of those nested lists,
    so the full <li> content is reachable but each level forms its own
    chunk at its own depth.
    """
    text: str
    children: list["_ListItem"] = field(default_factory=list)


@dataclass
class _Section:
    """A heading and everything that belongs to it before the next heading
    of equal-or-shallower level."""
    level: int                          # 1–6 (h1–h6)
    title: str
    body_text: str                      # <p> text directly under this heading
    list_items: list[_ListItem] = field(default_factory=list)
    children: list["_Section"] = field(default_factory=list)


class HTMLChunker:
    """
    Chunk any HTML document into a hierarchy driven by heading levels
    AND list nesting.

    Parameters
    ----------
    root_slug : str
        Slug for the depth-0 root chunk. Default ``"doc"``.
    min_chunk_chars : int
        Body-only sections shorter than this are still emitted, but
        terminal list items (no children) shorter than this are
        suppressed. List items WITH children survive regardless — their
        children carry the content. Default 40.
    include_list_items : bool
        When *True* (default), ``<ol>``/``<ul>`` nesting under headings
        contributes additional depth levels. Set *False* to keep the
        hierarchy purely heading-driven.
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

        # ``html.parser`` — see legal.py for the lxml gotcha on
        # concatenated multi-document HTML.
        soup = BeautifulSoup(html, "html.parser")
        # Do NOT remove <header>: article title headings (h1/h2) on sites
        # like gdpr-info.eu live inside <header> elements. Removing header
        # would drop the article title and leave only boilerplate sub-sections
        # (e.g. "Suitable Recitals") as the only visible headings. <nav> is
        # kept in the list so navigation lists inside headers are still removed.
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()

        # Walk the WHOLE soup, not soup.body. ``soup.body`` returns only
        # the first <body> tag; concatenated multi-document HTML has many
        # body tags and we need to traverse them all.
        sections = self._build_section_tree(soup)

        title = (
            doc_title
            or (soup.title.get_text(strip=True) if soup.title else None)
            or "Document"
        )

        chunks: list[Chunk] = []
        idx = 0
        chunks.append(Chunk(
            id=idx, depth=0, node_path=self.root_slug,
            text=f"{self.root_slug}\n{title}",
            source_file="<html>", start_line=1, end_line=1,
        ))
        idx += 1

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
        Walk the body in document order. Each heading opens a new
        section at its level. Top-level ``<ol>``/``<ul>`` elements under
        the current section are parsed into a list tree and attached.
        """
        roots: list[_Section] = []
        stack: list[_Section] = []   # currently-open sections, deepest last
        consumed_lists: set[int] = set()  # ids of <ol>/<ul> already parsed

        for el in body.descendants:
            name = getattr(el, "name", None)
            if not name:
                continue

            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(name[1])
                while stack and stack[-1].level >= level:
                    stack.pop()
                section = _Section(
                    level=level,
                    title=el.get_text(" ", strip=True),
                    body_text="",
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

            elif name in ("ol", "ul") and self.include_list_items:
                # Only parse top-level lists. Lists nested inside a
                # parent list will be consumed by the parent's recursive
                # parse and added to ``consumed_lists`` so we skip them
                # when the descendant walk reaches them.
                if id(el) in consumed_lists:
                    continue
                if el.find_parent(["ol", "ul"]):
                    continue
                items = self._parse_list(el)
                if items:
                    current.list_items.extend(items)
                for nested in el.find_all(["ol", "ul"]):
                    consumed_lists.add(id(nested))

        return roots

    def _parse_list(self, list_el) -> list[_ListItem]:
        """Recursively parse an ``<ol>``/``<ul>`` into a tree of _ListItem.

        Each item's ``text`` is the DIRECT text of its <li> with nested
        list content excluded; nested lists become ``children`` instead,
        so each depth in the list nesting maps to a distinct chunk depth.
        """
        items: list[_ListItem] = []
        for li in list_el.find_all("li", recursive=False):
            direct_text = self._li_direct_text(li)
            children: list[_ListItem] = []
            for nested in li.find_all(["ol", "ul"], recursive=False):
                children.extend(self._parse_list(nested))
            if direct_text or children:
                items.append(_ListItem(text=direct_text, children=children))
        return items

    @staticmethod
    def _li_direct_text(li) -> str:
        """Text of a ``<li>`` element excluding nested ``<ol>``/``<ul>``.

        Walking children once (rather than re-parsing via BeautifulSoup
        as legal.py does) keeps this O(N) over the document instead of
        O(N²) on deeply nested lists.
        """
        parts: list[str] = []
        for child in li.children:
            child_name = getattr(child, "name", None)
            if child_name is None:
                text = str(child).strip()
                if text:
                    parts.append(text)
            elif child_name in ("ol", "ul"):
                continue
            else:
                text = child.get_text(" ", strip=True)
                if text:
                    parts.append(text)
        return " ".join(parts)

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

            # Child sections (recurse into deeper headings)
            next_id = self._emit_sections(
                sec.children, parent_path=path,
                parent_depth=depth, chunks=chunks, next_id=next_id,
            )

            # List items (recurse into list nesting). These coexist with
            # child sections at the same depth — both are children of
            # this section in the subtree-expand sense.
            if sec.list_items and self.include_list_items:
                next_id = self._emit_list_items(
                    sec.list_items,
                    parent_path=path,
                    parent_depth=depth,
                    section_title=sec.title,
                    chunks=chunks,
                    next_id=next_id,
                )

        return next_id

    def _emit_list_items(
        self,
        items: list[_ListItem],
        parent_path: str,
        parent_depth: int,
        section_title: str,
        chunks: list[Chunk],
        next_id: int,
    ) -> int:
        for i, item in enumerate(items, 1):
            # Skip short LEAF items only; items with children survive
            # because their children carry the content.
            if len(item.text) < self.min_chunk_chars and not item.children:
                continue
            item_path = f"{parent_path}.li{i}"
            depth = parent_depth + 1
            chunks.append(Chunk(
                id=next_id, depth=depth, node_path=item_path,
                text=f"{item_path}\n{section_title} (item {i})\n\n{item.text}",
                source_file="<html>", start_line=1, end_line=1,
            ))
            next_id += 1

            if item.children:
                next_id = self._emit_list_items(
                    item.children,
                    parent_path=item_path,
                    parent_depth=depth,
                    section_title=f"{section_title} item {i}",
                    chunks=chunks,
                    next_id=next_id,
                )

        return next_id
