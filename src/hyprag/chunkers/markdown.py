"""
hyprag.chunkers.markdown
~~~~~~~~~~~~~~~~~~~~~~~~
Source-agnostic Markdown chunker.

Uses two universal structural signals — ATX headings (``#``–``######``)
AND list nesting (``-``/``*``/``1.``) — to build a hierarchy, mirroring
``HTMLChunker``. No knowledge of any specific markdown flavour is
encoded; this works on README files, technical docs, and prose.

Hierarchy
---------
    depth 0   document root          (always emitted)
    depth 1   ``# H1``               top-level sections
    depth 2   ``## H2``              sub-sections — and top-level list
                                     items under a depth-1 heading
    depth 3   ``### H3``             sub-sub-sections — and nested list
                                     items
    depth N   ``#{N} HN``            sections, plus list nesting

Setext-style headings (``Title`` underlined with ``===`` or ``---``)
are also supported as h1/h2.

Code fences (``` ``` ```) are stripped from chunk text but their content
is kept inline within the surrounding section body. List items inside
code fences are not promoted to chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hyprag.chunker import Chunk

__all__ = ["MarkdownChunker"]


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ATX_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_H1_RE = re.compile(r"^=+\s*$")
_SETEXT_H2_RE = re.compile(r"^-+\s*$")
_BULLET_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_FENCE_RE = re.compile(r"^(\s*)(```+|~~~+)")


def _slug(text: str, max_len: int = 32) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return (s[:max_len] or "section").rstrip("-")


@dataclass
class _ListItem:
    """One list bullet parsed into direct text + nested children."""
    text: str
    children: list["_ListItem"] = field(default_factory=list)


@dataclass
class _Section:
    """A heading and everything that belongs to it before the next heading
    of equal-or-shallower level."""
    level: int                          # 1–6
    title: str
    body_text: str = ""
    list_items: list[_ListItem] = field(default_factory=list)
    children: list["_Section"] = field(default_factory=list)


class MarkdownChunker:
    """
    Chunk any Markdown document into a hierarchy driven by heading levels
    AND list nesting.

    Parameters
    ----------
    root_slug : str
        Slug for the depth-0 root chunk. Default ``"doc"``.
    min_chunk_chars : int
        Terminal list items shorter than this are suppressed. Items with
        children survive regardless. Default 40.
    include_list_items : bool
        When *True* (default), list nesting under headings contributes
        additional depth levels. Set *False* to keep the hierarchy purely
        heading-driven.
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

    def chunk_markdown(
        self, text: str, *, doc_title: str | None = None
    ) -> list[Chunk]:
        lines = text.splitlines()
        sections, inferred_title = self._build_section_tree(lines)
        title = doc_title or inferred_title or "Document"

        chunks: list[Chunk] = []
        idx = 0
        chunks.append(Chunk(
            id=idx, depth=0, node_path=self.root_slug,
            text=f"{self.root_slug}\n{title}",
            source_file="<markdown>", start_line=1, end_line=len(lines) or 1,
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

    def _build_section_tree(
        self, lines: list[str]
    ) -> tuple[list[_Section], str | None]:
        roots: list[_Section] = []
        stack: list[_Section] = []
        inferred_title: str | None = None

        in_fence = False
        fence_marker: str | None = None
        # Collect list lines, then parse the whole block when it ends so
        # indent-based nesting is preserved.
        pending_list_lines: list[str] = []

        def flush_list() -> None:
            if not pending_list_lines:
                return
            if not stack or not self.include_list_items:
                pending_list_lines.clear()
                return
            items = _parse_list_block(pending_list_lines)
            if items:
                stack[-1].list_items.extend(items)
            pending_list_lines.clear()

        def add_body(text: str) -> None:
            if not stack or not text.strip():
                return
            sec = stack[-1]
            sec.body_text = f"{sec.body_text}\n{text}" if sec.body_text else text

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Fenced code block — pass through unchanged
            m_fence = _FENCE_RE.match(line)
            if m_fence:
                marker = m_fence.group(2)[:3]
                if not in_fence:
                    flush_list()
                    in_fence = True
                    fence_marker = marker
                elif fence_marker and line.lstrip().startswith(fence_marker):
                    in_fence = False
                    fence_marker = None
                add_body(line)
                i += 1
                continue
            if in_fence:
                add_body(line)
                i += 1
                continue

            # ATX heading
            m_atx = _ATX_RE.match(line)
            if m_atx:
                flush_list()
                level = len(m_atx.group(1))
                title_text = m_atx.group(2).strip()
                while stack and stack[-1].level >= level:
                    stack.pop()
                section = _Section(level=level, title=title_text)
                if stack:
                    stack[-1].children.append(section)
                else:
                    roots.append(section)
                stack.append(section)
                if inferred_title is None and level == 1:
                    inferred_title = title_text
                i += 1
                continue

            # Setext heading: previous non-blank line underlined with === or ---
            if (
                stripped
                and i + 1 < len(lines)
                and (
                    _SETEXT_H1_RE.match(lines[i + 1])
                    or _SETEXT_H2_RE.match(lines[i + 1])
                )
                and not _BULLET_RE.match(line)
            ):
                flush_list()
                level = 1 if _SETEXT_H1_RE.match(lines[i + 1]) else 2
                title_text = stripped
                while stack and stack[-1].level >= level:
                    stack.pop()
                section = _Section(level=level, title=title_text)
                if stack:
                    stack[-1].children.append(section)
                else:
                    roots.append(section)
                stack.append(section)
                if inferred_title is None and level == 1:
                    inferred_title = title_text
                i += 2
                continue

            # List bullet
            if _BULLET_RE.match(line):
                pending_list_lines.append(line)
                i += 1
                continue

            # Blank line — terminates a list block
            if not stripped:
                flush_list()
                add_body("")
                i += 1
                continue

            # Continuation of a list item (indented under a bullet)
            if pending_list_lines and line.startswith((" ", "\t")):
                pending_list_lines.append(line)
                i += 1
                continue

            # Plain paragraph line
            flush_list()
            add_body(stripped)
            i += 1

        flush_list()
        return roots, inferred_title

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
            body_clean = sec.body_text.strip("\n")
            if body_clean:
                text_parts.append("")
                text_parts.append(body_clean)

            chunks.append(Chunk(
                id=next_id, depth=depth, node_path=path,
                text="\n".join(text_parts),
                source_file="<markdown>", start_line=1, end_line=1,
            ))
            next_id += 1

            next_id = self._emit_sections(
                sec.children, parent_path=path,
                parent_depth=depth, chunks=chunks, next_id=next_id,
            )

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
            if len(item.text) < self.min_chunk_chars and not item.children:
                continue
            item_path = f"{parent_path}.li{i}"
            depth = parent_depth + 1
            chunks.append(Chunk(
                id=next_id, depth=depth, node_path=item_path,
                text=f"{item_path}\n{section_title} (item {i})\n\n{item.text}",
                source_file="<markdown>", start_line=1, end_line=1,
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


# ---------------------------------------------------------------------------
# Helpers — indent-based list parsing
# ---------------------------------------------------------------------------

def _parse_list_block(block_lines: list[str]) -> list[_ListItem]:
    """
    Parse a block of list lines (with possible continuations and nesting)
    into a tree of ``_ListItem``.

    Uses leading-whitespace width to determine nesting. Each bullet's
    direct text excludes its nested-bullet children.
    """
    # First pass: identify bullet lines and their indent + text.
    entries: list[tuple[int, str, list[str]]] = []  # (indent, first_line_text, continuation_lines)
    current: tuple[int, str, list[str]] | None = None
    for line in block_lines:
        m = _BULLET_RE.match(line)
        if m:
            if current is not None:
                entries.append(current)
            indent = len(m.group(1).expandtabs(4))
            text = m.group(3).strip()
            current = (indent, text, [])
        else:
            if current is not None and line.strip():
                current[2].append(line.strip())
    if current is not None:
        entries.append(current)

    # Merge continuation lines into the bullet's text.
    flat: list[tuple[int, str]] = [
        (indent, " ".join([text, *cont]).strip())
        for indent, text, cont in entries
    ]
    if not flat:
        return []

    # Second pass: turn flat indent sequence into nested _ListItem tree.
    # Normalise indents to levels (0, 1, 2, …) based on order of appearance.
    indents = sorted({ind for ind, _ in flat})
    level_of = {ind: lvl for lvl, ind in enumerate(indents)}

    roots: list[_ListItem] = []
    stack: list[tuple[int, _ListItem]] = []  # (level, item)
    for indent, text in flat:
        level = level_of[indent]
        item = _ListItem(text=text)
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            stack[-1][1].children.append(item)
        else:
            roots.append(item)
        stack.append((level, item))
    return roots
