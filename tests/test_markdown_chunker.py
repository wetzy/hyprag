"""
tests.test_markdown_chunker
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for ``hyprag.chunkers.markdown.MarkdownChunker``.
"""

from __future__ import annotations

from hyprag.chunkers.markdown import MarkdownChunker


def _depth_dist(chunks):
    dist = {}
    for c in chunks:
        dist[c.depth] = dist.get(c.depth, 0) + 1
    return dist


def _paths(chunks):
    return [c.node_path for c in chunks]


# ---------------------------------------------------------------------------
# Happy path: headings → hierarchy
# ---------------------------------------------------------------------------

def test_root_only_when_no_headings():
    chunks = MarkdownChunker().chunk_markdown("just a paragraph with no heading at all")
    assert len(chunks) == 1
    assert chunks[0].depth == 0
    assert chunks[0].node_path == "doc"


def test_atx_heading_hierarchy():
    md = """
# Top
top body

## Middle
middle body

### Bottom
bottom body
"""
    chunks = MarkdownChunker().chunk_markdown(md)
    paths = _paths(chunks)
    assert paths == ["doc", "doc.top", "doc.top.middle", "doc.top.middle.bottom"]
    assert _depth_dist(chunks) == {0: 1, 1: 1, 2: 1, 3: 1}


def test_sibling_headings_at_same_level():
    md = """
# First
# Second
# Third
"""
    chunks = MarkdownChunker().chunk_markdown(md)
    assert _depth_dist(chunks) == {0: 1, 1: 3}


def test_setext_heading_h1():
    md = """
Setext Title
============

body line one
"""
    chunks = MarkdownChunker().chunk_markdown(md)
    paths = _paths(chunks)
    assert paths == ["doc", "doc.setext-title"]


def test_setext_heading_h2():
    md = """
# Top

Setext Sub
----------

body
"""
    chunks = MarkdownChunker().chunk_markdown(md)
    paths = _paths(chunks)
    assert "doc.top.setext-sub" in paths


def test_doc_title_inferred_from_first_h1():
    md = "# My Document\n\nbody"
    chunks = MarkdownChunker().chunk_markdown(md)
    assert "My Document" in chunks[0].text


def test_explicit_doc_title_overrides():
    md = "# Other\n\nbody"
    chunks = MarkdownChunker().chunk_markdown(md, doc_title="Forced Title")
    assert "Forced Title" in chunks[0].text


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------

def test_top_level_bullets_become_chunks():
    md = """
# Section

- First bullet with enough text to exceed the forty character minimum threshold.
- Second bullet, also long enough to pass the minimum chunk size filter check.
"""
    chunks = MarkdownChunker().chunk_markdown(md)
    item_chunks = [c for c in chunks if c.depth == 2]
    assert len(item_chunks) == 2
    assert item_chunks[0].node_path == "doc.section.li1"
    assert item_chunks[1].node_path == "doc.section.li2"


def test_nested_bullets_become_deeper_depth():
    md = """
# Section

- Outer bullet one with enough characters to clear the chunk size minimum threshold here.
  - Inner bullet a with enough text to be emitted as a chunk on its own line.
  - Inner bullet b with enough text to be emitted as a chunk on its own line.
"""
    chunks = MarkdownChunker().chunk_markdown(md)
    inner = [c for c in chunks if c.depth == 3]
    assert len(inner) == 2
    assert inner[0].node_path == "doc.section.li1.li1"
    assert inner[1].node_path == "doc.section.li1.li2"


def test_numbered_list_treated_like_bullets():
    md = """
# Section

1. First numbered item, long enough to clear the minimum chunk size threshold.
2. Second numbered item, also long enough to pass the chunk minimum easily.
"""
    chunks = MarkdownChunker().chunk_markdown(md)
    item_chunks = [c for c in chunks if c.depth == 2]
    assert len(item_chunks) == 2


def test_short_leaf_bullets_filtered():
    md = """
# Section

- tiny
- also tiny
- This bullet is long enough to survive the minimum chunk size filter cleanly.
"""
    chunks = MarkdownChunker().chunk_markdown(md)
    item_chunks = [c for c in chunks if c.depth == 2]
    assert len(item_chunks) == 1


def test_include_list_items_false_disables_list_chunks():
    md = """
# Section

- Long enough list item to normally produce a chunk by itself easily here.
"""
    chunks = MarkdownChunker(include_list_items=False).chunk_markdown(md)
    assert _depth_dist(chunks) == {0: 1, 1: 1}


# ---------------------------------------------------------------------------
# Code fences should not be parsed as headings or bullets
# ---------------------------------------------------------------------------

def test_code_fence_hash_not_treated_as_heading():
    md = """
# Real Section

```
# This is a Python comment, not a heading
- This is shell output, not a bullet
```

paragraph after
"""
    chunks = MarkdownChunker().chunk_markdown(md)
    paths = _paths(chunks)
    # Only the real heading should produce a section chunk
    assert paths == ["doc", "doc.real-section"]


# ---------------------------------------------------------------------------
# Parent paths consistent with emitted node paths
# ---------------------------------------------------------------------------

def test_parent_paths_match_emitted_node_paths():
    md = """
# A

## B

- This li item has enough text to be emitted as a real chunk here.
  - Nested item with also enough characters to clear the limit cleanly.
"""
    chunks = MarkdownChunker().chunk_markdown(md)
    emitted = {c.node_path for c in chunks}
    for c in chunks:
        if c.depth == 0:
            assert c.parent_path == ""
        else:
            assert c.parent_path in emitted, (
                f"chunk {c.node_path!r} parent={c.parent_path!r} not in {emitted!r}"
            )


def test_chunk_ids_are_unique_and_sequential():
    md = """
# A

aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa

## B

bbbb bbbb bbbb bbbb bbbb bbbb bbbb bbbb
"""
    chunks = MarkdownChunker().chunk_markdown(md)
    ids = [c.id for c in chunks]
    assert ids == list(range(len(chunks)))
