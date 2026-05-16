"""
tests.test_html_chunker
~~~~~~~~~~~~~~~~~~~~~~~
Tests for ``hyprag.chunkers.html_generic.HTMLChunker``.

These exist primarily as regression tests for the three structural bugs
hit while validating the chunker on the GDPR corpus:

1. ``soup.body`` returned only the first <body> tag, breaking
   concatenated multi-document HTML.
2. ``<header>`` was in the decompose list, removing article-title
   headings on sites like gdpr-info.eu and leaving only boilerplate
   sub-sections as visible headings.
3. List items under a section were suppressed when the section had
   child headings, killing paragraph emission whenever a sibling
   sub-section heading existed (e.g. "Suitable Recitals").

Plus a few smoke tests for the happy path on small synthetic HTML.
"""

from __future__ import annotations

import pytest

bs4 = pytest.importorskip("bs4")  # HTMLChunker needs beautifulsoup4

from hyprag.chunkers.html_generic import HTMLChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _depth_dist(chunks):
    dist = {}
    for c in chunks:
        dist[c.depth] = dist.get(c.depth, 0) + 1
    return dist


def _paths(chunks):
    return [c.node_path for c in chunks]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_root_only_when_no_headings():
    chunks = HTMLChunker().chunk_html("<html><body><p>no headings here</p></body></html>")
    assert len(chunks) == 1
    assert chunks[0].depth == 0
    assert chunks[0].node_path == "doc"


def test_simple_heading_hierarchy():
    html = """
    <html><body>
      <h1>Top</h1>
      <p>Top body</p>
      <h2>Middle</h2>
      <p>Middle body</p>
      <h3>Bottom</h3>
      <p>Bottom body</p>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    assert _depth_dist(chunks) == {0: 1, 1: 1, 2: 1, 3: 1}
    paths = _paths(chunks)
    assert paths == ["doc", "doc.top", "doc.top.middle", "doc.top.middle.bottom"]


def test_sibling_headings_at_same_level():
    html = """
    <html><body>
      <h1>First</h1>
      <h1>Second</h1>
      <h1>Third</h1>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    assert _depth_dist(chunks) == {0: 1, 1: 3}


def test_custom_root_slug():
    chunks = HTMLChunker(root_slug="gdpr").chunk_html(
        "<html><body><h1>Top</h1></body></html>"
    )
    assert chunks[0].node_path == "gdpr"
    assert chunks[1].node_path == "gdpr.top"


def test_doc_title_in_root_chunk():
    html = "<html><head><title>My Document</title></head><body><h1>Top</h1></body></html>"
    chunks = HTMLChunker().chunk_html(html)
    assert "My Document" in chunks[0].text


# ---------------------------------------------------------------------------
# List nesting as hierarchy signal
# ---------------------------------------------------------------------------

def test_top_level_list_items_become_depth_plus_one():
    html = """
    <html><body>
      <h1>Section</h1>
      <ol>
        <li>First item with enough text to exceed the forty character minimum.</li>
        <li>Second item, also long enough to pass the minimum chunk size filter.</li>
      </ol>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    assert _depth_dist(chunks) == {0: 1, 1: 1, 2: 2}
    item_chunks = [c for c in chunks if c.depth == 2]
    assert item_chunks[0].node_path == "doc.section.li1"
    assert item_chunks[1].node_path == "doc.section.li2"


def test_nested_list_items_become_deeper_depth():
    html = """
    <html><body>
      <h1>Section</h1>
      <ol>
        <li>Outer item one with enough characters to clear the chunk size minimum threshold.
          <ol>
            <li>Inner item a with enough text to be emitted as a chunk on its own.</li>
            <li>Inner item b with enough text to be emitted as a chunk on its own.</li>
          </ol>
        </li>
      </ol>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    assert _depth_dist(chunks) == {0: 1, 1: 1, 2: 1, 3: 2}
    inner = [c for c in chunks if c.depth == 3]
    assert inner[0].node_path == "doc.section.li1.li1"
    assert inner[1].node_path == "doc.section.li1.li2"


def test_short_terminal_list_items_are_filtered():
    html = """
    <html><body>
      <h1>Section</h1>
      <ol>
        <li>tiny</li>
        <li>also tiny</li>
        <li>This item is long enough to survive the min_chunk_chars filter cleanly.</li>
      </ol>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    item_chunks = [c for c in chunks if c.depth == 2]
    assert len(item_chunks) == 1
    assert "long enough" in item_chunks[0].text


def test_list_items_with_children_survive_short_text():
    html = """
    <html><body>
      <h1>Section</h1>
      <ol>
        <li>brief
          <ol>
            <li>This nested item is long enough to clear the minimum chunk size threshold.</li>
          </ol>
        </li>
      </ol>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    # Outer li.text is "brief" (< 40) but has children → must be emitted
    paths = _paths(chunks)
    assert "doc.section.li1" in paths
    assert "doc.section.li1.li1" in paths


def test_include_list_items_false_disables_list_chunks():
    html = """
    <html><body>
      <h1>Section</h1>
      <ol>
        <li>Long enough list item to normally produce a chunk by itself.</li>
      </ol>
    </body></html>
    """
    chunks = HTMLChunker(include_list_items=False).chunk_html(html)
    assert _depth_dist(chunks) == {0: 1, 1: 1}


# ---------------------------------------------------------------------------
# Regression: <header> must NOT be decomposed
# ---------------------------------------------------------------------------

def test_h1_inside_header_is_preserved():
    """
    Article title headings on gdpr-info.eu live inside <header>. If we
    decompose <header>, the title is lost and downstream path matching
    (looking for ``art-15`` etc.) fails on every chunk.
    """
    html = """
    <html><body>
      <header>
        <h1>Art. 15 GDPR - Right of access</h1>
      </header>
      <p>Body paragraph long enough to exceed minimum chunk size threshold.</p>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    paths = _paths(chunks)
    assert any("art-15" in p for p in paths), f"art-15 missing from {paths!r}"


def test_nav_inside_header_is_still_stripped():
    """
    We keep <header> but still decompose <nav>. Navigation links inside
    a header element should not produce chunks even if <header> survives.
    """
    html = """
    <html><body>
      <header>
        <nav>
          <ul>
            <li>Home navigation link entry</li>
            <li>About navigation link entry</li>
          </ul>
        </nav>
        <h1>Article Title</h1>
      </header>
      <p>Body paragraph long enough to exceed minimum chunk size threshold.</p>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    # The <ul> inside <nav> should be gone, so no nav-li chunks
    for c in chunks:
        assert "navigation link" not in c.text


# ---------------------------------------------------------------------------
# Regression: list items + child headings coexist
# ---------------------------------------------------------------------------

def test_list_items_emitted_alongside_child_sections():
    """
    Previously, list items were only emitted when a section had NO child
    headings. Every GDPR article has a "Suitable Recitals" sibling
    heading, which suppressed all paragraph chunks. List items and child
    sections must both survive at the same depth.
    """
    html = """
    <html><body>
      <h1>Article</h1>
      <ol>
        <li>This paragraph is long enough to be emitted as a depth-2 chunk.</li>
        <li>Another paragraph that should appear as a chunk regardless of siblings.</li>
      </ol>
      <h2>Suitable Recitals</h2>
      <p>Recital body content here, also long enough to clear the minimum.</p>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    paths = _paths(chunks)
    # Both child section AND list items must be present at depth 2
    assert "doc.article.suitable-recitals" in paths
    assert "doc.article.li1" in paths
    assert "doc.article.li2" in paths
    # All three are at depth 2 — parallel children of the article
    depth_2 = {c.node_path for c in chunks if c.depth == 2}
    assert depth_2 == {"doc.article.suitable-recitals", "doc.article.li1", "doc.article.li2"}


# ---------------------------------------------------------------------------
# Regression: concatenated multi-document HTML
# ---------------------------------------------------------------------------

def test_concatenated_documents_all_parsed():
    """
    ``soup.body`` returns only the first <body>. The GDPR corpus is 99
    stacked <html><body>...</body></html> pages, so walking soup.body
    only saw article 1. Walking the whole soup must capture all of them.
    """
    doc = """<html><body><h1>Article {n}</h1><p>Body of article {n} long enough.</p></body></html>"""
    html = "\n".join(doc.format(n=i) for i in range(1, 11))
    chunks = HTMLChunker().chunk_html(html)
    article_chunks = [c for c in chunks if c.depth == 1]
    assert len(article_chunks) == 10
    titles = {c.text.splitlines()[1] for c in article_chunks}
    assert titles == {f"Article {i}" for i in range(1, 11)}


# ---------------------------------------------------------------------------
# Decompose list still removes the things it should
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag", ["script", "style", "footer", "aside"])
def test_decomposed_tags_do_not_produce_chunks(tag):
    html = f"""
    <html><body>
      <h1>Real heading</h1>
      <{tag}>
        <h2>Heading inside {tag}</h2>
        <p>Content that should not appear in any chunk text.</p>
      </{tag}>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    for c in chunks:
        assert f"inside {tag}" not in c.text
        assert "should not appear" not in c.text


def test_nav_decomposed():
    html = """
    <html><body>
      <h1>Real heading</h1>
      <nav>
        <h2>Nav heading should vanish</h2>
      </nav>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    for c in chunks:
        assert "Nav heading" not in c.text


# ---------------------------------------------------------------------------
# Parent/child path consistency for subtree_expand
# ---------------------------------------------------------------------------

def test_parent_paths_match_node_paths():
    """
    subtree_expand walks ``parent_path``/``node_path`` relationships.
    Every non-root chunk's parent_path must match the node_path of an
    actually emitted ancestor chunk.
    """
    html = """
    <html><body>
      <h1>A</h1>
      <h2>B</h2>
      <ol>
        <li>This li item has enough text to be emitted as a real chunk.
          <ol><li>Nested item with also enough characters to clear the limit.</li></ol>
        </li>
      </ol>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    emitted = {c.node_path for c in chunks}
    for c in chunks:
        if c.depth == 0:
            assert c.parent_path == ""
        else:
            assert c.parent_path in emitted, (
                f"chunk {c.node_path!r} has parent_path={c.parent_path!r} "
                f"which is not an emitted node_path"
            )


def test_chunk_ids_are_unique_and_sequential():
    html = """
    <html><body>
      <h1>A</h1><p>aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa</p>
      <h2>B</h2><p>bbbb bbbb bbbb bbbb bbbb bbbb bbbb bbbb</p>
    </body></html>
    """
    chunks = HTMLChunker().chunk_html(html)
    ids = [c.id for c in chunks]
    assert ids == list(range(len(chunks)))
