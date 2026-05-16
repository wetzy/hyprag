"""
tests.test_pdf_chunker
~~~~~~~~~~~~~~~~~~~~~~
Tests for ``hyprag.chunkers.pdf.PDFChunker``.

Tests focus on the heading classifier and section-tree assembly. We do
not need a real PDF file: by injecting page-text strings through
``_chunks_from_pages`` we exercise every code path.
"""

from __future__ import annotations

from hyprag.chunkers.pdf import PDFChunker


def _paths(chunks):
    return [c.node_path for c in chunks]


# ---------------------------------------------------------------------------
# Heading classifier
# ---------------------------------------------------------------------------

def test_word_heading_chapter():
    cls = PDFChunker()._classify_line("Chapter 3 - Data Rights")
    assert cls == (1, "Chapter 3 - Data Rights")


def test_word_heading_article():
    cls = PDFChunker()._classify_line("Article 17 - Right to Erasure")
    assert cls == (2, "Article 17 - Right to Erasure")


def test_word_heading_section_roman():
    cls = PDFChunker()._classify_line("Section IV: Enforcement")
    assert cls == (2, "Section IV - Enforcement")


def test_numbered_heading_single_level():
    cls = PDFChunker()._classify_line("1. Introduction")
    assert cls == (1, "1 Introduction")


def test_numbered_heading_multi_level():
    cls = PDFChunker()._classify_line("2.1.3 Encryption Requirements")
    assert cls == (3, "2.1.3 Encryption Requirements")


def test_numbered_list_item_not_heading():
    # Numbered list item with lowercase continuation should NOT be a heading
    cls = PDFChunker()._classify_line("1. the controller shall provide notice within thirty days")
    assert cls is None


def test_allcaps_short_line_is_heading():
    cls = PDFChunker()._classify_line("DEFINITIONS")
    assert cls is not None
    assert cls[0] == 2


def test_long_allcaps_not_heading():
    # Very long all-caps lines are usually noise, not headings
    long = "THIS IS A VERY LONG ALL CAPS LINE THAT EXCEEDS THE LENGTH LIMIT FOR HEADING DETECTION AND CONTINUES"
    cls = PDFChunker()._classify_line(long)
    assert cls is None


def test_plain_paragraph_not_heading():
    cls = PDFChunker()._classify_line("The controller shall implement appropriate measures.")
    assert cls is None


# ---------------------------------------------------------------------------
# Tree assembly via _chunks_from_pages
# ---------------------------------------------------------------------------

def test_numbered_section_hierarchy():
    pages = [
        "1. Introduction\nIntro text body content here for the section.\n"
        "2. Main Body\n2.1 Sub Section\nSub body text here.\n2.2 Another Sub\nMore body.\n"
    ]
    chunks = PDFChunker()._chunks_from_pages(pages, source="/tmp/fake.pdf", doc_title="fake")
    paths = _paths(chunks)
    assert paths[0] == "doc"
    # Section 1 and 2 at depth 1, with subsections under 2 at depth 2
    assert any(p.startswith("doc.1-introduction") for p in paths)
    assert any(p.startswith("doc.2-main-body") for p in paths)
    assert any(".2-1-sub-section" in p for p in paths)


def test_chapter_article_hierarchy():
    pages = [
        "Chapter 3 - Data Rights\nGeneral chapter intro paragraph here.\n"
        "Article 15 - Right of Access\nThe data subject shall have the right...\n"
        "Article 17 - Right to Erasure\nThe data subject shall have the right to obtain...\n"
    ]
    chunks = PDFChunker()._chunks_from_pages(pages, source="contract.pdf", doc_title="contract")
    paths = _paths(chunks)
    # Chapter at depth 1, articles at depth 2 (children)
    chapter = [c for c in chunks if "chapter-3" in c.node_path and c.depth == 1]
    articles = [c for c in chunks if "article-15" in c.node_path or "article-17" in c.node_path]
    assert len(chapter) == 1
    assert len(articles) >= 2
    for a in articles:
        assert a.parent_path.endswith("chapter-3-data-rights")


def test_page_fallback_when_no_headings():
    pages = [
        "Just some prose with no obvious heading structure at all.",
        "Another page of prose, also without any headings present.",
    ]
    chunks = PDFChunker()._chunks_from_pages(pages, source="prose.pdf", doc_title="prose")
    paths = _paths(chunks)
    assert "doc.page1" in paths
    assert "doc.page2" in paths


def test_empty_pdf_returns_single_chunk_with_notice():
    chunks = PDFChunker()._chunks_from_pages(["", ""], source="empty.pdf", doc_title="empty")
    assert len(chunks) == 1
    assert "no extractable text" in chunks[0].text


def test_parent_paths_match_emitted_node_paths():
    pages = [
        "Chapter 1 - First\nIntro text.\n"
        "Article 1 - Definitions\nDef body.\n"
        "Article 2 - Scope\nScope body.\n"
    ]
    chunks = PDFChunker()._chunks_from_pages(pages, source="x.pdf", doc_title="x")
    emitted = {c.node_path for c in chunks}
    for c in chunks:
        if c.depth == 0:
            assert c.parent_path == ""
        else:
            assert c.parent_path in emitted
