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
# Heading classifier — now returns (kind, level_or_relative, title)
# ---------------------------------------------------------------------------

def test_word_heading_chapter():
    cls = PDFChunker()._classify_line("Chapter 3 - Data Rights")
    assert cls == ("word", 1, "Chapter 3 - Data Rights")


def test_word_heading_article():
    cls = PDFChunker()._classify_line("Article 17 - Right to Erasure")
    assert cls == ("word", 2, "Article 17 - Right to Erasure")


def test_word_heading_section_roman():
    cls = PDFChunker()._classify_line("Section IV: Enforcement")
    assert cls == ("word", 2, "Section IV - Enforcement")


def test_spanish_word_heading_articulo():
    cls = PDFChunker()._classify_line("Artículo 83 - Condiciones generales")
    assert cls is not None
    kind, level, title = cls
    assert kind == "word"
    assert level == 2
    assert "83" in title


def test_spanish_word_heading_capitulo():
    cls = PDFChunker()._classify_line("Capítulo II - Principios")
    assert cls is not None
    kind, level, _ = cls
    assert kind == "word"
    assert level == 1


def test_numbered_heading_single_level():
    cls = PDFChunker()._classify_line("1. Introduction")
    assert cls == ("numbered", 1, "1 Introduction")


def test_numbered_heading_multi_level():
    cls = PDFChunker()._classify_line("2.1.3 Encryption Requirements")
    assert cls == ("numbered", 3, "2.1.3 Encryption Requirements")


def test_numbered_list_item_not_heading():
    cls = PDFChunker()._classify_line("1. the controller shall provide notice within thirty days")
    assert cls is None


def test_allcaps_short_line_is_heading():
    cls = PDFChunker()._classify_line("DEFINITIONS")
    assert cls is not None
    kind, level, _ = cls
    assert kind == "allcaps"
    assert level == 2


def test_long_allcaps_not_heading():
    long = "THIS IS A VERY LONG ALL CAPS LINE THAT EXCEEDS THE LENGTH LIMIT FOR HEADING DETECTION AND CONTINUES"
    cls = PDFChunker()._classify_line(long)
    assert cls is None


def test_plain_paragraph_not_heading():
    cls = PDFChunker()._classify_line("The controller shall implement appropriate measures.")
    assert cls is None


# ---------------------------------------------------------------------------
# Backend validation
# ---------------------------------------------------------------------------

def test_unknown_backend_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown backend"):
        PDFChunker(backend="ghostscript")  # type: ignore[arg-type]


def test_default_backend_is_pypdf():
    assert PDFChunker().backend == "pypdf"


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


# ---------------------------------------------------------------------------
# THE KEY HIERARCHY FIX: numbered paragraphs nest under articles
# ---------------------------------------------------------------------------

def test_numbered_paragraphs_nest_under_article():
    """
    BOE/GDPR-style structure: numbered paragraphs (1., 2., 3.) under an
    article should be children of that article, not top-level sections.

    Without a chapter above, Articulo sits at depth 1 and the
    paragraphs at depth 2.
    """
    pages = [
        "Artículo 83 - Condiciones generales para multas\n"
        "1. Cada autoridad de control garantizará que la imposición de multas.\n"
        "2. Al decidir la imposición de una multa administrativa.\n"
        "5. Las infracciones de las disposiciones siguientes se sancionarán.\n"
    ]
    chunks = PDFChunker()._chunks_from_pages(pages, source="gdpr.pdf", doc_title="gdpr")

    articulo = next(
        (c for c in chunks if c.node_path.startswith("doc.articulo-83") and c.depth == 1),
        None,
    )
    assert articulo is not None, (
        f"Articulo 83 must be detected at depth 1; got paths: "
        f"{[c.node_path for c in chunks]}"
    )

    # Paragraphs 1, 2, 5 must all be children of Articulo 83 at depth 2
    paragraphs = [
        c for c in chunks
        if c.depth == 2 and c.parent_path == articulo.node_path
    ]
    assert len(paragraphs) >= 3, (
        f"expected >=3 paragraphs under {articulo.node_path}, got "
        f"{[c.node_path for c in paragraphs]}"
    )


def test_chapter_article_paragraph_three_level_nesting():
    """
    Full BOE-style nesting: Capítulo → Artículo → numbered paragraph.
    """
    pages = [
        "Capítulo III - Derechos del interesado\n"
        "Artículo 15 - Derecho de acceso\n"
        "1. El interesado tendrá derecho a obtener del responsable confirmación.\n"
        "2. Cuando se transfieran datos personales a un tercer país.\n"
        "Artículo 16 - Derecho de rectificación\n"
        "1. El interesado tendrá derecho a obtener sin dilación indebida.\n"
    ]
    chunks = PDFChunker()._chunks_from_pages(pages, source="x.pdf", doc_title="x")

    capitulo = next(c for c in chunks if "capitulo-iii" in c.node_path.lower())
    assert capitulo.depth == 1

    articulos = [c for c in chunks if c.depth == 2 and "articulo" in c.node_path]
    assert len(articulos) == 2, f"expected 2 articles, got {[a.node_path for a in articulos]}"
    for a in articulos:
        assert a.parent_path == capitulo.node_path

    paragraphs = [c for c in chunks if c.depth == 3]
    assert len(paragraphs) == 3, f"expected 3 paragraphs, got {[p.node_path for p in paragraphs]}"
    article_paths = {a.node_path for a in articulos}
    for p in paragraphs:
        assert p.parent_path in article_paths, (
            f"paragraph {p.node_path!r} parent={p.parent_path!r} not in {article_paths!r}"
        )


def test_numbered_without_word_heading_stays_top_level():
    """
    Regression: a document with no article/chapter headings but with
    "1. Section name" / "2. Section name" should still produce depth-1
    sections (not depth-2 under an imaginary parent). This is the
    common case for ordinary numbered docs.
    """
    pages = [
        "1. Introduction\nBody for section one here.\n"
        "2. Methods\nBody for section two here.\n"
        "3. Results\nBody for section three here.\n"
    ]
    chunks = PDFChunker()._chunks_from_pages(pages, source="paper.pdf", doc_title="paper")
    depth1 = [c for c in chunks if c.depth == 1]
    assert len(depth1) == 3


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
