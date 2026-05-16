"""
tests.test_sources
~~~~~~~~~~~~~~~~~~
Tests for the dispatcher in ``hyprag.sources``. Focuses on detection
logic and routing; URL fetching is mocked because we don't want
network in unit tests.
"""

from __future__ import annotations

import pytest

bs4 = pytest.importorskip("bs4")  # HTMLChunker / MarkdownChunker dispatch

from hyprag.sources import chunks_from_source, sniff_kind


# ---------------------------------------------------------------------------
# sniff_kind
# ---------------------------------------------------------------------------

def test_sniff_url():
    assert sniff_kind("https://en.wikipedia.org/wiki/GDPR") == "url"
    assert sniff_kind("http://example.com/foo") == "url"


def test_sniff_html_string():
    assert sniff_kind("<html><body><h1>x</h1></body></html>") == "html"
    assert sniff_kind("<div>just a div</div>") == "html"


def test_sniff_markdown_string():
    assert sniff_kind("# A heading\n\nbody text here") == "markdown"


def test_sniff_plain_text():
    assert sniff_kind("just some words with nothing structural about them") == "text"


def test_sniff_existing_path(tmp_path):
    f = tmp_path / "foo.txt"
    f.write_text("hello", encoding="utf-8")
    assert sniff_kind(str(f)) == "path"


def test_sniff_nonexistent_path_falls_through_to_text():
    # A long path that doesn't exist should NOT be flagged as a path
    assert sniff_kind("/this/path/does/not/exist/anywhere.txt") == "text"


# ---------------------------------------------------------------------------
# chunks_from_source: dispatch by type
# ---------------------------------------------------------------------------

def test_dispatch_list_of_strings():
    chunks = chunks_from_source(["one", "two", "three"])
    assert len(chunks) == 3
    assert {c.node_path for c in chunks} == {"text0", "text1", "text2"}
    for c in chunks:
        assert c.parent_path == ""


def test_dispatch_html_string():
    html = "<html><body><h1>Top</h1><p>body</p></body></html>"
    chunks = chunks_from_source(html)
    paths = [c.node_path for c in chunks]
    assert "doc" in paths
    assert "doc.top" in paths


def test_dispatch_markdown_string():
    md = "# Top\n\nbody paragraph here that has enough text to mean something."
    chunks = chunks_from_source(md)
    paths = [c.node_path for c in chunks]
    assert "doc" in paths
    assert "doc.top" in paths


def test_dispatch_plain_text_string():
    text = (
        "First paragraph of plain text that should be long enough.\n\n"
        "Second paragraph that also exceeds the minimum chunk size easily."
    )
    chunks = chunks_from_source(text)
    paths = [c.node_path for c in chunks]
    assert paths[0] == "doc"
    assert any(p.startswith("doc.p") for p in paths)


def test_dispatch_html_file(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text(
        "<html><body><h1>From File</h1><p>body</p></body></html>",
        encoding="utf-8",
    )
    chunks = chunks_from_source(str(f))
    paths = [c.node_path for c in chunks]
    assert "doc.from-file" in paths


def test_dispatch_markdown_file(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# From File\n\nbody content here.", encoding="utf-8")
    chunks = chunks_from_source(str(f))
    paths = [c.node_path for c in chunks]
    assert "doc.from-file" in paths


def test_dispatch_plain_text_file(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text(
        "First plain text paragraph long enough to qualify.\n\n"
        "Second plain text paragraph also long enough.",
        encoding="utf-8",
    )
    chunks = chunks_from_source(str(f))
    paths = [c.node_path for c in chunks]
    assert paths[0] == "doc"
    assert any(p.startswith("doc.p") for p in paths)


def test_dispatch_directory(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    (tmp_path / "b.txt").write_text(
        "First paragraph long enough.\n\nSecond paragraph long enough.",
        encoding="utf-8",
    )
    chunks = chunks_from_source(tmp_path)
    # Both files should contribute chunks
    assert len(chunks) >= 4
    # Ids must be unique and sequential across the whole directory
    ids = [c.id for c in chunks]
    assert ids == list(range(len(chunks)))


def test_unsupported_source_type_raises():
    with pytest.raises(TypeError):
        chunks_from_source(12345)
