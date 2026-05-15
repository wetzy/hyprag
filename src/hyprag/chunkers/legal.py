"""
hyprag.chunkers.legal
~~~~~~~~~~~~~~~~~~~~~
Chunker for hierarchical legal documents — specifically GDPR (EU 2016/679).

Hierarchy produced
------------------
    depth 0  gdpr                          whole-document root
    depth 1  gdpr.ch3                      chapter
    depth 2  gdpr.ch3.art15                article
    depth 3  gdpr.ch3.art15.p1             numbered paragraph
    depth 4  gdpr.ch3.art15.p1.pa          lettered point  (a), (b) …

parent_path is derived automatically from node_path by the Chunk dataclass,
so subtree_expand works without any changes.

The chunker is DOM-driven (gdpr-info.eu structure). Article boundaries come
from URLs (one article per page); paragraph and point boundaries come from
``<ol><li>`` nesting, not from text patterns. Article-level chunk text is
assembled from the parsed paragraphs so the encoder sees clean numbered
prose rather than a get_text() dump that flattens lists into space-joined
soup.

Usage
-----
    chunker = GDPRChunker()
    chunks = chunker.load()                          # fetches gdpr-info.eu
    chunks = chunker.load(html_path=Path("..."))     # local concatenated HTML
    chunks = chunker.load(html_string="<html>...")   # in-memory HTML
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from hyprag.chunker import Chunk

__all__ = ["GDPRChunker"]


GDPR_INFO_URL = "https://gdpr-info.eu/art-{n}-gdpr/"

# Article number → (chapter_number, chapter_slug). Static; comes from the
# regulation itself, not from any specific scraping source.
_CHAPTER_MAP: dict[int, tuple[int, str]] = {}
_RANGES = [
    (range(1, 5),    1,  "general_provisions"),
    (range(5, 12),   2,  "principles"),
    (range(12, 24),  3,  "rights_data_subject"),
    (range(24, 44),  4,  "controller_processor"),
    (range(44, 51),  5,  "third_country_transfers"),
    (range(51, 60),  6,  "supervisory_authorities"),
    (range(60, 77),  7,  "cooperation_consistency"),
    (range(77, 85),  8,  "remedies_liability"),
    (range(85, 92),  9,  "specific_situations"),
    (range(92, 94),  10, "delegated_acts"),
    (range(94, 100), 11, "final_provisions"),
]
for _rng, _cn, _cs in _RANGES:
    for _art in _rng:
        _CHAPTER_MAP[_art] = (_cn, _cs)


@dataclass
class _Article:
    """Intermediate representation of one parsed article."""
    n: int
    title: str
    paragraphs: list["_Paragraph"]
    source: str


@dataclass
class _Paragraph:
    n: int                        # 1, 2, 3 … from <ol><li> position
    text: str                     # direct text of the <li> (no nested ol)
    points: list[tuple[str, str]] # (letter, text) for nested <li> items


class GDPRChunker:
    """
    Parse the GDPR into hierarchical Chunk objects.

    Parameters
    ----------
    min_para_chars : int
        Paragraphs shorter than this are skipped (kept only inside the
        article-level chunk). Default 40.
    min_point_chars : int
        Same for lettered points. Default 40.
    """

    def __init__(
        self,
        min_para_chars: int = 40,
        min_point_chars: int = 40,
    ) -> None:
        self.min_para_chars = min_para_chars
        self.min_point_chars = min_point_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        html_path: Path | None = None,
        html_string: str | None = None,
    ) -> list[Chunk]:
        """
        Load and chunk the GDPR.

        Resolution order: ``html_string`` > ``html_path`` > network fetch
        (per-article from gdpr-info.eu).
        """
        if html_string is not None:
            return self._chunks_from_articles(self._parse_html(html_string))
        if html_path is not None:
            html = html_path.read_text(encoding="utf-8", errors="replace")
            return self._chunks_from_articles(self._parse_html(html))
        return self._chunks_from_articles(self._fetch_articles())

    # ------------------------------------------------------------------
    # Fetching — per-article from gdpr-info.eu (EUR-Lex is Cloudflare-blocked)
    # ------------------------------------------------------------------

    def _fetch_articles(self) -> list[_Article]:
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "pip install requests beautifulsoup4  (needed by GDPRChunker)"
            ) from exc

        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "pip install beautifulsoup4  (needed by GDPRChunker)"
            ) from exc

        headers = {"User-Agent": "Mozilla/5.0 (research; hyprag)"}
        articles: list[_Article] = []

        for n in range(1, 100):
            url = GDPR_INFO_URL.format(n=n)
            html = None
            for attempt in range(3):
                try:
                    resp = requests.get(url, headers=headers, timeout=20)
                    resp.raise_for_status()
                    html = resp.text
                    break
                except Exception:
                    if attempt == 2:
                        break
                    time.sleep(1.5 ** attempt)
            if html is None:
                continue
            art = self._parse_article_html(html, art_num=n, source=url)
            if art is not None:
                articles.append(art)
            time.sleep(0.3)

        return articles

    # ------------------------------------------------------------------
    # Parsing — DOM-driven, no regex on plain text
    # ------------------------------------------------------------------

    def _parse_html(self, html: str) -> list[_Article]:
        """
        Parse a (possibly concatenated) HTML document containing GDPR articles.

        Each article is detected as either:
          - a top-level ``<article>`` element with a recognisable heading, OR
          - a ``<div class="entry-content">`` block preceded by an ``<h1>``
            that contains an article number.

        Falls back to single-article parsing if neither pattern matches.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError("pip install beautifulsoup4") from exc

        soup = BeautifulSoup(html, "html.parser")  # not lxml — see note below
        # NOTE: ``html.parser`` is intentional. ``lxml`` silently collapses
        # repeated <html>/<body> blocks in concatenated documents, dropping
        # every article after the first.

        # Find every entry-content block; each one is a candidate article body.
        entry_contents = soup.find_all("div", class_="entry-content")
        if not entry_contents:
            # Single-article path
            art = self._parse_article_html(html, art_num=None, source="<inline>")
            return [art] if art else []

        articles: list[_Article] = []
        for ec in entry_contents:
            # Find the article number from the nearest preceding <h1> or <h2>
            heading = ec.find_previous(["h1", "h2"])
            art_num = _extract_article_number(heading.get_text() if heading else "")
            if art_num is None:
                continue
            title = heading.get_text(strip=True) if heading else f"Article {art_num}"
            paragraphs = self._extract_paragraphs(ec)
            articles.append(_Article(
                n=art_num, title=title, paragraphs=paragraphs, source="<inline>"
            ))

        return articles

    def _parse_article_html(
        self,
        html: str,
        art_num: int | None,
        source: str,
    ) -> _Article | None:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        # Drop the recitals sidebar gdpr-info.eu appends below the article body
        for tag in soup.find_all("div", class_="empfehlung-erwaegungsgruende"):
            tag.decompose()

        entry = soup.find("div", class_="entry-content") or soup.find("article")
        if not entry:
            return None

        if art_num is None:
            heading = soup.find(["h1", "h2"])
            art_num = _extract_article_number(heading.get_text() if heading else "")
            if art_num is None:
                return None

        heading = soup.find(["h1", "h2"])
        title = heading.get_text(strip=True) if heading else f"Article {art_num}"
        paragraphs = self._extract_paragraphs(entry)
        return _Article(n=art_num, title=title, paragraphs=paragraphs, source=source)

    def _extract_paragraphs(self, entry) -> list[_Paragraph]:
        """
        Extract numbered paragraphs and their lettered points from the article
        body. Structure on gdpr-info.eu:

            <div class="entry-content">
              <ol>
                <li> paragraph 1 text
                  <ol>
                    <li> point (a) text </li>
                    <li> point (b) text </li>
                  </ol>
                </li>
                <li> paragraph 2 text </li>
              </ol>
            </div>

        Paragraph numbers come from <li> position, NOT from text patterns.
        Point letters likewise (chr('a' + index)).
        """
        top_ol = entry.find("ol")
        if not top_ol:
            # Some articles have a single unnumbered paragraph; fall back to
            # the entry-content text as paragraph 1.
            body = entry.get_text(" ", strip=True)
            return [_Paragraph(n=1, text=body, points=[])] if body else []

        paragraphs: list[_Paragraph] = []
        for idx, li in enumerate(top_ol.find_all("li", recursive=False), 1):
            para_text = _li_direct_text(li)
            if not para_text:
                continue

            points: list[tuple[str, str]] = []
            nested_ol = li.find("ol")
            if nested_ol:
                for j, pt_li in enumerate(nested_ol.find_all("li", recursive=False)):
                    pt_text = pt_li.get_text(" ", strip=True)
                    if pt_text:
                        points.append((chr(ord("a") + j), pt_text))

            paragraphs.append(_Paragraph(n=idx, text=para_text, points=points))

        return paragraphs

    # ------------------------------------------------------------------
    # Chunk assembly
    # ------------------------------------------------------------------

    def _chunks_from_articles(self, articles: list[_Article]) -> list[Chunk]:
        chunks: list[Chunk] = []
        idx = 0
        root_path = "gdpr"

        # depth 0 — root
        chunks.append(Chunk(
            id=idx, depth=0, node_path=root_path,
            text=(
                "gdpr\n"
                "REGULATION (EU) 2016/679 — General Data Protection Regulation\n\n"
                "EU regulation on the protection of natural persons with regard to "
                "the processing of personal data and on the free movement of such data."
            ),
            source_file="gdpr-info.eu",
            start_line=1, end_line=1,
        ))
        idx += 1

        emitted_chapters: set[int] = set()

        for art in sorted(articles, key=lambda a: a.n):
            if art.n not in _CHAPTER_MAP:
                continue
            ch_num, ch_slug = _CHAPTER_MAP[art.n]
            ch_path = f"{root_path}.ch{ch_num}"

            # depth 1 — chapter (once)
            if ch_num not in emitted_chapters:
                chunks.append(Chunk(
                    id=idx, depth=1, node_path=ch_path,
                    text=(
                        f"{ch_path}\n"
                        f"Chapter {ch_num} — {ch_slug.replace('_', ' ').title()}"
                    ),
                    source_file="gdpr-info.eu",
                    start_line=art.n, end_line=art.n,
                ))
                idx += 1
                emitted_chapters.add(ch_num)

            art_path = f"{ch_path}.art{art.n}"

            # depth 2 — article. Text is built from the parsed paragraphs so
            # the encoder sees structured prose, not a flattened DOM dump.
            art_text = _build_article_text(art_path, art.title, art.paragraphs)
            chunks.append(Chunk(
                id=idx, depth=2, node_path=art_path,
                text=art_text,
                source_file=art.source,
                start_line=art.n, end_line=art.n,
            ))
            idx += 1

            # depth 3 — paragraphs (skip very short ones)
            for para in art.paragraphs:
                if len(para.text) < self.min_para_chars:
                    continue
                para_path = f"{art_path}.p{para.n}"
                chunks.append(Chunk(
                    id=idx, depth=3, node_path=para_path,
                    text=(
                        f"{para_path}\n"
                        f"{art.title}, paragraph {para.n}\n\n"
                        f"{para.text}"
                    ),
                    source_file=art.source,
                    start_line=art.n, end_line=art.n,
                ))
                idx += 1

                # depth 4 — lettered points
                for letter, pt_text in para.points:
                    if len(pt_text) < self.min_point_chars:
                        continue
                    pt_path = f"{para_path}.p{letter}"
                    chunks.append(Chunk(
                        id=idx, depth=4, node_path=pt_path,
                        text=(
                            f"{pt_path}\n"
                            f"{art.title} §{para.n}({letter})\n\n"
                            f"{pt_text}"
                        ),
                        source_file=art.source,
                        start_line=art.n, end_line=art.n,
                    ))
                    idx += 1

        return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _li_direct_text(li_tag) -> str:
    """
    Return the text of a ``<li>`` element WITHOUT its nested ``<ol>`` children.

    Without this, ``li.get_text()`` would include all the lettered-point text
    inside the paragraph, polluting the paragraph chunk with content that
    already has its own (depth-4) chunks.
    """
    from bs4 import BeautifulSoup
    node = BeautifulSoup(str(li_tag), "html.parser").find("li")
    if node is None:
        return ""
    for nested in node.find_all("ol"):
        nested.decompose()
    return node.get_text(" ", strip=True)


def _extract_article_number(heading_text: str) -> int | None:
    """Extract the article number from a heading like 'Art. 15 GDPR - …'."""
    import re
    m = re.search(r'\bArt(?:icle|\.)?\s*(\d+)\b', heading_text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _build_article_text(
    art_path: str,
    title: str,
    paragraphs: list[_Paragraph],
    max_chars: int = 1200,
) -> str:
    """
    Build the depth-2 article chunk text from the parsed paragraphs.

    The previous implementation took ``content.get_text(' ')[:600]`` which
    collapsed nested lists into space-joined soup and inlined cross-reference
    link text (e.g. "Article 89(1)"). That degraded encoder quality. Here we
    rebuild the article as clean numbered prose, which the encoder embeds far
    better — at the cost of slightly longer text (still capped).
    """
    parts: list[str] = [art_path, title, ""]
    used = sum(len(p) for p in parts)

    for para in paragraphs:
        snippet = f"{para.n}. {para.text}"
        if used + len(snippet) > max_chars and parts[-1] != "":
            break
        parts.append(snippet)
        used += len(snippet) + 1

    return "\n".join(parts)
