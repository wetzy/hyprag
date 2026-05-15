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

Usage
-----
    chunker = GDPRChunker()
    chunks = chunker.load()          # downloads from EUR-Lex
    # or
    chunks = chunker.load(html_path=Path("gdpr.html"))   # local file
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from hyprag.chunker import Chunk

__all__ = ["GDPRChunker"]

EURLEX_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
    "?uri=CELEX:32016R0679"
)

# Article number → (chapter_number, chapter_slug)
_CHAPTER_MAP: dict[int, tuple[int, str]] = {}
_RANGES = [
    (range(1, 5),   1,  "general_provisions"),
    (range(5, 12),  2,  "principles"),
    (range(12, 24), 3,  "rights_data_subject"),
    (range(24, 44), 4,  "controller_processor"),
    (range(44, 51), 5,  "third_country_transfers"),
    (range(51, 60), 6,  "supervisory_authorities"),
    (range(60, 77), 7,  "cooperation_consistency"),
    (range(77, 85), 8,  "remedies_liability"),
    (range(85, 92), 9,  "specific_situations"),
    (range(92, 94), 10, "delegated_acts"),
    (range(94, 100),11, "final_provisions"),
]
for _rng, _cn, _cs in _RANGES:
    for _art in _rng:
        _CHAPTER_MAP[_art] = (_cn, _cs)


class GDPRChunker:
    """
    Parse the GDPR into hierarchical Chunk objects.

    Parameters
    ----------
    min_para_chars : int
        Paragraphs shorter than this are merged into the article-level chunk
        instead of being emitted as depth-3 chunks.  Default 80.
    """

    def __init__(self, min_para_chars: int = 80) -> None:
        self.min_para_chars = min_para_chars

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

        Parameters
        ----------
        html_path : Path, optional
            Path to a locally saved GDPR HTML file.
        html_string : str, optional
            Raw HTML passed in-process (e.g. through an HTTP request body).
            Takes precedence over ``html_path``.

        When both are omitted the text is fetched from EUR-Lex (requires
        network access; the live site sits behind Cloudflare so this is
        unreliable — prefer per-article curl + concatenation upstream).
        """
        if html_string is not None:
            return self._parse(html_string)
        text = self._fetch(html_path)
        return self._parse(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch(self, html_path: Path | None) -> str:
        if html_path is not None:
            return html_path.read_text(encoding="utf-8", errors="replace")

        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests  (needed to fetch GDPR from EUR-Lex)")

        headers = {"User-Agent": "Mozilla/5.0 (research; hyprag benchmark)"}
        for attempt in range(3):
            try:
                resp = requests.get(EURLEX_URL, headers=headers, timeout=30)
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Failed to fetch GDPR from EUR-Lex: {exc}") from exc
                time.sleep(2 ** attempt)
        return ""  # unreachable

    def _parse(self, html: str) -> list[Chunk]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("pip install beautifulsoup4 lxml  (needed by GDPRChunker)")

        soup = BeautifulSoup(html, "lxml")

        # Remove script/style noise
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        raw_text = soup.get_text(separator="\n")
        return self._build_chunks(raw_text)

    def _build_chunks(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        id_counter = 0

        # ── Root chunk (depth 0) ──────────────────────────────────────────
        root_path = "gdpr"
        root_text = (
            "gdpr\n"
            "REGULATION (EU) 2016/679 — General Data Protection Regulation\n\n"
            "EU regulation on the protection of natural persons with regard to "
            "the processing of personal data and on the free movement of such data."
        )
        chunks.append(Chunk(
            id=id_counter, text=root_text, depth=0,
            node_path=root_path, source_file="EUR-Lex:32016R0679",
            start_line=1, end_line=1,
        ))
        id_counter += 1

        # ── Split into articles ───────────────────────────────────────────
        # EUR-Lex typically renders each article starting with "Article N"
        article_pattern = re.compile(
            r'(?:^|\n)\s*(Article\s+(\d+))\s*\n',
            re.IGNORECASE,
        )
        parts = article_pattern.split(text)
        # parts = [pre, "Article 1", "1", body1, "Article 2", "2", body2, ...]

        # Track which chapters we have already emitted
        emitted_chapters: set[int] = set()

        i = 1  # skip preamble in parts[0]
        while i + 2 < len(parts):
            art_header = parts[i].strip()       # "Article 1"
            art_num_str = parts[i + 1].strip()  # "1"
            art_body = parts[i + 2]             # everything until next article
            i += 3

            try:
                art_num = int(art_num_str)
            except ValueError:
                continue

            if art_num not in _CHAPTER_MAP:
                continue

            ch_num, ch_slug = _CHAPTER_MAP[art_num]
            ch_path = f"{root_path}.ch{ch_num}"

            # ── Chapter chunk (depth 1) — emit once per chapter ───────────
            if ch_num not in emitted_chapters:
                ch_label = ch_slug.replace("_", " ").title()
                ch_text = f"{ch_path}\nChapter {ch_num} — {ch_label}"
                chunks.append(Chunk(
                    id=id_counter, text=ch_text, depth=1,
                    node_path=ch_path, source_file="EUR-Lex:32016R0679",
                    start_line=art_num, end_line=art_num,
                ))
                id_counter += 1
                emitted_chapters.add(ch_num)

            # ── Extract article title (first non-empty line of body) ───────
            body_lines = art_body.split("\n")
            art_title = ""
            body_start_idx = 0
            for idx, line in enumerate(body_lines):
                stripped = line.strip()
                if stripped and not stripped[0].isdigit() and stripped != art_header:
                    art_title = stripped
                    body_start_idx = idx + 1
                    break

            art_path = f"{ch_path}.art{art_num}"
            full_article_text = "\n".join(body_lines).strip()

            # ── Article chunk (depth 2) ───────────────────────────────────
            art_chunk_text = (
                f"{art_path}\n"
                f"{art_header}"
                + (f" — {art_title}" if art_title else "")
                + f"\n\n{full_article_text[:600]}"
            )
            chunks.append(Chunk(
                id=id_counter, text=art_chunk_text, depth=2,
                node_path=art_path, source_file="EUR-Lex:32016R0679",
                start_line=art_num, end_line=art_num,
            ))
            id_counter += 1

            # ── Paragraph chunks (depth 3) ────────────────────────────────
            paragraphs = self._split_paragraphs(body_lines[body_start_idx:])
            for para_num, (para_label, para_text) in enumerate(paragraphs, 1):
                if len(para_text) < self.min_para_chars:
                    continue

                para_path = f"{art_path}.p{para_num}"
                para_chunk_text = (
                    f"{para_path}\n"
                    f"{art_header}"
                    + (f" — {art_title}" if art_title else "")
                    + f", paragraph {para_num}\n\n{para_text}"
                )
                chunks.append(Chunk(
                    id=id_counter, text=para_chunk_text, depth=3,
                    node_path=para_path, source_file="EUR-Lex:32016R0679",
                    start_line=art_num, end_line=art_num,
                ))
                id_counter += 1

                # ── Point chunks (depth 4, lettered sub-items) ───────────
                points = self._split_points(para_text)
                for point_letter, point_text in points:
                    if len(point_text) < self.min_para_chars:
                        continue
                    point_path = f"{para_path}.p{point_letter}"
                    point_chunk_text = (
                        f"{point_path}\n"
                        f"{art_header} §{para_num}({point_letter})\n\n"
                        f"{point_text}"
                    )
                    chunks.append(Chunk(
                        id=id_counter, text=point_chunk_text, depth=4,
                        node_path=point_path, source_file="EUR-Lex:32016R0679",
                        start_line=art_num, end_line=art_num,
                    ))
                    id_counter += 1

        # Re-number IDs to guarantee contiguous range
        for idx, c in enumerate(chunks):
            c.id = idx

        return chunks

    # ------------------------------------------------------------------
    # Text splitting helpers
    # ------------------------------------------------------------------

    _PARA_RE = re.compile(r'^\s*(\d+)\.\s+(.+)', re.DOTALL)
    _POINT_RE = re.compile(r'^\s*\(([a-z])\)\s+(.+)', re.DOTALL)

    def _split_paragraphs(self, lines: list[str]) -> list[tuple[str, str]]:
        """
        Split article body into numbered paragraphs (1. text, 2. text …).
        Returns list of (label, full_text) tuples.
        Falls back to one paragraph containing the whole body.
        """
        paragraphs: list[tuple[str, str]] = []
        current_label = ""
        current_lines: list[str] = []

        for line in lines:
            m = self._PARA_RE.match(line)
            if m:
                if current_lines:
                    paragraphs.append((current_label, " ".join(current_lines).strip()))
                current_label = m.group(1)
                current_lines = [m.group(2).strip()]
            else:
                stripped = line.strip()
                if stripped:
                    current_lines.append(stripped)

        if current_lines:
            paragraphs.append((current_label, " ".join(current_lines).strip()))

        if not paragraphs:
            body = " ".join(l.strip() for l in lines if l.strip())
            if body:
                paragraphs = [("1", body)]

        return paragraphs

    def _split_points(self, para_text: str) -> list[tuple[str, str]]:
        """
        Split a paragraph into lettered points (a) text, (b) text …
        Returns list of (letter, text) tuples.
        """
        points: list[tuple[str, str]] = []
        current_letter = ""
        current_lines: list[str] = []

        for line in para_text.split("\n"):
            m = self._POINT_RE.match(line)
            if m:
                if current_lines and current_letter:
                    points.append((current_letter, " ".join(current_lines).strip()))
                current_letter = m.group(1)
                current_lines = [m.group(2).strip()]
            else:
                stripped = line.strip()
                if stripped and current_letter:
                    current_lines.append(stripped)

        if current_lines and current_letter:
            points.append((current_letter, " ".join(current_lines).strip()))

        return points
