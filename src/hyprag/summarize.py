"""
hyprag.summarize
~~~~~~~~~~~~~~~~
Pre-compute one-sentence LLM summaries for chunks to bridge the semantic gap
between natural-language queries and code implementation tokens.

Usage
-----
    from hyprag.summarize import ChunkSummarizer, apply_summaries

    summarizer = ChunkSummarizer(cache_path="summaries.json")
    summaries = summarizer.generate(chunks)          # idempotent
    texts = apply_summaries(chunks, summaries)        # use in encoder
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyprag.chunker import Chunk

__all__ = ["ChunkSummarizer", "load_summaries", "apply_summaries"]

_PROMPT = """\
You are writing one-line documentation for a Python standard library function.
Given the code node below, write exactly one sentence (≤20 words) in plain English
describing what it does. Focus on purpose and behavior, not implementation.

Node: {node_path}
{body}

One-sentence summary:"""


def _chunk_key(chunk: "Chunk") -> str:
    """16-char stable key — changes if node_path or text changes."""
    raw = f"{chunk.node_path}|{chunk.text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_prompt(chunk: "Chunk") -> str:
    body = chunk.text[:400].strip()
    return _PROMPT.format(node_path=chunk.node_path, body=body)


class ChunkSummarizer:
    """
    Async batch summarizer backed by the Anthropic API.

    Caches results to a JSON file keyed by chunk hash — re-running is
    idempotent and resumes from where it left off.

    Parameters
    ----------
    cache_path : path to the JSON cache file (created if absent)
    model      : Anthropic model ID (default: claude-haiku-4-5-20251001)
    concurrency: max parallel API requests
    api_key    : overrides ANTHROPIC_API_KEY env var when set
    """

    def __init__(
        self,
        cache_path: str | Path,
        model: str = "claude-haiku-4-5-20251001",
        concurrency: int = 20,
        api_key: str | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.model = model
        self.concurrency = concurrency
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, str]:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text()).get("summaries", {})
        return {}

    def _save_cache(self, summaries: dict[str, str]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_summaries": len(summaries),
            "summaries": summaries,
        }
        self.cache_path.write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------
    # Async core
    # ------------------------------------------------------------------

    async def _call_one(
        self,
        client,
        chunk: "Chunk",
        sem: asyncio.Semaphore,
    ) -> tuple[str, str]:
        async with sem:
            msg = await client.messages.create(
                model=self.model,
                max_tokens=60,
                messages=[{"role": "user", "content": _build_prompt(chunk)}],
            )
        summary = msg.content[0].text.strip().rstrip(".")
        return _chunk_key(chunk), summary

    async def _run_async(
        self, chunks: list["Chunk"], existing: dict[str, str]
    ) -> dict[str, str]:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed — run: pip install anthropic"
            )

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        sem = asyncio.Semaphore(self.concurrency)

        todo = [c for c in chunks if _chunk_key(c) not in existing]
        print(f"  {len(existing):,} cached, {len(todo):,} to generate…")

        summaries = dict(existing)
        done = 0

        for coro in asyncio.as_completed(
            [self._call_one(client, c, sem) for c in todo]
        ):
            key, summary = await coro
            summaries[key] = summary
            done += 1
            if done % 500 == 0 or done == len(todo):
                self._save_cache(summaries)
                print(f"    {done:,}/{len(todo):,} done — saved.")

        return summaries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, chunks: list["Chunk"]) -> dict[str, str]:
        """
        Generate summaries for all chunks, using and updating the cache.

        Returns a dict mapping chunk_key → summary string.
        """
        existing = self._load_cache()
        summaries = asyncio.run(self._run_async(chunks, existing))
        self._save_cache(summaries)
        return summaries


# ------------------------------------------------------------------
# Helpers used by the benchmark
# ------------------------------------------------------------------

def load_summaries(cache_path: str | Path) -> dict[str, str]:
    """Load a pre-computed summary cache. Returns key→summary map."""
    return json.loads(Path(cache_path).read_text()).get("summaries", {})


def apply_summaries(
    chunks: list["Chunk"],
    summaries: dict[str, str],
) -> list[str]:
    """
    Build embedding texts: replace chunk text with node_path + summary
    when a summary is available; fall back to original text otherwise.

    Returns a list of strings parallel to *chunks*.
    """
    out: list[str] = []
    for c in chunks:
        key = _chunk_key(c)
        if key in summaries:
            out.append(f"{c.node_path}\n{summaries[key]}")
        else:
            out.append(c.text)
    return out
