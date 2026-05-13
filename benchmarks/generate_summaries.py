"""
benchmarks.generate_summaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-compute LLM summaries for every chunk in a corpus.  Idempotent —
re-running resumes from the existing cache and only generates missing entries.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...

    python -m benchmarks.generate_summaries \
        --cpython-lib cpython/Lib \
        --out benchmarks/results/summaries.json

Then pass the result to the benchmark:

    python -m benchmarks.run_benchmark \
        --cpython-lib cpython/Lib \
        --summaries benchmarks/results/summaries.json

Cost estimate (claude-haiku-4-5, 16k chunks):
    Input:  ~1.4M tokens  → $0.35
    Output: ~240k tokens  → $0.30
    Total:  ~$0.65 one-time
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hyprag.chunker import HierarchicalChunker
from hyprag.summarize import ChunkSummarizer

EXCLUDE_DIRS = {"test", "tests", "idlelib", "turtledemo", "__pycache__"}


def load_corpus(lib_path: Path) -> list:
    chunker = HierarchicalChunker()
    chunks: list = []
    for py_file in sorted(lib_path.rglob("*.py")):
        if any(part in EXCLUDE_DIRS for part in py_file.parts):
            continue
        file_chunks = chunker.chunk_file(py_file)
        for c in file_chunks:
            c.id += len(chunks)
        chunks.extend(file_chunks)
    return chunks


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate LLM summaries for all corpus chunks."
    )
    p.add_argument(
        "--cpython-lib", type=Path, required=True,
        help="Path to the cpython/Lib directory",
    )
    p.add_argument(
        "--out", type=Path, default=Path("benchmarks/results/summaries.json"),
        help="Output cache file (default: benchmarks/results/summaries.json)",
    )
    p.add_argument(
        "--model", default="claude-haiku-4-5-20251001",
        help="Anthropic model ID for summaries",
    )
    p.add_argument(
        "--concurrency", type=int, default=20,
        help="Parallel API requests (default 20)",
    )
    args = p.parse_args()

    if not args.cpython_lib.exists():
        raise SystemExit(f"Path not found: {args.cpython_lib}")

    print(f"Loading corpus from {args.cpython_lib}…")
    chunks = load_corpus(args.cpython_lib)
    print(f"  → {len(chunks):,} chunks")

    print(f"\nGenerating summaries with {args.model} (concurrency={args.concurrency})…")
    print(f"  Output: {args.out}")
    summarizer = ChunkSummarizer(
        cache_path=args.out,
        model=args.model,
        concurrency=args.concurrency,
    )
    summaries = summarizer.generate(chunks)
    print(f"\nDone. {len(summaries):,} summaries saved to {args.out}")


if __name__ == "__main__":
    main()
