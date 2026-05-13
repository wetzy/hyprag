"""
hyprag.bm25
~~~~~~~~~~~
BM25 lexical index over HypRAG Chunk objects.

No external dependencies — uses a hand-rolled inverted index so that the
per-query cost is O(|q_terms| * avg_postings_list) rather than O(n_docs).
For a 16k-chunk corpus this keeps search under 5ms even on CPU.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np

__all__ = ["BM25Index"]


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric boundaries, drop single chars."""
    return [t for t in re.split(r"[^a-z0-9_]", text.lower()) if len(t) > 1]


class BM25Index:
    """
    BM25 sparse retrieval index.

    Build once after chunking, then call .search() per query.
    Thread-safe for concurrent reads after .build() completes.

    Parameters
    ----------
    k1 : float
        Term-frequency saturation. Higher → more weight on repeated terms.
        Okapi BM25 default: 1.5.
    b : float
        Length normalisation. 1.0 = full normalisation, 0.0 = none.
        Okapi BM25 default: 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._n: int = 0
        self._avgdl: float = 1.0
        self._dl: list[int] = []
        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        self._inverted: dict[str, list[int]] = {}  # term → posting list (doc_ids)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build(self, texts: Sequence[str]) -> None:
        """
        Tokenise and index all texts. Replaces any previous index state.

        Parameters
        ----------
        texts : sequence of str
            One string per chunk, in corpus order (index == chunk id).
        """
        self._n = len(texts)
        self._tf = []
        self._dl = []
        self._df = {}
        self._inverted = {}
        total_len = 0

        for doc_id, text in enumerate(texts):
            tokens = _tokenize(text)
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            self._tf.append(tf_map)
            dl = len(tokens)
            self._dl.append(dl)
            total_len += dl
            for term in tf_map:
                self._df[term] = self._df.get(term, 0) + 1
                if term not in self._inverted:
                    self._inverted[term] = []
                self._inverted[term].append(doc_id)

        self._avgdl = total_len / self._n if self._n > 0 else 1.0

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the top-k BM25 matches for *query*.

        Parameters
        ----------
        query : str
            Natural-language or code query.
        k : int
            Number of results to return.

        Returns
        -------
        scores : ndarray, shape (k,)
            BM25 scores, descending.
        ids : ndarray, shape (k,), dtype int64
            Corpus indices corresponding to *scores*.
        """
        if self._n == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        q_terms = _tokenize(query)
        if not q_terms:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        # Accumulate BM25 scores via the inverted index (sparse traversal)
        score_map: dict[int, float] = {}
        for term in set(q_terms):  # deduplicate query terms
            if term not in self._inverted:
                continue
            df = self._df[term]
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id in self._inverted[term]:
                tf = self._tf[doc_id].get(term, 0)
                dl = self._dl[doc_id]
                norm_tf = tf * (self.k1 + 1.0) / (
                    tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                )
                score_map[doc_id] = score_map.get(doc_id, 0.0) + idf * norm_tf

        if not score_map:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        # Convert to arrays and partial-sort for top-k
        doc_ids_arr = np.array(list(score_map.keys()), dtype=np.int64)
        scores_arr = np.array(list(score_map.values()), dtype=np.float32)

        top_k = min(k, len(scores_arr))
        if top_k < len(scores_arr):
            # argpartition is O(n), then sort only the top-k slice
            part = np.argpartition(scores_arr, -top_k)[-top_k:]
            part = part[np.argsort(scores_arr[part])[::-1]]
        else:
            part = np.argsort(scores_arr)[::-1]

        return scores_arr[part], doc_ids_arr[part]
