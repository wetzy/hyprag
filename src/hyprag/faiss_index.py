"""
hyprag.faiss_index
~~~~~~~~~~~~~~~~~~
Thin wrapper over ``faiss.IndexFlatIP`` with the surface API the rest of the
package expects (``.add(vectors, depths=None)``, ``.search(q, k)``,
``.ntotal``, ``.reset()``).

Why a wrapper at all?
---------------------
Older versions of this package used a Poincaré-ball index in the same slot.
Two ablations across two corpora (CPython stdlib and GDPR legal text)
established that the ball contributes nothing over flat retrieval, and adds
~13× latency. The wrapper now plays a much smaller role: it just lets every
other module continue to call ``self._index.add(vecs, depths=...)`` without
caring whether the backend is FAISS or anything else, and it gives us one
place to normalise vectors and handle the ``depths`` kwarg consistently.

Vectors are L2-normalised before being added or searched, which turns the
inner product into cosine similarity — the BGE family is trained against
cosine, so this is what we want. The ``depths`` argument is accepted for
backward compatibility (callers still pass it) and is ignored: hierarchy
information is consumed downstream by ``subtree_expand``, not by the index.
"""

from __future__ import annotations

import warnings
from typing import Sequence, Tuple

import faiss
import numpy as np

__all__ = ["FaissIndex"]


class FaissIndex:
    """
    Cosine-similarity nearest-neighbour index over flat embeddings.

    Parameters
    ----------
    dim : int
        Dimensionality of the input embeddings.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)

    # ------------------------------------------------------------------
    # FAISS-style surface
    # ------------------------------------------------------------------

    def add(
        self,
        vectors: np.ndarray,
        depths: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        """
        Append vectors to the index.

        ``depths`` is accepted and shape-checked (so older calling code keeps
        working) but does not affect retrieval. Hierarchy is used downstream
        by ``subtree_expand``.
        """
        vectors = _coerce(vectors, self.dim)
        if depths is not None:
            depths_arr = np.asarray(depths)
            if depths_arr.shape != (len(vectors),):
                raise ValueError(
                    f"depths must have one entry per vector; "
                    f"got depths.shape={depths_arr.shape}, n_vectors={len(vectors)}"
                )
        self._index.add(_l2_normalize(vectors))

    def search(
        self,
        query: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the *k* nearest neighbours of each query.

        Returns
        -------
        distances : ndarray, shape (nq, k), float32
            ``1 − cosine_similarity`` — lower is closer, matches the FAISS
            distance convention used elsewhere in this package.
        indices : ndarray, shape (nq, k), int64
            Row indices into the stored corpus. Returns ``-1`` padding when
            the corpus has fewer than k items.
        """
        if self._index.ntotal == 0:
            raise RuntimeError("Index is empty — call .add() first.")

        query = _coerce(query, self.dim)
        k_eff = min(k, self._index.ntotal)
        if k_eff < k:
            warnings.warn(
                f"Requested k={k} but index only contains {self._index.ntotal} items; "
                f"returning {k_eff} neighbours.",
                stacklevel=2,
            )

        sims, ids = self._index.search(_l2_normalize(query), k_eff)
        distances = (1.0 - sims).astype(np.float32)

        if k_eff < k:
            pad = k - k_eff
            distances = np.pad(distances, ((0, 0), (0, pad)), constant_values=np.inf)
            ids = np.pad(ids, ((0, 0), (0, pad)), constant_values=-1)

        return distances, ids.astype(np.int64)

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def reset(self) -> None:
        self._index.reset()

    def __repr__(self) -> str:
        return f"FaissIndex(dim={self.dim}, ntotal={self.ntotal})"


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _coerce(arr: np.ndarray, expected_dim: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D array, got shape {arr.shape}")
    if arr.shape[1] != expected_dim:
        raise ValueError(
            f"Dimension mismatch: index has dim={expected_dim}, "
            f"input has dim={arr.shape[1]}"
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return arr / norms
