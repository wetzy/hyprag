"""
hyprag.index
~~~~~~~~~~~~
Poincaré-ball nearest-neighbour index with a FAISS-compatible surface API.

Drop-in replacement pattern
----------------------------
    # Before (FAISS flat)
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    distances, ids = index.search(query, k)

    # After (HypRAG — two line change)
    index = PoincareBallIndex(dim)
    index.add(vectors)
    distances, ids = index.search(query, k)

Lift behaviour
---------------
Vectors are mapped onto the open Poincaré ball via the exponential map at
the origin (``expmap0``).  The map preserves both *direction* and *relative
magnitude* of the input — a larger input vector lands further from the
origin (closer to the boundary, where geodesic distance grows).

The ``depths`` argument to ``.add()`` is accepted for backward compatibility
but no longer alters geometry.  Prior versions forced every depth-N node to
a fixed pre-expmap norm, which (a) discarded the magnitude signal encoded
by sentence-embedding models and (b) destroyed query/document symmetry
because queries had no depth.  Depth is now a chunk annotation only; use
it downstream (e.g. subtree expansion) rather than inside the lift.
"""

from __future__ import annotations

import warnings
from typing import Sequence, Tuple

import numpy as np
import torch
import geoopt

__all__ = ["PoincareBallIndex"]


class PoincareBallIndex:
    """
    Nearest-neighbour index on the Poincaré ball (curvature c = 1 by default).

    Embeddings from any flat model are lifted to the manifold via the
    exponential map at the origin (expmap0).  Retrieval uses exact geodesic
    distance, which respects the hyperbolic geometry of the space.

    Parameters
    ----------
    dim : int
        Dimensionality of the *input* flat embeddings.
    curvature : float
        Curvature of the Poincaré ball (positive scalar, default 1.0).
        Higher values → stronger hyperbolic distortion.
    ball_scale : float
        Pre-expmap scaling factor applied to every input vector.  Must be in
        (0, 1).  Default 0.9.  Inputs of typical magnitude (≈ 1) then land at
        post-expmap norm ≈ tanh(0.9) ≈ 0.716 — comfortably away from the
        boundary, where float32 geodesic distance becomes ill-conditioned.
    max_depth : int
        Retained for backward compatibility.  No longer affects the lift —
        depth is no longer encoded geometrically.  Default 2.
    min_norm : float
        Retained for backward compatibility.  No longer affects the lift.
        Must still satisfy ``0 < min_norm < ball_scale``.  Default 0.05.
    device : str
        ``"cpu"`` or ``"cuda"``.  Defaults to CUDA when available.
    """

    def __init__(
        self,
        dim: int,
        *,
        curvature: float = 1.0,
        ball_scale: float = 0.9,
        max_depth: int = 2,
        min_norm: float = 0.05,
        device: str | None = None,
    ) -> None:
        if curvature <= 0:
            raise ValueError(f"curvature must be > 0, got {curvature}")
        if not (0 < ball_scale < 1):
            raise ValueError(f"ball_scale must be in (0, 1), got {ball_scale}")
        if max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth}")
        if not (0 < min_norm < ball_scale):
            raise ValueError(
                f"min_norm must satisfy 0 < min_norm < ball_scale; "
                f"got min_norm={min_norm}, ball_scale={ball_scale}"
            )

        self.dim = dim
        self.ball_scale = ball_scale
        self.max_depth = max_depth
        self._min_norm = min_norm
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.manifold: geoopt.PoincareBall = geoopt.PoincareBall(c=curvature)

        # Storage — filled by .add()
        self._points: torch.Tensor | None = None  # (N, dim) on manifold
        self._raw_ids: list[int] = []

    # ------------------------------------------------------------------
    # FAISS-compatible surface
    # ------------------------------------------------------------------

    def add(
        self,
        vectors: np.ndarray,
        depths: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        """
        Lift flat embeddings onto the Poincaré ball and store them.

        Parameters
        ----------
        vectors : np.ndarray
            Shape ``(n, dim)``, dtype float32 or float64.
            Can be called multiple times; items are appended (FAISS semantics).
        depths : sequence of int, optional
            Accepted and shape-validated for backward compatibility, but no
            longer used by the geometry.  Earlier versions interpolated a
            per-node pre-expmap norm from depth; that step destroyed both the
            input magnitude signal and query/document symmetry (see module
            docstring).  Pass it freely if you have it — the index will
            simply ignore it.
        """
        vectors = _coerce(vectors, self.dim)

        depths_arr: np.ndarray | None = None
        if depths is not None:
            depths_arr = np.asarray(depths, dtype=np.float32)
            if depths_arr.shape != (len(vectors),):
                raise ValueError(
                    f"depths must have one entry per vector; "
                    f"got depths.shape={depths_arr.shape}, n_vectors={len(vectors)}"
                )

        lifted = self._lift(vectors, depths_arr)

        if self._points is None:
            self._points = lifted
        else:
            self._points = torch.cat([self._points, lifted], dim=0)

        n_prev = len(self._raw_ids)
        self._raw_ids.extend(range(n_prev, n_prev + len(vectors)))

    def search(
        self,
        query: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the *k* nearest stored points to each query vector.

        Query vectors are lifted with uniform ``ball_scale`` (no depth
        assumption for queries — we search the full ball).

        Parameters
        ----------
        query : np.ndarray
            Shape ``(nq, dim)`` or ``(dim,)`` for a single query.
        k : int
            Number of neighbours to return.

        Returns
        -------
        distances : np.ndarray, shape (nq, k)
            Poincaré geodesic distances (lower → closer), float32.
        indices : np.ndarray, shape (nq, k)
            Row indices into the stored corpus (same semantics as FAISS).
            Returns -1 padding when the corpus has fewer than k items.
        """
        if self._points is None or len(self._raw_ids) == 0:
            raise RuntimeError("Index is empty — call .add() first.")

        query = _coerce(query, self.dim)
        k_eff = min(k, len(self._raw_ids))

        if k_eff < k:
            warnings.warn(
                f"Requested k={k} but index only contains {len(self._raw_ids)} items; "
                f"returning {k_eff} neighbours.",
                stacklevel=2,
            )

        # Queries are lifted without depth (uniform ball_scale)
        query_lifted = self._lift(query, depths=None)

        # Geodesic distance matrix: manifold.dist broadcasts (nq,1,d)×(1,N,d)→(nq,N)
        q = query_lifted.unsqueeze(1)
        db = self._points.unsqueeze(0)
        dist_matrix = self.manifold.dist(q, db)

        top_dists, top_idx = torch.topk(dist_matrix, k_eff, dim=1, largest=False)

        distances = top_dists.cpu().numpy().astype(np.float32)
        indices = top_idx.cpu().numpy().astype(np.int64)

        # Pad to exactly k columns (FAISS convention)
        if k_eff < k:
            pad = k - k_eff
            distances = np.pad(distances, ((0, 0), (0, pad)), constant_values=np.inf)
            indices = np.pad(indices, ((0, 0), (0, pad)), constant_values=-1)

        return distances, indices

    # ------------------------------------------------------------------
    # FAISS parity helpers
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        """Number of stored vectors (mirrors faiss.Index.ntotal)."""
        return len(self._raw_ids)

    def reset(self) -> None:
        """Remove all stored vectors (mirrors faiss.Index.reset)."""
        self._points = None
        self._raw_ids = []

    # ------------------------------------------------------------------
    # Internal geometry
    # ------------------------------------------------------------------

    def _lift(
        self,
        flat: np.ndarray,
        depths: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Map flat ℝ^d vectors → Poincaré ball via expmap0.

        Implementation
        --------------
            x = expmap0(ball_scale · v)

        The exponential map at the origin sends the entire tangent space ℝ^d
        into the open unit Poincaré ball — for c=1 it is
        ``tanh(‖v‖) · v / ‖v‖`` — so the output norm is always strictly less
        than 1 regardless of the input magnitude.

        Properties preserved by this transform:

        * **Direction.**  expmap0 is radial; rotating the input rotates the
          output by the same orthogonal transformation.
        * **Relative magnitude.**  ``‖x_a‖ < ‖x_b‖`` iff ``‖v_a‖ < ‖v_b‖``.
          Two inputs that share a direction but differ in magnitude land at
          different points on the ball — the larger one is closer to the
          boundary, with strictly greater geodesic distance from the origin.
        * **Query/document symmetry.**  The same transform is applied to
          stored vectors and to queries, so the identity vector retrieves
          itself at distance 0 regardless of any per-document metadata.

        The ``depths`` parameter is accepted for backward compatibility and
        is intentionally ignored (see module docstring).
        """
        del depths  # accepted for API compat, intentionally unused
        t = torch.tensor(flat, dtype=torch.float32, device=self.device)
        return self.manifold.expmap0(t * self.ball_scale)

    def __repr__(self) -> str:
        return (
            f"PoincareBallIndex("
            f"dim={self.dim}, "
            f"ntotal={self.ntotal}, "
            f"c={self.manifold.c.item():.2f}, "
            f"max_depth={self.max_depth}, "
            f"device={self.device})"
        )


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _coerce(arr: np.ndarray, expected_dim: int) -> np.ndarray:
    """Validate and reshape input arrays into (n, dim) float32."""
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
    return arr
