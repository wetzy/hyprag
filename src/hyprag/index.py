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

Depth-weighted projection
--------------------------
Pass ``depths`` to ``.add()`` to encode hierarchy into geometry:

    index.add(vectors, depths=[0, 1, 1, 2, 2, 2])

depth 0 (module roots) → small Euclidean norm before expmap0 → near center.
depth max_depth (leaves) → norm ≈ ball_scale → near the boundary.
This means the Poincaré ball radius carries semantic meaning: *how specific*
a node is, not just *how similar*.
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
        Maximum Euclidean norm before expmap0.  Points with ``depth == max_depth``
        are placed at this radius.  Must be in (0, 1).  Default 0.9.
    max_depth : int
        The maximum depth value expected in your hierarchy (default 2 for the
        module → class → method three-level scheme).  Used to compute per-node
        norm when ``depths`` is supplied to ``.add()``.
    min_norm : float
        Euclidean norm used for depth-0 (root) nodes.  Must satisfy
        ``0 < min_norm < ball_scale``.  Default 0.05 places roots close to
        the origin so subtree structure fans outward naturally.
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
            Hierarchy depth for each vector.  When supplied, the Euclidean norm
            before expmap0 is linearly interpolated from ``min_norm`` (depth 0)
            to ``ball_scale`` (depth ``max_depth``), encoding hierarchy into
            radial position on the ball.  When omitted every vector is placed
            at ``ball_scale`` (backward-compatible with depth-free usage).
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
        depths: np.ndarray | None,
    ) -> torch.Tensor:
        """
        Map flat ℝ^d vectors → Poincaré ball via expmap0.

        Steps
        -----
        1. L2-normalise each vector to the unit sphere (direction preserved).
        2. Scale norms:
           - Without depth metadata: uniform ``ball_scale``.
           - With depth metadata: linear interpolation from ``_min_norm``
             (depth 0) to ``ball_scale`` (depth ``max_depth``), clamped to
             [_min_norm, ball_scale].  This encodes hierarchy radially so that
             root nodes cluster near the origin and leaves fan toward the edge.
        3. Apply expmap0: the Riemannian exponential map at the origin
           (identity on the unit ball in the Poincaré model, but geoopt's
           implementation applies the correct conformal factor).

        The resulting manifold points satisfy ``‖x‖ < 1``.
        """
        t = torch.tensor(flat, dtype=torch.float32, device=self.device)

        # Step 1: L2-normalise → unit direction vectors
        norms = t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        t_unit = t / norms

        # Step 2: scale to target norm
        if depths is not None:
            d = torch.tensor(depths, dtype=torch.float32, device=self.device)
            # Linear: depth 0 → _min_norm, depth max_depth → ball_scale
            alpha = (d / max(self.max_depth, 1)).clamp(0.0, 1.0)
            target_norm = self._min_norm + (self.ball_scale - self._min_norm) * alpha
            t_scaled = t_unit * target_norm.unsqueeze(-1)
        else:
            t_scaled = t_unit * self.ball_scale

        # Step 3: exponential map at origin
        return self.manifold.expmap0(t_scaled)

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
