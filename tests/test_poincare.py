"""
Tests for the Poincaré-ball index in src/hyprag/index.py.

The implementation was audited and three bugs in ``_lift()`` were fixed:

  Bug #1  L2-normalisation destroyed input magnitude information.
  Bug #2  Forced per-depth radial shells overrode embedding semantics.
  Bug #3  Queries lifted at leaf radius broke query/document symmetry.

The fix replaces ``_lift()`` with ``expmap0(ball_scale · v)`` — a single
radial scaling followed by the exponential map at the origin.  Every
section below targets either a mathematical invariant of the new lift,
the public surface, or one of the three former bugs (now asserted as
*absent*).

Sections
--------
1.  Construction & input validation
2.  Ball-membership invariant (every lifted point has ‖x‖ < 1)
3.  Fix #1 — magnitude preserved through the lift
4.  Fix #2 — depth metadata does not change geometry
5.  Fix #3 — query/document symmetry
6.  Geodesic distance — metric axioms
7.  Rotation equivariance — expmap0 commutes with orthogonal transforms
8.  FAISS-compatible search surface (.add / .search / padding)
9.  Index lifecycle (.ntotal / .reset / repeated .add)
10. Determinism
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hyprag.index import PoincareBallIndex


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------

DIM = 8


def _random_vecs(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """Random unit-ish embeddings — float32, BGE-style magnitude ~1."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return v


@pytest.fixture
def index() -> PoincareBallIndex:
    return PoincareBallIndex(dim=DIM, device="cpu")


@pytest.fixture
def small_corpus() -> np.ndarray:
    return _random_vecs(16, DIM, seed=1)


# ----------------------------------------------------------------------
# 1. Construction & input validation
# ----------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        idx = PoincareBallIndex(dim=DIM, device="cpu")
        assert idx.dim == DIM
        assert idx.ntotal == 0
        assert idx.ball_scale == 0.9

    @pytest.mark.parametrize("bad_curvature", [0.0, -1.0])
    def test_rejects_non_positive_curvature(self, bad_curvature):
        with pytest.raises(ValueError, match="curvature"):
            PoincareBallIndex(dim=DIM, curvature=bad_curvature, device="cpu")

    @pytest.mark.parametrize("bad_scale", [0.0, 1.0, 1.5, -0.1])
    def test_rejects_ball_scale_outside_unit_interval(self, bad_scale):
        with pytest.raises(ValueError, match="ball_scale"):
            PoincareBallIndex(dim=DIM, ball_scale=bad_scale, device="cpu")

    def test_rejects_negative_max_depth(self):
        # max_depth is now a no-op geometrically but its constructor
        # validation is still part of the public contract.
        with pytest.raises(ValueError, match="max_depth"):
            PoincareBallIndex(dim=DIM, max_depth=-1, device="cpu")

    @pytest.mark.parametrize(
        "min_norm,ball_scale",
        [(0.0, 0.9), (0.95, 0.9), (-0.1, 0.9), (0.9, 0.9)],
    )
    def test_rejects_invalid_min_norm(self, min_norm, ball_scale):
        with pytest.raises(ValueError, match="min_norm"):
            PoincareBallIndex(
                dim=DIM, min_norm=min_norm, ball_scale=ball_scale, device="cpu"
            )


# ----------------------------------------------------------------------
# 2. Ball-membership invariant
# ----------------------------------------------------------------------

class TestBallMembership:
    """Every lifted point must satisfy ‖x‖ < 1 (open unit Poincaré ball)."""

    def test_lifted_points_strictly_inside_unit_ball(self, index, small_corpus):
        index.add(small_corpus)
        norms = index._points.norm(dim=-1).cpu().numpy()
        assert np.all(norms < 1.0), f"max norm {norms.max()} >= 1"
        assert np.all(np.isfinite(norms))

    def test_ball_invariant_holds_for_extreme_magnitudes(self, index):
        """Inputs of huge or tiny norm must still land strictly inside the ball."""
        base = _random_vecs(1, DIM, seed=2)
        for scale in [1e-4, 1.0, 10.0, 1e3]:
            index.reset()
            index.add(base * scale)
            n = index._points[0].norm().item()
            assert 0.0 <= n < 1.0, f"scale={scale} produced norm {n}"

    def test_origin_vector_maps_to_origin(self, index):
        zero = np.zeros((1, DIM), dtype=np.float32)
        index.add(zero)
        assert index._points[0].norm().item() == pytest.approx(0.0, abs=1e-7)


# ----------------------------------------------------------------------
# 3. Fix #1 — magnitude preserved through the lift
# ----------------------------------------------------------------------

class TestMagnitudePreserved:
    """
    Pre-fix, ``_lift()`` did ``t / ‖t‖``, collapsing all vectors with the
    same direction onto a single point.  Post-fix, magnitude is carried
    through the exponential map and shows up in the radial component.
    """

    def test_same_direction_different_magnitude_yield_different_points(self, index):
        # Magnitudes are deliberately chosen inside tanh's float32-usable range
        # (‖v‖·ball_scale ≲ 5).  Beyond that, tanh saturates and the test
        # would degenerate — see test_extreme_magnitudes_saturate_at_boundary
        # which documents that regime explicitly.
        direction = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                             dtype=np.float32)
        direction /= np.linalg.norm(direction)
        small = direction * 0.3
        large = direction * 1.5
        index.add(np.vstack([small, large]))
        p_small = index._points[0].cpu().numpy()
        p_large = index._points[1].cpu().numpy()
        # They MUST differ — this is the inverse of the old bug assertion.
        assert not np.allclose(p_small, p_large, atol=1e-3)
        assert np.linalg.norm(p_small) < np.linalg.norm(p_large)

    def test_extreme_magnitudes_saturate_at_boundary(self, index):
        """
        Documents the upper-magnitude regime: for ‖v‖·ball_scale ≫ 1, tanh
        saturates and any two vectors with the same direction round to the
        same point near the boundary in float32.  This is a property of the
        exponential map, not a bug — it's the right thing for the geometry
        (the boundary represents 'infinitely specific'), and any caller
        relying on magnitude differentiation should keep ‖v‖ ≲ 1.
        """
        direction = _random_vecs(1, DIM, seed=8)  # unit direction
        index.add(direction * np.float32(100.0))
        index.add(direction * np.float32(1000.0))
        norms = index._points.norm(dim=-1).cpu().numpy()
        # Both saturate to geoopt's internal numerical clip (just below 1.0)
        # — the same clip for both inputs, so they land on top of each other.
        assert norms[0] == pytest.approx(norms[1], abs=1e-5)
        assert norms[0] > 0.99
        assert norms[0] < 1.0  # ball invariant still holds

    def test_larger_input_lands_closer_to_boundary(self, index):
        direction = _random_vecs(1, DIM, seed=3)
        index.add(np.vstack([direction * 0.5, direction * 5.0]))
        n_small = index._points[0].norm().item()
        n_large = index._points[1].norm().item()
        assert n_small < n_large

    def test_lift_is_monotonic_in_input_magnitude(self, index):
        direction = _random_vecs(1, DIM, seed=4)
        scales = [0.1, 0.5, 1.0, 2.0, 5.0, 20.0]
        vecs = np.vstack([direction * s for s in scales])
        index.add(vecs)
        norms = index._points.norm(dim=-1).cpu().numpy()
        # Strictly increasing post-lift norms.
        assert np.all(np.diff(norms) > 0), f"non-monotonic: {norms}"

    def test_radial_direction_preserved(self, index):
        """expmap0 acts only on the radial component; the unit direction is fixed."""
        vec = np.array([[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                       dtype=np.float32)
        index.add(vec)
        lifted = index._points[0].cpu().numpy()
        input_dir = vec[0] / np.linalg.norm(vec[0])
        out_dir = lifted / (np.linalg.norm(lifted) + 1e-12)
        np.testing.assert_allclose(out_dir, input_dir, atol=1e-5)


# ----------------------------------------------------------------------
# 4. Fix #2 — depth metadata does not change geometry
# ----------------------------------------------------------------------

class TestDepthIgnored:
    """
    Pre-fix, the same embedding at different depths landed on different
    radial shells.  Post-fix, ``depths`` is accepted for API compat but
    has no effect on the lifted position.
    """

    def test_same_embedding_different_depth_same_lifted_point(self):
        idx = PoincareBallIndex(
            dim=DIM, max_depth=2, min_norm=0.05, ball_scale=0.9, device="cpu"
        )
        v = _random_vecs(1, DIM, seed=42)
        idx.add(np.vstack([v, v, v]), depths=[0, 1, 2])
        p0 = idx._points[0]
        p1 = idx._points[1]
        p2 = idx._points[2]
        assert torch.allclose(p0, p1, atol=1e-6)
        assert torch.allclose(p1, p2, atol=1e-6)

    def test_geodesic_between_identical_embeddings_is_zero(self):
        """Direct rebuttal of bug #2's characterisation test."""
        idx = PoincareBallIndex(
            dim=DIM, max_depth=2, min_norm=0.05, ball_scale=0.9, device="cpu"
        )
        v = _random_vecs(1, DIM, seed=42)
        idx.add(np.vstack([v, v]), depths=[0, 2])
        d = idx.manifold.dist(idx._points[0], idx._points[1]).item()
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_depth_arg_still_validated_for_shape(self, index, small_corpus):
        """Shape validation is still part of the public contract."""
        with pytest.raises(ValueError, match="depths"):
            index.add(small_corpus, depths=[0, 1])  # too few

    def test_depth_arg_accepted_and_ignored(self, index, small_corpus):
        """Passing depths must not raise, and must produce the same points
        as omitting them."""
        index.add(small_corpus, depths=[0] * len(small_corpus))
        with_depth = index._points.clone()

        index.reset()
        index.add(small_corpus)
        without_depth = index._points.clone()

        assert torch.allclose(with_depth, without_depth, atol=1e-6)


# ----------------------------------------------------------------------
# 5. Fix #3 — query/document symmetry
# ----------------------------------------------------------------------

class TestQueryDocumentSymmetry:
    """
    Pre-fix, queries were forced to ``ball_scale`` while documents went onto
    depth-dependent shells, so the index could not return self-similarity 0
    for a stored vector if its depth was anything other than ``max_depth``.
    Post-fix, the same transform is applied to both sides.
    """

    def test_self_retrieval_at_zero_distance_regardless_of_stored_depth(self):
        v = _random_vecs(1, DIM, seed=7)
        for stored_depth in (0, 1, 2):
            idx = PoincareBallIndex(
                dim=DIM, max_depth=2, min_norm=0.05, ball_scale=0.9, device="cpu"
            )
            idx.add(v, depths=[stored_depth])
            d, ids = idx.search(v, k=1)
            assert ids[0, 0] == 0
            assert d[0, 0] == pytest.approx(0.0, abs=1e-5), (
                f"stored_depth={stored_depth} did not recover self at d=0"
            )

    def test_identical_docs_at_different_depths_tie_on_search(self):
        """
        Pre-fix: the leaf-depth copy always won.  Post-fix: both copies are
        co-located on the ball, so their distances to a query are equal.
        """
        v = _random_vecs(1, DIM, seed=11)
        idx = PoincareBallIndex(
            dim=DIM, max_depth=2, min_norm=0.05, ball_scale=0.9, device="cpu"
        )
        idx.add(np.vstack([v, v]), depths=[0, 2])
        q = _random_vecs(1, DIM, seed=12)
        d, _ = idx.search(q, k=2)
        assert d[0, 0] == pytest.approx(d[0, 1], abs=1e-5)


# ----------------------------------------------------------------------
# 6. Geodesic distance — metric axioms
# ----------------------------------------------------------------------

class TestGeodesicDistance:
    def test_self_distance_is_zero(self, index, small_corpus):
        index.add(small_corpus)
        d, ids = index.search(small_corpus[0:1], k=1)
        assert ids[0, 0] == 0
        assert d[0, 0] == pytest.approx(0.0, abs=1e-4)

    def test_symmetry(self, index):
        a = _random_vecs(1, DIM, seed=10)
        b = _random_vecs(1, DIM, seed=11)
        index.add(np.vstack([a, b]))
        d_ab, _ = index.search(a, k=2)
        d_ba, _ = index.search(b, k=2)
        cross_ab = max(d_ab[0])
        cross_ba = max(d_ba[0])
        assert cross_ab == pytest.approx(cross_ba, rel=1e-4)

    def test_non_negative_and_finite(self, index, small_corpus):
        index.add(small_corpus)
        d, _ = index.search(small_corpus, k=5)
        assert np.all(d >= 0)
        assert np.all(np.isfinite(d))

    def test_triangle_inequality(self, index):
        a = _random_vecs(1, DIM, seed=20)
        b = _random_vecs(1, DIM, seed=21)
        c = _random_vecs(1, DIM, seed=22)
        index.add(np.vstack([a, b, c]))
        pts = index._points
        d = index.manifold.dist
        d_ab = d(pts[0], pts[1]).item()
        d_bc = d(pts[1], pts[2]).item()
        d_ac = d(pts[0], pts[2]).item()
        assert d_ac <= d_ab + d_bc + 1e-5


# ----------------------------------------------------------------------
# 7. Rotation equivariance
# ----------------------------------------------------------------------

class TestRotationEquivariance:
    """
    expmap0 is radial, so applying an orthogonal transformation U to the
    inputs is equivalent to applying U to the outputs.  Geodesic distances
    must therefore be invariant under any common rotation of two points.
    """

    @staticmethod
    def _random_orthogonal(d: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((d, d))
        q, _ = np.linalg.qr(a)
        return q.astype(np.float32)

    def test_pairwise_geodesic_invariant_under_rotation(self):
        u = self._random_orthogonal(DIM, seed=99)
        a = _random_vecs(1, DIM, seed=50)
        b = _random_vecs(1, DIM, seed=51)

        idx1 = PoincareBallIndex(dim=DIM, device="cpu")
        idx1.add(np.vstack([a, b]))
        d_orig = idx1.manifold.dist(idx1._points[0], idx1._points[1]).item()

        idx2 = PoincareBallIndex(dim=DIM, device="cpu")
        idx2.add(np.vstack([a @ u, b @ u]))
        d_rot = idx2.manifold.dist(idx2._points[0], idx2._points[1]).item()

        assert d_orig == pytest.approx(d_rot, rel=1e-4)


# ----------------------------------------------------------------------
# 8. FAISS-compatible search surface
# ----------------------------------------------------------------------

class TestSearchAPI:
    def test_search_returns_top_k_in_order(self, index, small_corpus):
        index.add(small_corpus)
        d, ids = index.search(small_corpus[:3], k=5)
        assert d.shape == (3, 5)
        assert ids.shape == (3, 5)
        for row in d:
            assert np.all(np.diff(row) >= -1e-5)

    def test_single_query_1d_input_accepted(self, index, small_corpus):
        index.add(small_corpus)
        d, ids = index.search(small_corpus[0], k=3)
        assert d.shape == (1, 3)
        assert ids.shape == (1, 3)

    def test_search_pads_when_k_exceeds_ntotal(self, index):
        v = _random_vecs(3, DIM)
        index.add(v)
        with pytest.warns(UserWarning, match="only contains"):
            d, ids = index.search(v[0:1], k=5)
        assert d.shape == (1, 5)
        assert ids.shape == (1, 5)
        assert (ids[0, 3:] == -1).all()
        assert np.isinf(d[0, 3:]).all()

    def test_search_on_empty_index_raises(self, index):
        with pytest.raises(RuntimeError, match="empty"):
            index.search(_random_vecs(1, DIM), k=1)

    def test_dim_mismatch_raises(self, index):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            index.add(_random_vecs(2, DIM + 1))

    def test_returns_int64_ids_and_float32_distances(self, index, small_corpus):
        index.add(small_corpus)
        d, ids = index.search(small_corpus[0:1], k=3)
        assert ids.dtype == np.int64
        assert d.dtype == np.float32


# ----------------------------------------------------------------------
# 9. Index lifecycle
# ----------------------------------------------------------------------

class TestLifecycle:
    def test_ntotal_tracks_add_calls(self, index, small_corpus):
        assert index.ntotal == 0
        index.add(small_corpus[:5])
        assert index.ntotal == 5
        index.add(small_corpus[5:])
        assert index.ntotal == len(small_corpus)

    def test_reset_clears_index(self, index, small_corpus):
        index.add(small_corpus)
        index.reset()
        assert index.ntotal == 0
        assert index._points is None
        with pytest.raises(RuntimeError):
            index.search(small_corpus[0:1], k=1)

    def test_repeated_add_concatenates(self, index):
        a = _random_vecs(4, DIM, seed=30)
        b = _random_vecs(6, DIM, seed=31)
        index.add(a)
        index.add(b)
        assert index.ntotal == 10
        assert index._points.shape == (10, DIM)


# ----------------------------------------------------------------------
# 10. Determinism
# ----------------------------------------------------------------------

class TestDeterminism:
    def test_lift_is_deterministic(self):
        v = _random_vecs(8, DIM, seed=100)
        idx1 = PoincareBallIndex(dim=DIM, device="cpu")
        idx2 = PoincareBallIndex(dim=DIM, device="cpu")
        idx1.add(v)
        idx2.add(v)
        assert torch.allclose(idx1._points, idx2._points, atol=1e-6)
