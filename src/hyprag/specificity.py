"""
hyprag.specificity
~~~~~~~~~~~~~~~~~~
Query-specificity heuristic + depth-aware rerankers.

Motivation
----------
The Path C hypothesis: the Poincaré ball has a free degree of freedom —
the radial dimension — that the current lift throws away because every
BGE-encoded chunk lands on essentially the same shell.  If we encode
hierarchy *depth* into the radius for documents and infer a target
specificity for queries, the radial dimension carries real signal and
the geodesic distance combines semantic similarity with depth-matching
in a single, principled scalar.

Pipeline (used by benchmarks/specificity_ablation.py)
-----------------------------------------------------
    1. FAISS retrieves N candidates (wide pool, e.g. N = 4·K).
    2. Re-rank candidates by combining semantic distance with
       depth-vs-specificity match.  Two flavours:
         classical : score = −‖q − c‖ + α · depth_match(c.depth, target)
         hyperbolic: score = −d_Poincaré(lift(q, target), lift(c, c.depth))
    3. Take top-K as seeds → subtree_expand → measure recall.

If the hyperbolic reranker beats the classical one, the geometry is
contributing something the linear combination can't replicate; otherwise
depth is just a useful feature and the ball is decoration.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import torch
import geoopt

__all__ = [
    "infer_query_specificity",
    "classical_rerank",
    "hyperbolic_rerank",
    "depth_aware_lift",
]


# ---------------------------------------------------------------------------
# Query → target specificity in [0, max_depth]
# ---------------------------------------------------------------------------

# Verbs that signal "I want a method, not an overview"
_ACTION_VERBS = {
    "schedule", "handle", "configure", "parse", "launch", "connect",
    "encode", "decode", "validate", "match", "join", "encode",
    "implement", "implements", "maintains", "share", "shares",
    "write", "read", "process", "compute", "execute", "run",
    "compile", "tokenize", "serialize", "deserialize", "format",
    "split", "merge", "compare", "iterate", "yield", "raise",
}

# Phrasings that pull toward the leaf end (specific method)
_SPECIFIC_PATTERNS = [
    re.compile(r"\bhow does\b", re.I),
    re.compile(r"\bhow is\b", re.I),
    re.compile(r"\bwhere is\b", re.I),
    re.compile(r"\binternally\b", re.I),
    re.compile(r"\bimplemented\b", re.I),
    re.compile(r"\bspecifically\b", re.I),
]

# Phrasings that pull toward the root end (module / overview)
_ABSTRACT_PATTERNS = [
    re.compile(r"\bwhat is\b", re.I),
    re.compile(r"\boverview\b", re.I),
    re.compile(r"\bdescribe\b", re.I),
    re.compile(r"\bin general\b", re.I),
    re.compile(r"\bwhat does the\b.*\bmodule\b", re.I),
    re.compile(r"\bwhat does the\b.*\bprovide\b", re.I),
]


def infer_query_specificity(query: str, max_depth: int = 2) -> float:
    """
    Return a continuous specificity score in ``[0, max_depth]`` for the query.

    Higher → expect leaf chunks (methods).  Lower → expect root chunks
    (modules / overview docstrings).  Halfway → class-level (depth 1).

    The heuristic is intentionally simple and explainable.  We tune by
    looking at which signals carry weight on the held-out CPython queries.

    Signals
    -------
    +0.25 each  abstract phrase  (caps at one hit)
    +0.20 each  specific phrase  (caps at two hits)
    +0.15 per   action verb      (caps at two)
    +0.10 per   CapWord (likely identifier) excluding first token
    +0.05 per   word over five, up to +0.15 (length proxy)
    """
    s = 0.45  # neutral start, slightly below class-level

    if any(p.search(query) for p in _ABSTRACT_PATTERNS):
        s -= 0.25

    specific_hits = sum(1 for p in _SPECIFIC_PATTERNS if p.search(query))
    s += 0.20 * min(specific_hits, 2)

    tokens = query.split()
    verb_hits = sum(1 for t in tokens if t.lower().rstrip("s,?.").rstrip("ing")
                    in _ACTION_VERBS)
    s += 0.15 * min(verb_hits, 2)

    cap_words = sum(1 for t in tokens[1:] if t and t[0].isupper())
    s += 0.10 * min(cap_words, 3)

    if len(tokens) > 5:
        s += 0.05 * min(len(tokens) - 5, 3)

    s = max(0.0, min(1.0, s))
    return s * max_depth


# ---------------------------------------------------------------------------
# Classical reranker — linear combo of semantic + depth-match
# ---------------------------------------------------------------------------

def classical_rerank(
    candidate_indices: Sequence[int],
    candidate_distances: Sequence[float],
    candidate_depths: Sequence[int],
    target_specificity: float,
    max_depth: int,
    alpha: float,
) -> list[int]:
    """
    Re-rank by ``score = −distance + α · (1 − |depth − target| / max_depth)``.

    Returns the candidate_indices reordered, highest score first.
    """
    if max_depth <= 0:
        max_depth = 1
    n = len(candidate_indices)
    scored = []
    for i in range(n):
        depth_match = 1.0 - abs(candidate_depths[i] - target_specificity) / max_depth
        score = -float(candidate_distances[i]) + alpha * depth_match
        scored.append((score, candidate_indices[i]))
    scored.sort(reverse=True)
    return [idx for _, idx in scored]


# ---------------------------------------------------------------------------
# Hyperbolic reranker — depth-encoded Poincaré-ball lift, geodesic distance
# ---------------------------------------------------------------------------

def depth_aware_lift(
    vecs: np.ndarray,
    depths: Sequence[float],
    ball_scale: float,
    min_norm: float,
    max_depth: int,
    manifold: geoopt.PoincareBall,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Lift Euclidean vectors onto the Poincaré ball with a radius that
    encodes hierarchy depth, while preserving the semantic direction.

    Unlike the index's main ``_lift()`` (which uses uniform radius), this
    lift deliberately uses the radial dimension as a hierarchy signal.
    Reserved for the reranker pipeline — do NOT use as the primary index
    because it discards input magnitude.
    """
    t = torch.tensor(vecs, dtype=torch.float32, device=device)
    direction = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    d = torch.tensor(np.asarray(depths, dtype=np.float32), device=device)
    md = max(max_depth, 1)
    alpha = (d / md).clamp(0.0, 1.0)
    radius = min_norm + (ball_scale - min_norm) * alpha

    return manifold.expmap0(direction * radius.unsqueeze(-1))


def hyperbolic_rerank(
    query_vec: np.ndarray,
    query_specificity: float,
    candidate_indices: Sequence[int],
    candidate_vecs: np.ndarray,
    candidate_depths: Sequence[int],
    ball_scale: float,
    min_norm: float,
    max_depth: int,
    curvature: float = 1.0,
    device: torch.device | str = "cpu",
) -> list[int]:
    """
    Re-rank candidates by geodesic distance on a depth-encoded Poincaré ball.

    The query is lifted at the radius implied by ``query_specificity``; each
    candidate is lifted at the radius implied by its own depth.  Geodesic
    distance therefore combines direction (semantics) and radial gap
    (specificity mismatch) into one scalar.
    """
    manifold = geoopt.PoincareBall(c=curvature)

    q_lifted = depth_aware_lift(
        query_vec.reshape(1, -1),
        depths=[query_specificity],
        ball_scale=ball_scale, min_norm=min_norm,
        max_depth=max_depth, manifold=manifold, device=device,
    )[0]

    c_lifted = depth_aware_lift(
        candidate_vecs,
        depths=candidate_depths,
        ball_scale=ball_scale, min_norm=min_norm,
        max_depth=max_depth, manifold=manifold, device=device,
    )

    dists = manifold.dist(q_lifted.unsqueeze(0), c_lifted).cpu().numpy()
    order = np.argsort(dists)
    return [candidate_indices[i] for i in order]
