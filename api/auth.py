"""
api.auth
~~~~~~~~
API-key authentication context and tier-limit configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Tier = Literal["free", "paid"]

__all__ = ["APIKeyAuth", "TierLimits", "TIER_LIMITS"]


@dataclass(frozen=True)
class APIKeyAuth:
    """Per-request authenticated user context."""
    user_id: str
    tier: Tier
    raw_key: str  # kept only for audit logging; never returned in responses


@dataclass(frozen=True)
class TierLimits:
    max_vectors: int
    daily_queries: int      # 0 = unlimited
    ttl_seconds: int        # 0 = never expire
    max_archive_bytes: int

    @property
    def daily_queries_unlimited(self) -> bool:
        return self.daily_queries == 0


TIER_LIMITS: dict[Tier, TierLimits] = {
    "free": TierLimits(
        max_vectors=100_000,
        daily_queries=100,
        ttl_seconds=7 * 24 * 3600,           # 7 days
        max_archive_bytes=50 * 1_000_000,    # 50 MB
    ),
    "paid": TierLimits(
        max_vectors=10_000_000,
        daily_queries=0,                     # unlimited
        ttl_seconds=0,                       # persistent
        max_archive_bytes=500 * 1_000_000,   # 500 MB
    ),
}
