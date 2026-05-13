"""
api.store
~~~~~~~~~
In-memory user and index stores.

These are deliberately simple — single-process, no persistence beyond the
container's lifetime. Production swap path:
  - UserStore   → Postgres (Supabase, Neon, RDS)
  - IndexStore  → Redis + S3 (hot index in Redis, cold paginated to S3)

For a Fly.io demo deployment, in-memory is fine until you have real users.
"""

from __future__ import annotations

import datetime as _dt
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from api.auth import Tier

__all__ = ["UserStore", "IndexStore", "User"]


# ---------------------------------------------------------------------------
# UserStore
# ---------------------------------------------------------------------------

@dataclass
class User:
    user_id: str
    email: str
    tier: Tier
    api_key: str
    daily_usage: dict[str, int] = field(default_factory=dict)  # date_str → queries


class UserStore:
    """In-memory user registry with daily query counters."""

    def __init__(self) -> None:
        self._users_by_key: dict[str, User] = {}
        self._users_by_email: dict[str, User] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "UserStore":
        """
        Seed from environment variables.

        ``HYPRAG_DEMO_KEY`` — pre-shared free-tier key for the public demo.
        ``HYPRAG_ADMIN_KEY`` — pre-shared paid-tier key for admin/testing.

        Both default to obvious dev keys; you MUST set them in production.
        """
        store = cls()
        demo_key = os.environ.get("HYPRAG_DEMO_KEY", "demo-key-change-me")
        admin_key = os.environ.get("HYPRAG_ADMIN_KEY", "admin-key-change-me")

        store.register(
            user_id="demo",
            email="demo@hyprag.local",
            tier="free",
            api_key=demo_key,
        )
        store.register(
            user_id="admin",
            email="admin@hyprag.local",
            tier="paid",
            api_key=admin_key,
        )
        return store

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, user_id: str, email: str, tier: Tier, api_key: str) -> User:
        user = User(user_id=user_id, email=email, tier=tier, api_key=api_key)
        with self._lock:
            self._users_by_key[api_key] = user
            self._users_by_email[email] = user
        return user

    def lookup(self, api_key: str) -> Optional[User]:
        return self._users_by_key.get(api_key)

    def set_tier_by_email(self, email: str, tier: Tier) -> bool:
        """Used by the Stripe webhook to upgrade/downgrade a user."""
        with self._lock:
            user = self._users_by_email.get(email)
            if user is None:
                return False
            user.tier = tier
            return True

    # ------------------------------------------------------------------
    # Rate-limiting
    # ------------------------------------------------------------------

    def consume_query(self, user_id: str, daily_limit: int) -> Optional[int]:
        """
        Increment today's query counter; return new count, or None if over limit.
        A daily_limit of 0 means unlimited.
        """
        today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
        with self._lock:
            user = next(
                (u for u in self._users_by_key.values() if u.user_id == user_id),
                None,
            )
            if user is None:
                return None
            count = user.daily_usage.get(today, 0)
            if daily_limit and count >= daily_limit:
                return None
            user.daily_usage[today] = count + 1
            return count + 1


# ---------------------------------------------------------------------------
# IndexStore
# ---------------------------------------------------------------------------

@dataclass
class _IndexEntry:
    retriever: object  # HypragRetriever; typed loosely to avoid circular import
    created_at: float
    expires_at: float | None


class IndexStore:
    """
    In-memory map of index_id → HypragRetriever, with TTL eviction.

    Single-process only. Survives process lifetime but not restarts.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _IndexEntry] = {}
        self._lock = threading.Lock()

    def put(self, index_id: str, retriever: object, *, ttl_seconds: int) -> None:
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds > 0 else None
        with self._lock:
            self._entries[index_id] = _IndexEntry(
                retriever=retriever,
                created_at=now,
                expires_at=expires_at,
            )

    def get(self, index_id: str) -> Optional[object]:
        with self._lock:
            entry = self._entries.get(index_id)
            if entry is None:
                return None
            if entry.expires_at and time.time() > entry.expires_at:
                del self._entries[index_id]
                return None
            return entry.retriever

    def delete(self, index_id: str) -> bool:
        with self._lock:
            return self._entries.pop(index_id, None) is not None

    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
