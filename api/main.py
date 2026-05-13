"""
hyprag.api.main
~~~~~~~~~~~~~~~
FastAPI server exposing HypRAG over HTTP.

Endpoints
---------
    GET  /health
    POST /index   { codebase_zip_b64 | text_documents }  -> { index_id, n_chunks }
    POST /search  { index_id, query, k, expand_subtree } -> { results }

Auth
----
Every request (except /health) requires the ``X-API-Key`` header. Keys live
in an in-memory store seeded from environment variables on startup. This is
deliberately simple — production should swap ``UserStore`` for a database.

Tiering
-------
    free : 100k vectors total, 100 queries/day, indexes purged after 7 days
    paid : 10M vectors, unlimited queries, persistent

Limits are enforced inside the handlers; the Stripe webhook (stubbed) is the
hook that flips a user's tier when payment status changes.
"""

from __future__ import annotations

import base64
import io
import tarfile
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    status,
)
from pydantic import BaseModel, Field

from hyprag import HypragRetriever
from hyprag.chunker import HierarchicalChunker
from hyprag.hybrid import HybridRetriever

from api.auth import APIKeyAuth, TIER_LIMITS
from api.store import UserStore, IndexStore


# ---------------------------------------------------------------------------
# Lifespan: model loading, store init
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load encoder once at startup; keep indexes in memory."""
    app.state.user_store = UserStore.from_env()
    app.state.index_store = IndexStore()
    # The encoder is loaded lazily on first /index call — startup stays fast.
    app.state.encoder_model = "all-MiniLM-L6-v2"
    yield
    # On shutdown, evict in-memory indexes (no-op for free tier; paid tier
    # should persist to disk — see roadmap).
    app.state.index_store.clear()


app = FastAPI(
    title="HypRAG",
    version="0.2.0",
    description="Hyperbolic retrieval API. Drop-in FAISS replacement that respects hierarchy.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class IndexFromArchive(BaseModel):
    """Upload a codebase as a base64-encoded .tar.gz or .zip."""
    archive_b64: str = Field(..., description="base64-encoded archive bytes")
    archive_format: str = Field("zip", pattern="^(zip|tar\\.gz)$")


class IndexFromTexts(BaseModel):
    """Index raw text documents with explicit hierarchy."""
    documents: list[dict] = Field(
        ...,
        description="List of {text, node_path, depth} dicts. "
                    "node_path uses dots, e.g. 'docs.guide.intro'.",
    )


class SearchRequest(BaseModel):
    index_id: str
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(10, ge=1, le=50)
    expand_subtree: bool = True
    use_hybrid: bool = Field(True, description="Merge BM25 lexical + semantic via RRF (recommended)")


class ChunkResponse(BaseModel):
    node_path: str
    depth: int
    text: str
    source_file: str | None = None
    score_rank: int


class SearchResponse(BaseModel):
    index_id: str
    n_results: int
    results: list[ChunkResponse]
    elapsed_ms: float


class IndexResponse(BaseModel):
    index_id: str
    n_chunks: int
    n_files: int
    elapsed_ms: float
    expires_at: float | None = None


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def get_auth(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> APIKeyAuth:
    user = app.state.user_store.lookup(x_api_key)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return APIKeyAuth(user_id=user.user_id, tier=user.tier, raw_key=x_api_key)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "hyprag",
        "version": app.version,
        "n_indexes": app.state.index_store.size(),
    }


# ---------------------------------------------------------------------------
# /index  (two modes: codebase archive OR raw texts)
# ---------------------------------------------------------------------------

@app.post("/index/codebase", response_model=IndexResponse)
def index_codebase(
    body: IndexFromArchive,
    auth: APIKeyAuth = Depends(get_auth),
) -> IndexResponse:
    """Chunk-and-index a Python codebase uploaded as a base64 archive."""
    t0 = time.perf_counter()
    limits = TIER_LIMITS[auth.tier]

    # Decode + extract to a tempdir
    try:
        raw = base64.b64decode(body.archive_b64)
    except Exception:
        raise HTTPException(400, "archive_b64 is not valid base64")

    if len(raw) > limits.max_archive_bytes:
        raise HTTPException(
            413, f"Archive exceeds tier limit ({limits.max_archive_bytes // 1_000_000} MB)"
        )

    import tempfile
    workdir = Path(tempfile.mkdtemp(prefix="hyprag_"))
    try:
        _extract_archive(raw, body.archive_format, workdir)
    except Exception as exc:
        raise HTTPException(400, f"Failed to extract archive: {exc}")

    # Build retriever (this lazily loads the encoder)
    retriever = _get_or_create_retriever(app, auth)
    n_added = retriever.index_path(workdir)

    if retriever.ntotal > limits.max_vectors:
        # Roll back: too many vectors for tier
        app.state.index_store.delete(_index_id_for(auth))
        raise HTTPException(
            413,
            f"Indexed corpus ({retriever.ntotal:,} chunks) exceeds tier limit "
            f"({limits.max_vectors:,}). Upgrade to paid tier for 10M vectors."
        )

    index_id = _index_id_for(auth)
    app.state.index_store.put(
        index_id, retriever, ttl_seconds=limits.ttl_seconds
    )

    n_files = len({c.source_file for c in retriever.chunks})
    return IndexResponse(
        index_id=index_id,
        n_chunks=n_added,
        n_files=n_files,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        expires_at=time.time() + limits.ttl_seconds if limits.ttl_seconds else None,
    )


@app.post("/index/texts", response_model=IndexResponse)
def index_texts(
    body: IndexFromTexts,
    auth: APIKeyAuth = Depends(get_auth),
) -> IndexResponse:
    """Index pre-chunked text documents with explicit hierarchy metadata."""
    t0 = time.perf_counter()
    limits = TIER_LIMITS[auth.tier]

    if len(body.documents) > limits.max_vectors:
        raise HTTPException(
            413,
            f"{len(body.documents)} documents exceeds tier limit "
            f"({limits.max_vectors:,})."
        )

    retriever = _get_or_create_retriever(app, auth)

    # Manually build chunks instead of going through the chunker
    from hyprag.chunker import Chunk
    chunks: list[Chunk] = []
    for i, doc in enumerate(body.documents):
        chunks.append(Chunk(
            id=retriever.ntotal + i,
            text=doc["text"],
            depth=int(doc.get("depth", 0)),
            node_path=doc["node_path"],
            source_file=doc.get("source_file", "<inline>"),
            start_line=1,
            end_line=1,
        ))

    texts = [c.text for c in chunks]
    depths = [c.depth for c in chunks]
    vecs = retriever._encoder.encode(  # type: ignore[attr-defined]
        texts, show_progress_bar=False, convert_to_numpy=True
    )
    retriever._index.add(vecs, depths=depths)  # type: ignore[attr-defined]
    retriever._chunks.extend(chunks)  # type: ignore[attr-defined]

    index_id = _index_id_for(auth)
    app.state.index_store.put(
        index_id, retriever, ttl_seconds=limits.ttl_seconds
    )

    return IndexResponse(
        index_id=index_id,
        n_chunks=len(chunks),
        n_files=1,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        expires_at=time.time() + limits.ttl_seconds if limits.ttl_seconds else None,
    )


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    auth: APIKeyAuth = Depends(get_auth),
) -> SearchResponse:
    limits = TIER_LIMITS[auth.tier]

    # Daily query budget
    used = app.state.user_store.consume_query(auth.user_id, limits.daily_queries)
    if used is None:
        raise HTTPException(
            429,
            f"Daily query limit reached ({limits.daily_queries}). "
            f"Resets at UTC midnight; upgrade for unlimited queries."
        )

    retriever = app.state.index_store.get(body.index_id)
    if retriever is None:
        raise HTTPException(404, "index_id not found (or expired)")

    # Tenancy check — a user can only query their own indexes
    if not body.index_id.startswith(auth.user_id):
        raise HTTPException(403, "Forbidden: index belongs to another user")

    t0 = time.perf_counter()
    query_kwargs = dict(k=body.k, expand_subtree=body.expand_subtree)
    if isinstance(retriever, HybridRetriever):
        query_kwargs["use_hybrid"] = body.use_hybrid
    chunks = retriever.query(body.query, **query_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return SearchResponse(
        index_id=body.index_id,
        n_results=len(chunks),
        elapsed_ms=elapsed_ms,
        results=[
            ChunkResponse(
                node_path=c.node_path,
                depth=c.depth,
                text=c.text[:1000],  # truncate to keep payload sane
                source_file=c.source_file,
                score_rank=i,
            )
            for i, c in enumerate(chunks)
        ],
    )


# ---------------------------------------------------------------------------
# Stripe webhook  (STUBBED — replace signature verification with real code)
# ---------------------------------------------------------------------------

@app.post("/_internal/stripe-webhook")
def stripe_webhook(payload: dict) -> dict:
    """
    Stub: flip a user's tier when their subscription status changes.

    Production must:
      1. Read the raw request body (not a parsed dict) for HMAC verification.
      2. Verify ``stripe-signature`` header with the webhook secret.
      3. Handle ``customer.subscription.updated`` / ``deleted`` events.
      4. Idempotency by event ID.

    The point of stubbing this is to keep the auth model honest while leaving
    the actual billing integration for when there's a paying user to bill.
    """
    event_type = payload.get("type")
    customer_email = (
        payload.get("data", {}).get("object", {}).get("customer_email")
    )
    if not event_type or not customer_email:
        raise HTTPException(400, "Malformed webhook payload")

    if event_type == "customer.subscription.created":
        app.state.user_store.set_tier_by_email(customer_email, "paid")
    elif event_type in ("customer.subscription.deleted", "customer.subscription.paused"):
        app.state.user_store.set_tier_by_email(customer_email, "free")

    return {"received": True, "event": event_type}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _index_id_for(auth: APIKeyAuth) -> str:
    """One index per user — keep this simple until multi-index is needed."""
    return f"{auth.user_id}_main"


def _get_or_create_retriever(app: FastAPI, auth: APIKeyAuth) -> HybridRetriever:
    index_id = _index_id_for(auth)
    existing = app.state.index_store.get(index_id)
    if existing is not None:
        return existing
    return HybridRetriever(encoder_model=app.state.encoder_model)


def _extract_archive(raw: bytes, fmt: str, dest: Path) -> None:
    buf = io.BytesIO(raw)
    if fmt == "zip":
        with zipfile.ZipFile(buf) as zf:
            _safe_extract_zip(zf, dest)
    elif fmt == "tar.gz":
        with tarfile.open(fileobj=buf, mode="r:gz") as tf:
            _safe_extract_tar(tf, dest)


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    """Refuse path-traversal attempts (CVE-2007-4559 family)."""
    dest = dest.resolve()
    for name in zf.namelist():
        target = (dest / name).resolve()
        if not str(target).startswith(str(dest)):
            raise ValueError(f"Refusing path-traversal entry: {name}")
    zf.extractall(dest)


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tf.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise ValueError(f"Refusing path-traversal entry: {member.name}")
    tf.extractall(dest)
