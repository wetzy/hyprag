# syntax=docker/dockerfile:1.6
#
# HypRAG API container — multi-stage build.
#
# Stage 1 (builder): install deps + pre-download the sentence-transformers
# model so the running container doesn't make outbound HuggingFace calls
# at request time. This is important on Fly.io: cold starts must serve
# the first /search call within a few seconds.
#
# Stage 2 (runtime): slim Python image with only what's needed to serve.
#
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System deps for torch + faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer when only source changes)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip \
    && pip install . fastapi uvicorn[standard] psutil

# Pre-download the encoder model into a known cache dir
ENV HF_HOME=/build/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/build/.cache/sentence-transformers
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"


# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    PORT=8080

WORKDIR /app

# Runtime deps only (no compiler, no curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages + pre-downloaded model from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /build/.cache /app/.cache

# Copy source
COPY src/ ./src/
COPY api/ ./api/

# Non-root user
RUN useradd --create-home --shell /bin/bash hyprag \
    && chown -R hyprag:hyprag /app
USER hyprag

# Healthcheck — Fly.io uses this to detect cold-start readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; \
        urllib.request.urlopen('http://localhost:${PORT}/health', timeout=3)" \
        || exit 1

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
