# Production Dockerfile for Requirements GraphRAG API
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Builder stage - install dependencies
# ============================================================================
FROM base AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files (from repo root context)
COPY backend/pyproject.toml backend/uv.lock ./

# Copy README for build (required by pyproject.toml)
COPY backend/README.md ./

# Install dependencies (no dev dependencies for production)
RUN uv sync --frozen --no-dev --no-editable

# Download spaCy model for PII detection (Presidio requires this)
# Note: uv venvs don't include pip, so we use uv pip install directly
RUN uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl

# ============================================================================
# Production stage
# ============================================================================
FROM base AS production

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code (from repo root context)
COPY backend/src/ ./src/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose port for HTTP transport
EXPOSE 8000

# Health check (Railway uses its own healthcheck, this is for local Docker)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Default command - REST API mode for Railway deployment
# Use $PORT for Railway (defaults to 8000 for local development)
CMD ["sh", "-c", "python -m uvicorn requirements_graphrag_api.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
