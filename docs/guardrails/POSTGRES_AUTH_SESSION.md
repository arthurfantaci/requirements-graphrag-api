# Claude Code Session: PostgreSQL Auth Store Implementation

## Context

This session picks up from Issue #99 to implement PostgreSQL-backed API key storage for production deployment.

**Branch**: `feat/postgres-auth-store`
**Issue**: https://github.com/arthurfantaci/requirements-graphrag-api/issues/99
**Spec**: `docs/guardrails/POSTGRES_AUTH_SETUP.md`

## Prerequisites

- Phase 3 access control is merged (PR #96)
- Phase 4 guardrails is merged (PR #98)
- Feature branch `feat/postgres-auth-store` is created

## Implementation Tasks

### 1. Add asyncpg dependency
```bash
cd backend && uv add asyncpg
```

### 2. Create PostgresAPIKeyStore

**File**: `backend/src/requirements_graphrag_api/auth/postgres_store.py`

Reference implementation is provided in `docs/guardrails/POSTGRES_AUTH_SETUP.md` (Part 2).

Key requirements:
- Implements `APIKeyStore` abstract base class
- Uses asyncpg with connection pooling
- Auto-creates tables via `CREATE TABLE IF NOT EXISTS`
- Includes proper async context manager support
- Logs initialization and shutdown

### 3. Update api.py lifespan

Modify `backend/src/requirements_graphrag_api/api.py` to conditionally use PostgresAPIKeyStore:

```python
# After existing imports, add:
import os

# In lifespan function, replace InMemoryAPIKeyStore initialization with:
auth_db_url = os.getenv("AUTH_DATABASE_URL")
if auth_db_url:
    from requirements_graphrag_api.auth.postgres_store import PostgresAPIKeyStore
    api_key_store = PostgresAPIKeyStore(auth_db_url)
    await api_key_store.initialize()
    logger.info("Using PostgreSQL for API key storage")
else:
    api_key_store = InMemoryAPIKeyStore()
    logger.warning("Using in-memory API key store (not for production!)")

# Ensure cleanup in lifespan:
# Add to cleanup section:
if hasattr(api_key_store, 'close'):
    await api_key_store.close()
```

### 4. Update auth module exports

**File**: `backend/src/requirements_graphrag_api/auth/__init__.py`

Add:
```python
from requirements_graphrag_api.auth.postgres_store import PostgresAPIKeyStore
```

And update `__all__` to include `PostgresAPIKeyStore`.

### 5. Create key management script

**File**: `scripts/create_api_key.py`

Reference implementation in `docs/guardrails/POSTGRES_AUTH_SETUP.md` (Part 3).

### 6. Write tests

**File**: `backend/tests/test_auth/test_postgres_store.py`

Test cases:
- `test_initialize_creates_tables` - Verify table creation
- `test_create_and_get_key` - Full key lifecycle
- `test_revoke_key` - Key revocation works
- `test_list_keys` - List and filter keys
- `test_list_keys_by_organization` - Filter by org
- `test_get_nonexistent_key` - Returns None
- `test_not_initialized_raises` - Runtime error if not initialized
- `test_context_manager` - async with works properly

Use mocking or a test database approach consistent with existing tests.

## Validation Checklist

Before creating PR:
- [ ] `uv run ruff check src/` passes
- [ ] `uv run pytest tests/` passes (all 621+ tests)
- [ ] New tests for PostgresAPIKeyStore pass
- [ ] Script works with test database
- [ ] Documentation is clear

## PR Creation

When ready:
```bash
git add -A
git commit -m "feat(auth): implement PostgreSQL API key storage

Implements persistent API key storage for production:
- Add PostgresAPIKeyStore using asyncpg with connection pooling
- Conditional store selection based on AUTH_DATABASE_URL
- Key management script for creating API keys
- Comprehensive tests for new store

Closes #99

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

git push -u origin feat/postgres-auth-store
gh pr create --title "feat(auth): implement PostgreSQL API key storage" --body "..."
```

## Notes

- The `asyncpg` library is preferred over `psycopg` for async PostgreSQL
- Connection pooling is essential for production performance
- The InMemoryAPIKeyStore fallback ensures development works without a database
- Railway will automatically inject DATABASE_URL; we use AUTH_DATABASE_URL to be explicit
