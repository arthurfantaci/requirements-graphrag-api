# PostgreSQL API Key Storage Setup Guide

This document provides step-by-step instructions for setting up PostgreSQL on Railway to persist API keys in production, and enabling authentication for the GraphRAG API.

## Prerequisites

- Phase 3 access control merged (PR #96)
- Railway CLI installed (`brew install railway` or `npm i -g @railway/cli`)
- Access to your Railway project

---

## Part 1: Create Railway PostgreSQL Addon

### Step 1: Add PostgreSQL to Your Railway Project

**Option A: Via Railway Dashboard (Recommended)**

1. Go to [railway.app](https://railway.app) and open your project
2. Click **"+ New"** in the project canvas
3. Select **"Database"** → **"PostgreSQL"**
4. Railway will provision the database (takes ~30 seconds)
5. Click on the PostgreSQL service to see connection details

**Option B: Via Railway CLI**

```bash
# Login to Railway
railway login

# Link to your project (run from repo root)
railway link

# Add PostgreSQL plugin
railway add --plugin postgresql
```

### Step 2: Get Connection String

1. In Railway dashboard, click on the PostgreSQL service
2. Go to **"Connect"** tab
3. Copy the `DATABASE_URL` (format: `postgresql://user:pass@host:port/dbname`)

### Step 3: Add Environment Variable to Backend Service

1. Click on your backend service in Railway
2. Go to **"Variables"** tab
3. Add: `AUTH_DATABASE_URL` = `<your DATABASE_URL>`

---

## Part 2: Implement PostgresAPIKeyStore

Create the following file in your backend:

### `backend/src/requirements_graphrag_api/auth/postgres_store.py`

```python
"""PostgreSQL-backed API key storage for production use."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import asyncpg

from requirements_graphrag_api.auth.api_key import (
    APIKeyInfo,
    APIKeyStore,
    APIKeyTier,
    hash_api_key,
)

logger = logging.getLogger(__name__)

# SQL schema for API keys table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash VARCHAR(64) PRIMARY KEY,
    key_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    tier VARCHAR(20) NOT NULL DEFAULT 'free',
    organization VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    rate_limit VARCHAR(50) NOT NULL DEFAULT '10/minute',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    scopes TEXT[] NOT NULL DEFAULT ARRAY['chat', 'search'],
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys(key_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_organization ON api_keys(organization);
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active);
"""


class PostgresAPIKeyStore(APIKeyStore):
    """PostgreSQL-backed API key store for production.

    Uses asyncpg for high-performance async PostgreSQL operations.
    Connection pooling is handled automatically.

    Usage:
        store = PostgresAPIKeyStore(database_url)
        await store.initialize()  # Creates table if not exists

        # In app lifespan
        async with store:
            app.state.api_key_store = store
            yield
    """

    def __init__(self, database_url: str, min_pool_size: int = 2, max_pool_size: int = 10):
        """Initialize the store with connection parameters.

        Args:
            database_url: PostgreSQL connection URL.
            min_pool_size: Minimum connections in pool.
            max_pool_size: Maximum connections in pool.
        """
        self.database_url = database_url
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Initialize connection pool and create tables."""
        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=self.min_pool_size,
            max_size=self.max_pool_size,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)
        logger.info("PostgresAPIKeyStore initialized")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("PostgresAPIKeyStore connection pool closed")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def get_by_hash(self, key_hash: str) -> APIKeyInfo | None:
        """Retrieve API key info by hash."""
        if not self._pool:
            raise RuntimeError("Store not initialized")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT key_id, name, tier, organization, created_at,
                       expires_at, rate_limit, is_active, scopes, metadata
                FROM api_keys
                WHERE key_hash = $1
                """,
                key_hash,
            )

        if row is None:
            return None

        return APIKeyInfo(
            key_id=row["key_id"],
            name=row["name"],
            tier=APIKeyTier(row["tier"]),
            organization=row["organization"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            rate_limit=row["rate_limit"],
            is_active=row["is_active"],
            scopes=tuple(row["scopes"]),
            metadata=row["metadata"] or {},
        )

    async def create(self, api_key: str, info: APIKeyInfo) -> None:
        """Store a new API key."""
        if not self._pool:
            raise RuntimeError("Store not initialized")

        key_hash = hash_api_key(api_key)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO api_keys (
                    key_hash, key_id, name, tier, organization,
                    created_at, expires_at, rate_limit, is_active, scopes, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                key_hash,
                info.key_id,
                info.name,
                info.tier,
                info.organization,
                info.created_at,
                info.expires_at,
                info.rate_limit,
                info.is_active,
                list(info.scopes),
                info.metadata,
            )

    async def revoke(self, key_id: str) -> bool:
        """Revoke an API key by ID."""
        if not self._pool:
            raise RuntimeError("Store not initialized")

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE api_keys SET is_active = FALSE WHERE key_id = $1",
                key_id,
            )
            return result == "UPDATE 1"

    async def list_keys(self, organization: str | None = None) -> list[APIKeyInfo]:
        """List all API keys, optionally filtered by organization."""
        if not self._pool:
            raise RuntimeError("Store not initialized")

        async with self._pool.acquire() as conn:
            if organization:
                rows = await conn.fetch(
                    """
                    SELECT key_id, name, tier, organization, created_at,
                           expires_at, rate_limit, is_active, scopes, metadata
                    FROM api_keys
                    WHERE organization = $1
                    ORDER BY created_at DESC
                    """,
                    organization,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT key_id, name, tier, organization, created_at,
                           expires_at, rate_limit, is_active, scopes, metadata
                    FROM api_keys
                    ORDER BY created_at DESC
                    """
                )

        return [
            APIKeyInfo(
                key_id=row["key_id"],
                name=row["name"],
                tier=APIKeyTier(row["tier"]),
                organization=row["organization"],
                created_at=row["created_at"],
                expires_at=row["expires_at"],
                rate_limit=row["rate_limit"],
                is_active=row["is_active"],
                scopes=tuple(row["scopes"]),
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]
```

### Update `api.py` to Use PostgreSQL Store

Modify `backend/src/requirements_graphrag_api/api.py`:

```python
# In the lifespan function, replace:
#   api_key_store = InMemoryAPIKeyStore()
# With:

import os
from requirements_graphrag_api.auth import InMemoryAPIKeyStore

# Check for PostgreSQL URL
auth_db_url = os.getenv("AUTH_DATABASE_URL")

if auth_db_url:
    from requirements_graphrag_api.auth.postgres_store import PostgresAPIKeyStore
    api_key_store = PostgresAPIKeyStore(auth_db_url)
    await api_key_store.initialize()
    logger.info("Using PostgreSQL for API key storage")
else:
    api_key_store = InMemoryAPIKeyStore()
    logger.warning("Using in-memory API key store (not for production!)")
```

### Add `asyncpg` Dependency

```bash
cd backend
uv add asyncpg
```

---

## Part 3: Create Your First API Key

### Option A: Via Python Script

Create `scripts/create_api_key.py`:

```python
"""Script to create API keys for the GraphRAG API."""

import asyncio
import os
import sys
from datetime import UTC, datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "src"))

from requirements_graphrag_api.auth import (
    generate_api_key,
    APIKeyInfo,
    APIKeyTier,
)
from requirements_graphrag_api.auth.postgres_store import PostgresAPIKeyStore


async def create_key(
    name: str,
    tier: str = "standard",
    organization: str | None = None,
    scopes: tuple[str, ...] = ("chat", "search", "feedback"),
):
    """Create a new API key."""
    database_url = os.environ.get("AUTH_DATABASE_URL")
    if not database_url:
        print("ERROR: AUTH_DATABASE_URL environment variable not set")
        sys.exit(1)

    store = PostgresAPIKeyStore(database_url)
    await store.initialize()

    # Generate key
    raw_key = generate_api_key()
    key_id = f"key_{raw_key[6:14]}"  # Use part of key as ID

    info = APIKeyInfo(
        key_id=key_id,
        name=name,
        tier=APIKeyTier(tier),
        organization=organization,
        created_at=datetime.now(UTC),
        expires_at=None,
        rate_limit={"free": "10/minute", "standard": "50/minute", "premium": "200/minute", "enterprise": "1000/minute"}[tier],
        is_active=True,
        scopes=scopes,
        metadata={"created_by": "admin_script"},
    )

    await store.create(raw_key, info)
    await store.close()

    print(f"\n✅ API Key Created!")
    print(f"   Name: {name}")
    print(f"   Tier: {tier}")
    print(f"   Key ID: {key_id}")
    print(f"\n🔑 API Key (SAVE THIS - shown only once):")
    print(f"   {raw_key}")
    print(f"\n📋 Usage:")
    print(f'   curl -H "X-API-Key: {raw_key}" https://your-api.railway.app/chat')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create API key")
    parser.add_argument("--name", required=True, help="Key name")
    parser.add_argument("--tier", default="standard", choices=["free", "standard", "premium", "enterprise"])
    parser.add_argument("--org", default=None, help="Organization name")

    args = parser.parse_args()
    asyncio.run(create_key(args.name, args.tier, args.org))
```

Usage:
```bash
# Set the database URL from Railway
export AUTH_DATABASE_URL="postgresql://..."

# Create a key
python scripts/create_api_key.py --name "My App" --tier standard
```

---

## Part 4: Enable Authentication in Production

### Step 1: Set Environment Variable

In Railway dashboard, add to your backend service:

```
REQUIRE_API_KEY=true
```

### Step 2: Deploy

Railway will automatically redeploy with authentication enabled.

### Step 3: Test

```bash
# Without key - should get 401
curl https://your-api.railway.app/chat -X POST -d '{"query": "test"}'

# With key - should work
curl https://your-api.railway.app/chat \
  -X POST \
  -H "X-API-Key: rgapi_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is requirements traceability?"}'
```

---

## Rollback Plan

If issues occur after enabling auth:

1. **Quick disable**: Set `REQUIRE_API_KEY=false` in Railway variables
2. **Check logs**: `railway logs` to see auth errors
3. **Verify DB**: Connect to PostgreSQL and check `api_keys` table

---

## Cost Considerations

| Railway Resource | Cost |
|-----------------|------|
| PostgreSQL Starter | ~$5/month |
| Storage (1GB) | Included |
| Connections | Up to 20 |

The Starter plan is sufficient for API key storage (very low traffic table).

---

## Security Checklist

Before going live:

- [ ] PostgreSQL provisioned on Railway
- [ ] `AUTH_DATABASE_URL` set in backend service
- [ ] `asyncpg` added to dependencies
- [ ] `postgres_store.py` implemented and tested
- [ ] At least one API key created
- [ ] `REQUIRE_API_KEY=true` set
- [ ] Frontend updated to include `X-API-Key` header (if needed)
- [ ] Tested auth flow end-to-end

---

## Troubleshooting

### "Store not initialized" Error
- Ensure `await api_key_store.initialize()` is called in lifespan
- Check `AUTH_DATABASE_URL` is set correctly

### "Connection refused" Error
- Verify PostgreSQL is running in Railway
- Check the connection string format
- Ensure your service can reach the database (same Railway project)

### Keys Not Persisting
- Confirm you're using `PostgresAPIKeyStore`, not `InMemoryAPIKeyStore`
- Check logs for "Using PostgreSQL for API key storage" message
