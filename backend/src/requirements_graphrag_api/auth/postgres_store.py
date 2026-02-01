"""PostgreSQL-backed API key storage for production use.

This module provides a PostgreSQL-backed implementation of the APIKeyStore
interface, suitable for production deployments where API keys must persist
across restarts.

Usage:
    store = PostgresAPIKeyStore(database_url)
    await store.initialize()  # Creates table if not exists

    # Or use as async context manager
    async with PostgresAPIKeyStore(database_url) as store:
        app.state.api_key_store = store
        ...
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import asyncpg

from requirements_graphrag_api.auth.api_key import (
    APIKeyInfo,
    APIKeyStore,
    APIKeyTier,
    hash_api_key,
)

if TYPE_CHECKING:
    from typing import Any

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

    Attributes:
        database_url: PostgreSQL connection URL.
        min_pool_size: Minimum connections in pool.
        max_pool_size: Maximum connections in pool.

    Example:
        store = PostgresAPIKeyStore("postgresql://user:pass@host:port/db")
        await store.initialize()

        # Create a key
        await store.create(raw_key, info)

        # Look up by hash
        info = await store.get_by_hash(hash_api_key(raw_key))
    """

    def __init__(
        self,
        database_url: str,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ) -> None:
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
        """Initialize connection pool and create tables.

        This must be called before using the store, unless using
        the async context manager which calls it automatically.

        Raises:
            asyncpg.PostgresError: If connection or table creation fails.
        """
        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=self.min_pool_size,
            max_size=self.max_pool_size,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)
        logger.info("PostgresAPIKeyStore initialized with connection pool")

    async def close(self) -> None:
        """Close the connection pool.

        Should be called during application shutdown to cleanly
        release database connections.
        """
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgresAPIKeyStore connection pool closed")

    async def __aenter__(self) -> PostgresAPIKeyStore:
        """Enter async context manager."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        await self.close()

    def _ensure_initialized(self) -> asyncpg.Pool:
        """Ensure the store is initialized and return the pool.

        Returns:
            The connection pool.

        Raises:
            RuntimeError: If store is not initialized.
        """
        if self._pool is None:
            msg = "PostgresAPIKeyStore not initialized. Call initialize() first."
            raise RuntimeError(msg)
        return self._pool

    async def get_by_hash(self, key_hash: str) -> APIKeyInfo | None:
        """Retrieve API key info by hash.

        Args:
            key_hash: SHA-256 hash of the API key.

        Returns:
            APIKeyInfo if found, None otherwise.
        """
        pool = self._ensure_initialized()

        async with pool.acquire() as conn:
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
            metadata=dict(row["metadata"]) if row["metadata"] else {},
        )

    async def create(self, api_key: str, info: APIKeyInfo) -> None:
        """Store a new API key.

        Args:
            api_key: The raw API key (will be hashed for storage).
            info: API key metadata.

        Raises:
            asyncpg.UniqueViolationError: If key already exists.
        """
        pool = self._ensure_initialized()
        key_hash = hash_api_key(api_key)

        async with pool.acquire() as conn:
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
                info.tier.value,
                info.organization,
                info.created_at,
                info.expires_at,
                info.rate_limit,
                info.is_active,
                list(info.scopes),
                json.dumps(info.metadata),  # Serialize dict to JSON string for JSONB
            )
        logger.info("Created API key: key_id=%s, tier=%s", info.key_id, info.tier)

    async def revoke(self, key_id: str) -> bool:
        """Revoke an API key by ID.

        Args:
            key_id: The key ID to revoke.

        Returns:
            True if key was found and revoked, False if not found.
        """
        pool = self._ensure_initialized()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE api_keys SET is_active = FALSE WHERE key_id = $1",
                key_id,
            )
            revoked = result == "UPDATE 1"

        if revoked:
            logger.info("Revoked API key: key_id=%s", key_id)
        return revoked

    async def list_keys(
        self,
        organization: str | None = None,
        include_inactive: bool = False,
    ) -> list[APIKeyInfo]:
        """List all API keys, optionally filtered.

        Args:
            organization: Filter by organization name.
            include_inactive: Include revoked keys in results.

        Returns:
            List of APIKeyInfo objects.
        """
        pool = self._ensure_initialized()

        # Build query based on filters
        conditions = []
        params: list[Any] = []

        if organization:
            conditions.append(f"organization = ${len(params) + 1}")
            params.append(organization)

        if not include_inactive:
            conditions.append("is_active = TRUE")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Note: conditions list contains only safe static strings, not user input
        query = f"""
            SELECT key_id, name, tier, organization, created_at,
                   expires_at, rate_limit, is_active, scopes, metadata
            FROM api_keys
            {where_clause}
            ORDER BY created_at DESC
        """  # noqa: S608 - conditions are static strings, not user input

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

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
                metadata=dict(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]

    async def delete(self, key_id: str) -> bool:
        """Permanently delete an API key.

        Use revoke() for soft deletion. This permanently removes the key.

        Args:
            key_id: The key ID to delete.

        Returns:
            True if key was found and deleted, False if not found.
        """
        pool = self._ensure_initialized()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM api_keys WHERE key_id = $1",
                key_id,
            )
            deleted = result == "DELETE 1"

        if deleted:
            logger.info("Deleted API key: key_id=%s", key_id)
        return deleted
