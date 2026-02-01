"""Tests for PostgreSQL API key store."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from requirements_graphrag_api.auth import (
    APIKeyInfo,
    APIKeyTier,
    generate_api_key,
    hash_api_key,
)
from requirements_graphrag_api.auth.postgres_store import PostgresAPIKeyStore


@pytest.fixture
def sample_key_info() -> APIKeyInfo:
    """Create sample API key info for testing."""
    return APIKeyInfo(
        key_id="key_test123",
        name="Test Key",
        tier=APIKeyTier.STANDARD,
        organization="Test Org",
        created_at=datetime.now(UTC),
        expires_at=None,
        rate_limit="50/minute",
        is_active=True,
        scopes=("chat", "search", "feedback"),
        metadata={"created_by": "test"},
    )


class MockAsyncContextManager:
    """Mock async context manager for connection acquisition."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_connection():
    """Create a mock asyncpg connection."""
    conn = AsyncMock()
    return conn


@pytest.fixture
def mock_pool(mock_connection):
    """Create a mock asyncpg pool that properly handles acquire()."""
    pool = MagicMock()
    pool.acquire.return_value = MockAsyncContextManager(mock_connection)
    pool.close = AsyncMock()
    return pool


class TestPostgresAPIKeyStoreInit:
    """Test initialization and lifecycle."""

    def test_init_stores_parameters(self):
        store = PostgresAPIKeyStore(
            "postgresql://user:pass@host:5432/db",
            min_pool_size=3,
            max_pool_size=15,
        )
        assert store.database_url == "postgresql://user:pass@host:5432/db"
        assert store.min_pool_size == 3
        assert store.max_pool_size == 15
        assert store._pool is None

    def test_init_default_pool_sizes(self):
        store = PostgresAPIKeyStore("postgresql://localhost/test")
        assert store.min_pool_size == 2
        assert store.max_pool_size == 10

    @pytest.mark.asyncio
    async def test_initialize_creates_pool_and_tables(self, mock_pool, mock_connection):
        with patch(
            "requirements_graphrag_api.auth.postgres_store.asyncpg.create_pool",
            new_callable=AsyncMock,
        ) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            store = PostgresAPIKeyStore("postgresql://localhost/test")
            await store.initialize()

            mock_create_pool.assert_called_once_with(
                "postgresql://localhost/test",
                min_size=2,
                max_size=10,
            )
            mock_connection.execute.assert_called_once()
            assert store._pool is not None

    @pytest.mark.asyncio
    async def test_close_closes_pool(self, mock_pool, mock_connection):
        with patch(
            "requirements_graphrag_api.auth.postgres_store.asyncpg.create_pool",
            new_callable=AsyncMock,
        ) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            store = PostgresAPIKeyStore("postgresql://localhost/test")
            await store.initialize()
            await store.close()

            mock_pool.close.assert_called_once()
            assert store._pool is None

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_pool, mock_connection):
        with patch(
            "requirements_graphrag_api.auth.postgres_store.asyncpg.create_pool",
            new_callable=AsyncMock,
        ) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            async with PostgresAPIKeyStore("postgresql://localhost/test") as store:
                assert store._pool is not None

            mock_pool.close.assert_called_once()


class TestPostgresAPIKeyStoreNotInitialized:
    """Test behavior when store is not initialized."""

    @pytest.mark.asyncio
    async def test_get_by_hash_raises_when_not_initialized(self):
        store = PostgresAPIKeyStore("postgresql://localhost/test")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.get_by_hash("somehash")

    @pytest.mark.asyncio
    async def test_create_raises_when_not_initialized(self, sample_key_info):
        store = PostgresAPIKeyStore("postgresql://localhost/test")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.create("rgapi_test", sample_key_info)

    @pytest.mark.asyncio
    async def test_revoke_raises_when_not_initialized(self):
        store = PostgresAPIKeyStore("postgresql://localhost/test")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.revoke("key_123")

    @pytest.mark.asyncio
    async def test_list_keys_raises_when_not_initialized(self):
        store = PostgresAPIKeyStore("postgresql://localhost/test")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.list_keys()


class TestPostgresAPIKeyStoreOperations:
    """Test CRUD operations."""

    @pytest.fixture
    def initialized_store(self, mock_pool, mock_connection):
        """Create an initialized store with mocked pool."""
        store = PostgresAPIKeyStore("postgresql://localhost/test")
        store._pool = mock_pool
        return store, mock_connection

    @pytest.mark.asyncio
    async def test_get_by_hash_returns_none_when_not_found(self, initialized_store):
        store, mock_conn = initialized_store
        mock_conn.fetchrow.return_value = None

        result = await store.get_by_hash("nonexistent_hash")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_hash_returns_key_info(self, initialized_store):
        store, mock_conn = initialized_store
        mock_conn.fetchrow.return_value = {
            "key_id": "key_test123",
            "name": "Test Key",
            "tier": "standard",
            "organization": "Test Org",
            "created_at": datetime.now(UTC),
            "expires_at": None,
            "rate_limit": "50/minute",
            "is_active": True,
            "scopes": ["chat", "search"],
            "metadata": {"foo": "bar"},
        }

        result = await store.get_by_hash("somehash")

        assert result is not None
        assert result.key_id == "key_test123"
        assert result.name == "Test Key"
        assert result.tier == APIKeyTier.STANDARD
        assert result.scopes == ("chat", "search")
        assert result.metadata == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_create_stores_key(self, initialized_store, sample_key_info):
        store, mock_conn = initialized_store
        raw_key = generate_api_key()

        await store.create(raw_key, sample_key_info)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert "INSERT INTO api_keys" in call_args[0]
        assert call_args[1] == hash_api_key(raw_key)
        assert call_args[2] == sample_key_info.key_id

    @pytest.mark.asyncio
    async def test_revoke_returns_true_when_found(self, initialized_store):
        store, mock_conn = initialized_store
        mock_conn.execute.return_value = "UPDATE 1"

        result = await store.revoke("key_123")

        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_returns_false_when_not_found(self, initialized_store):
        store, mock_conn = initialized_store
        mock_conn.execute.return_value = "UPDATE 0"

        result = await store.revoke("key_nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_keys_returns_all_active(self, initialized_store):
        store, mock_conn = initialized_store
        mock_conn.fetch.return_value = [
            {
                "key_id": "key_1",
                "name": "Key 1",
                "tier": "free",
                "organization": None,
                "created_at": datetime.now(UTC),
                "expires_at": None,
                "rate_limit": "10/minute",
                "is_active": True,
                "scopes": ["chat"],
                "metadata": {},
            },
            {
                "key_id": "key_2",
                "name": "Key 2",
                "tier": "premium",
                "organization": "Acme",
                "created_at": datetime.now(UTC),
                "expires_at": None,
                "rate_limit": "200/minute",
                "is_active": True,
                "scopes": ["chat", "search", "feedback"],
                "metadata": {},
            },
        ]

        result = await store.list_keys()

        assert len(result) == 2
        assert result[0].key_id == "key_1"
        assert result[1].key_id == "key_2"

    @pytest.mark.asyncio
    async def test_list_keys_filters_by_organization(self, initialized_store):
        store, mock_conn = initialized_store
        mock_conn.fetch.return_value = []

        await store.list_keys(organization="Acme")

        call_args = mock_conn.fetch.call_args[0]
        assert "organization = $1" in call_args[0]
        assert call_args[1] == "Acme"

    @pytest.mark.asyncio
    async def test_delete_returns_true_when_found(self, initialized_store):
        store, mock_conn = initialized_store
        mock_conn.execute.return_value = "DELETE 1"

        result = await store.delete("key_123")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_not_found(self, initialized_store):
        store, mock_conn = initialized_store
        mock_conn.execute.return_value = "DELETE 0"

        result = await store.delete("key_nonexistent")

        assert result is False


class TestPostgresAPIKeyStoreMetadataHandling:
    """Test metadata and edge cases."""

    @pytest.fixture
    def initialized_store(self, mock_pool, mock_connection):
        """Create an initialized store with mocked pool."""
        store = PostgresAPIKeyStore("postgresql://localhost/test")
        store._pool = mock_pool
        return store, mock_connection

    @pytest.mark.asyncio
    async def test_handles_null_metadata(self, initialized_store):
        store, mock_conn = initialized_store
        mock_conn.fetchrow.return_value = {
            "key_id": "key_test",
            "name": "Test",
            "tier": "free",
            "organization": None,
            "created_at": datetime.now(UTC),
            "expires_at": None,
            "rate_limit": "10/minute",
            "is_active": True,
            "scopes": ["chat"],
            "metadata": None,  # NULL in database
        }

        result = await store.get_by_hash("somehash")

        assert result is not None
        assert result.metadata == {}

    @pytest.mark.asyncio
    async def test_handles_empty_scopes(self, initialized_store):
        store, mock_conn = initialized_store
        mock_conn.fetchrow.return_value = {
            "key_id": "key_test",
            "name": "Test",
            "tier": "free",
            "organization": None,
            "created_at": datetime.now(UTC),
            "expires_at": None,
            "rate_limit": "10/minute",
            "is_active": True,
            "scopes": [],
            "metadata": {},
        }

        result = await store.get_by_hash("somehash")

        assert result is not None
        assert result.scopes == ()
