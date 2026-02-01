"""Tests for API key generation, validation, and storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from requirements_graphrag_api.auth.api_key import (
    API_KEY_PREFIX,
    APIKeyInfo,
    APIKeyTier,
    InMemoryAPIKeyStore,
    compare_hashes_constant_time,
    create_anonymous_key_info,
    generate_api_key,
    hash_api_key,
    validate_api_key_format,
    verify_api_key,
)


class TestAPIKeyGeneration:
    """Tests for API key generation."""

    def test_generates_key_with_correct_prefix(self):
        """Generated keys should start with the rgapi_ prefix."""
        key = generate_api_key()
        assert key.startswith(API_KEY_PREFIX)

    def test_generates_key_with_minimum_length(self):
        """Generated keys should be at least 40 characters."""
        key = generate_api_key()
        assert len(key) >= 40

    def test_generates_unique_keys(self):
        """Each generated key should be unique."""
        keys = [generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100, "Generated keys should be unique"

    def test_key_contains_valid_characters(self):
        """Key should only contain alphanumeric chars and url-safe chars."""
        key = generate_api_key()
        key_part = key.removeprefix(API_KEY_PREFIX)
        assert all(c.isalnum() or c in "-_" for c in key_part)


class TestAPIKeyHashing:
    """Tests for API key hashing."""

    def test_hash_is_consistent(self):
        """Same key should always produce the same hash."""
        key = generate_api_key()
        hash1 = hash_api_key(key)
        hash2 = hash_api_key(key)
        assert hash1 == hash2

    def test_different_keys_different_hashes(self):
        """Different keys should produce different hashes."""
        key1 = generate_api_key()
        key2 = generate_api_key()
        assert hash_api_key(key1) != hash_api_key(key2)

    def test_hash_is_hex_string(self):
        """Hash should be a valid hexadecimal string."""
        key = generate_api_key()
        key_hash = hash_api_key(key)
        assert len(key_hash) == 64  # SHA-256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in key_hash)

    def test_hash_strips_prefix(self):
        """Hash should be the same with or without prefix."""
        key = generate_api_key()
        key_without_prefix = key.removeprefix(API_KEY_PREFIX)
        # Hash should be of the key part only
        hash_with_prefix = hash_api_key(key)
        hash_without_prefix = hash_api_key(key_without_prefix)
        assert hash_with_prefix == hash_without_prefix


class TestConstantTimeComparison:
    """Tests for timing-safe hash comparison."""

    def test_equal_hashes_return_true(self):
        """Equal hashes should compare as equal."""
        hash1 = hash_api_key(generate_api_key())
        assert compare_hashes_constant_time(hash1, hash1)

    def test_different_hashes_return_false(self):
        """Different hashes should compare as not equal."""
        hash1 = hash_api_key(generate_api_key())
        hash2 = hash_api_key(generate_api_key())
        assert not compare_hashes_constant_time(hash1, hash2)


class TestAPIKeyFormatValidation:
    """Tests for API key format validation."""

    def test_valid_key_passes(self):
        """Valid generated keys should pass validation."""
        key = generate_api_key()
        assert validate_api_key_format(key)

    def test_empty_string_fails(self):
        """Empty string should fail validation."""
        assert not validate_api_key_format("")

    def test_none_like_values_fail(self):
        """None-like values should fail validation."""
        assert not validate_api_key_format(None)  # type: ignore[arg-type]

    def test_missing_prefix_fails(self):
        """Keys without the correct prefix should fail."""
        key = generate_api_key()
        key_without_prefix = key.removeprefix(API_KEY_PREFIX)
        assert not validate_api_key_format(key_without_prefix)

    def test_wrong_prefix_fails(self):
        """Keys with wrong prefix should fail."""
        assert not validate_api_key_format("wrong_prefix_abc123")
        assert not validate_api_key_format("sk_live_abc123")  # Stripe-like key

    def test_too_short_key_fails(self):
        """Keys that are too short should fail."""
        assert not validate_api_key_format("rgapi_short")
        assert not validate_api_key_format(f"{API_KEY_PREFIX}abc")

    def test_invalid_characters_fail(self):
        """Keys with invalid characters should fail."""
        assert not validate_api_key_format(f"{API_KEY_PREFIX}abc!@#$%")
        assert not validate_api_key_format(f"{API_KEY_PREFIX}abc def")  # space


class TestAPIKeyInfo:
    """Tests for APIKeyInfo dataclass."""

    def test_is_immutable(self):
        """APIKeyInfo should be immutable (frozen)."""
        info = APIKeyInfo(
            key_id="test-id",
            name="Test Key",
            tier=APIKeyTier.FREE,
            organization=None,
            created_at=datetime.now(UTC),
            expires_at=None,
            rate_limit="10/minute",
            is_active=True,
            scopes=("chat", "search"),
            metadata={},
        )
        with pytest.raises(AttributeError):
            info.name = "New Name"  # type: ignore[misc]

    def test_default_metadata(self):
        """Metadata should default to empty dict."""
        info = APIKeyInfo(
            key_id="test-id",
            name="Test Key",
            tier=APIKeyTier.FREE,
            organization=None,
            created_at=datetime.now(UTC),
            expires_at=None,
            rate_limit="10/minute",
            is_active=True,
            scopes=("chat",),
        )
        assert info.metadata == {}


class TestAPIKeyTier:
    """Tests for APIKeyTier enum."""

    def test_tier_values(self):
        """Verify tier string values."""
        assert APIKeyTier.FREE == "free"
        assert APIKeyTier.STANDARD == "standard"
        assert APIKeyTier.PREMIUM == "premium"
        assert APIKeyTier.ENTERPRISE == "enterprise"

    def test_all_tiers_defined(self):
        """Ensure all expected tiers are defined."""
        tiers = list(APIKeyTier)
        assert len(tiers) == 4


class TestAnonymousKeyInfo:
    """Tests for anonymous key creation."""

    def test_creates_anonymous_info(self):
        """Should create info for anonymous access."""
        info = create_anonymous_key_info()
        assert info.key_id == "anonymous"
        assert info.name == "Anonymous"
        assert info.tier == APIKeyTier.FREE
        assert info.is_active is True

    def test_anonymous_has_limited_scopes(self):
        """Anonymous should have limited scopes."""
        info = create_anonymous_key_info()
        assert "chat" in info.scopes
        assert "search" in info.scopes
        assert "admin" not in info.scopes


class TestInMemoryAPIKeyStore:
    """Tests for the in-memory API key store."""

    @pytest.fixture
    def store(self) -> InMemoryAPIKeyStore:
        """Create a fresh store for each test."""
        return InMemoryAPIKeyStore()

    @pytest.fixture
    def sample_key_info(self) -> APIKeyInfo:
        """Create sample API key info."""
        return APIKeyInfo(
            key_id="test-key-id",
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

    @pytest.mark.asyncio
    async def test_create_and_retrieve(
        self, store: InMemoryAPIKeyStore, sample_key_info: APIKeyInfo
    ):
        """Should store and retrieve key info."""
        raw_key = generate_api_key()
        await store.create(raw_key, sample_key_info)

        key_hash = hash_api_key(raw_key)
        retrieved = await store.get_by_hash(key_hash)

        assert retrieved is not None
        assert retrieved.key_id == sample_key_info.key_id
        assert retrieved.name == sample_key_info.name
        assert retrieved.tier == sample_key_info.tier

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, store: InMemoryAPIKeyStore):
        """Should return None for nonexistent keys."""
        result = await store.get_by_hash("nonexistent_hash")
        assert result is None

    @pytest.mark.asyncio
    async def test_revoke_key(self, store: InMemoryAPIKeyStore, sample_key_info: APIKeyInfo):
        """Should revoke a key by ID."""
        raw_key = generate_api_key()
        await store.create(raw_key, sample_key_info)

        # Revoke
        result = await store.revoke(sample_key_info.key_id)
        assert result is True

        # Verify revoked
        key_hash = hash_api_key(raw_key)
        retrieved = await store.get_by_hash(key_hash)
        assert retrieved is not None
        assert retrieved.is_active is False

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_returns_false(self, store: InMemoryAPIKeyStore):
        """Should return False when revoking nonexistent key."""
        result = await store.revoke("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_all_keys(self, store: InMemoryAPIKeyStore, sample_key_info: APIKeyInfo):
        """Should list all stored keys."""
        # Create multiple keys
        for i in range(3):
            raw_key = generate_api_key()
            info = APIKeyInfo(
                key_id=f"key-{i}",
                name=f"Key {i}",
                tier=APIKeyTier.STANDARD,
                organization="Test Org",
                created_at=datetime.now(UTC),
                expires_at=None,
                rate_limit="50/minute",
                is_active=True,
                scopes=("chat",),
                metadata={},
            )
            await store.create(raw_key, info)

        keys = await store.list_keys()
        assert len(keys) == 3

    @pytest.mark.asyncio
    async def test_list_keys_by_organization(self, store: InMemoryAPIKeyStore):
        """Should filter keys by organization."""
        # Create keys for different orgs
        for org in ["Org A", "Org A", "Org B"]:
            raw_key = generate_api_key()
            info = APIKeyInfo(
                key_id=f"key-{raw_key[:8]}",
                name="Key",
                tier=APIKeyTier.STANDARD,
                organization=org,
                created_at=datetime.now(UTC),
                expires_at=None,
                rate_limit="50/minute",
                is_active=True,
                scopes=("chat",),
                metadata={},
            )
            await store.create(raw_key, info)

        org_a_keys = await store.list_keys(organization="Org A")
        org_b_keys = await store.list_keys(organization="Org B")

        assert len(org_a_keys) == 2
        assert len(org_b_keys) == 1


class TestVerifyAPIKey:
    """Tests for the verify_api_key function."""

    @pytest.fixture
    def store(self) -> InMemoryAPIKeyStore:
        """Create a fresh store for each test."""
        return InMemoryAPIKeyStore()

    @pytest.fixture
    async def valid_key_and_store(
        self, store: InMemoryAPIKeyStore
    ) -> tuple[str, InMemoryAPIKeyStore]:
        """Create a store with a valid key."""
        raw_key = generate_api_key()
        info = APIKeyInfo(
            key_id="valid-key-id",
            name="Valid Key",
            tier=APIKeyTier.STANDARD,
            organization="Test Org",
            created_at=datetime.now(UTC),
            expires_at=None,
            rate_limit="50/minute",
            is_active=True,
            scopes=("chat", "search"),
            metadata={},
        )
        await store.create(raw_key, info)
        return raw_key, store

    @pytest.mark.asyncio
    async def test_valid_key_returns_info(
        self, valid_key_and_store: tuple[str, InMemoryAPIKeyStore]
    ):
        """Valid key should return APIKeyInfo."""
        raw_key, store = valid_key_and_store
        info = await verify_api_key(raw_key, key_store=store, require_auth=True)
        assert info.key_id == "valid-key-id"
        assert info.tier == APIKeyTier.STANDARD

    @pytest.mark.asyncio
    async def test_no_key_when_auth_not_required_returns_anonymous(
        self, store: InMemoryAPIKeyStore
    ):
        """No key when auth not required should return anonymous."""
        info = await verify_api_key(None, key_store=store, require_auth=False)
        assert info.key_id == "anonymous"

    @pytest.mark.asyncio
    async def test_no_key_when_auth_required_raises_401(self, store: InMemoryAPIKeyStore):
        """No key when auth required should raise 401."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(None, key_store=store, require_auth=True)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "missing_api_key"

    @pytest.mark.asyncio
    async def test_invalid_format_when_auth_required_raises_401(self, store: InMemoryAPIKeyStore):
        """Invalid format when auth required should raise 401."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key("invalid_key", key_store=store, require_auth=True)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "invalid_api_key_format"

    @pytest.mark.asyncio
    async def test_unknown_key_raises_403(self, store: InMemoryAPIKeyStore):
        """Unknown key should raise 403."""
        from fastapi import HTTPException

        unknown_key = generate_api_key()
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(unknown_key, key_store=store, require_auth=True)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "invalid_api_key"

    @pytest.mark.asyncio
    async def test_revoked_key_raises_403(
        self, valid_key_and_store: tuple[str, InMemoryAPIKeyStore]
    ):
        """Revoked key should raise 403."""
        from fastapi import HTTPException

        raw_key, store = valid_key_and_store
        await store.revoke("valid-key-id")

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(raw_key, key_store=store, require_auth=True)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "api_key_revoked"

    @pytest.mark.asyncio
    async def test_expired_key_raises_403(self, store: InMemoryAPIKeyStore):
        """Expired key should raise 403."""
        from fastapi import HTTPException

        raw_key = generate_api_key()
        info = APIKeyInfo(
            key_id="expired-key-id",
            name="Expired Key",
            tier=APIKeyTier.STANDARD,
            organization=None,
            created_at=datetime.now(UTC) - timedelta(days=30),
            expires_at=datetime.now(UTC) - timedelta(days=1),  # Expired yesterday
            rate_limit="50/minute",
            is_active=True,
            scopes=("chat",),
            metadata={},
        )
        await store.create(raw_key, info)

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(raw_key, key_store=store, require_auth=True)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "api_key_expired"

    @pytest.mark.asyncio
    async def test_no_store_returns_anonymous(self):
        """No store configured should return anonymous."""
        info = await verify_api_key("any_key", key_store=None, require_auth=False)
        assert info.key_id == "anonymous"
