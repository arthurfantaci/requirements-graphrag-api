"""API key generation, validation, and management.

This module provides secure API key handling for the GraphRAG API:
- Key generation with cryptographically secure random bytes
- SHA-256 hashing for secure storage (keys are never stored in plaintext)
- Format validation and tier-based access control
- Abstract store interface with in-memory implementation for dev/testing

Security considerations:
- Keys use 32 bytes of entropy (256 bits) for cryptographic strength
- Hash comparison uses constant-time operations to prevent timing attacks
- Keys are prefixed with 'rgapi_' for easy identification

Usage:
    # Generate a new key
    raw_key = generate_api_key()  # rgapi_<random>
    key_hash = hash_api_key(raw_key)  # Store this, not raw_key

    # Validate incoming key
    if validate_api_key_format(incoming_key):
        info = await store.get_by_hash(hash_api_key(incoming_key))
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

# API key header configuration
API_KEY_HEADER_NAME = "X-API-Key"
API_KEY_PREFIX = "rgapi_"  # Requirements GraphRAG API

# Security header for FastAPI dependency injection
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


class APIKeyTier(StrEnum):
    """API key access tiers with different rate limits.

    Tiers define the rate limits and capabilities available to API consumers:
    - FREE: Basic access for evaluation and small projects
    - STANDARD: Production access for typical workloads
    - PREMIUM: High-volume access with priority support
    - ENTERPRISE: Custom limits and SLA guarantees
    """

    FREE = "free"  # 10 requests/min
    STANDARD = "standard"  # 50 requests/min
    PREMIUM = "premium"  # 200 requests/min
    ENTERPRISE = "enterprise"  # Custom limits


# Default rate limits per tier (requests per minute)
TIER_RATE_LIMITS: dict[APIKeyTier, str] = {
    APIKeyTier.FREE: "10/minute",
    APIKeyTier.STANDARD: "50/minute",
    APIKeyTier.PREMIUM: "200/minute",
    APIKeyTier.ENTERPRISE: "1000/minute",
}


@dataclass(frozen=True, slots=True)
class APIKeyInfo:
    """Information associated with an API key.

    This dataclass is immutable (frozen) to ensure thread-safety when
    the same key info is accessed from multiple requests.

    Attributes:
        key_id: Unique identifier for the key (not the key itself).
        name: Human-readable name for dashboard display.
        tier: Access tier determining rate limits and capabilities.
        organization: Optional organization name for enterprise keys.
        created_at: When the key was created (UTC).
        expires_at: Optional expiration date (UTC), None means no expiry.
        rate_limit: Rate limit string (e.g., "50/minute").
        is_active: Whether the key is active (can be revoked).
        scopes: Tuple of authorized scopes (e.g., ("chat", "search")).
        metadata: Additional metadata for custom use cases.
    """

    key_id: str
    name: str
    tier: APIKeyTier
    organization: str | None
    created_at: datetime
    expires_at: datetime | None
    rate_limit: str
    is_active: bool
    scopes: tuple[str, ...]
    metadata: dict = field(default_factory=dict)


def generate_api_key() -> str:
    """Generate a new API key.

    Creates a cryptographically secure API key using Python's secrets module.
    The key format is: rgapi_<32 random bytes as base64url>

    Returns:
        A new API key string (~50 characters total).

    Example:
        >>> key = generate_api_key()
        >>> key.startswith("rgapi_")
        True
        >>> len(key) > 40
        True
    """
    random_bytes = secrets.token_urlsafe(32)
    return f"{API_KEY_PREFIX}{random_bytes}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage.

    Uses SHA-256 for fast lookups while maintaining security.
    The prefix is stripped before hashing to normalize keys.

    Args:
        api_key: The raw API key to hash.

    Returns:
        Hexadecimal SHA-256 hash of the key (64 characters).

    Note:
        The hash is deterministic - the same key always produces
        the same hash, allowing for lookup in the database.
    """
    # Strip prefix if present for consistent hashing
    key_value = api_key.removeprefix(API_KEY_PREFIX)
    return hashlib.sha256(key_value.encode()).hexdigest()


def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format without checking database.

    Performs fast local validation to reject obviously invalid keys
    before hitting the database.

    Args:
        api_key: The API key to validate.

    Returns:
        True if the format is valid, False otherwise.
    """
    if not api_key:
        return False
    if not api_key.startswith(API_KEY_PREFIX):
        return False
    if len(api_key) < 40:  # Minimum reasonable length (prefix + some chars)
        return False
    # Ensure only valid base64url characters after prefix
    key_part = api_key.removeprefix(API_KEY_PREFIX)
    return all(c.isalnum() or c in "-_" for c in key_part)


def compare_hashes_constant_time(hash1: str, hash2: str) -> bool:
    """Compare two hashes using constant-time comparison.

    Prevents timing attacks by ensuring the comparison takes
    the same amount of time regardless of where differences occur.

    Args:
        hash1: First hash to compare.
        hash2: Second hash to compare.

    Returns:
        True if hashes are equal, False otherwise.
    """
    return hmac.compare_digest(hash1.encode(), hash2.encode())


class APIKeyStore:
    """Abstract base for API key storage.

    This interface defines the contract for API key storage backends.
    In production, implement with a database backend (e.g., PostgreSQL, Neo4j).

    Subclasses must implement:
    - get_by_hash: Retrieve key info by its hash
    - create: Store a new API key
    - revoke: Revoke an existing key
    """

    async def get_by_hash(self, key_hash: str) -> APIKeyInfo | None:
        """Retrieve API key info by hash.

        Args:
            key_hash: SHA-256 hash of the API key.

        Returns:
            APIKeyInfo if found, None otherwise.
        """
        raise NotImplementedError

    async def create(self, api_key: str, info: APIKeyInfo) -> None:
        """Store a new API key.

        Args:
            api_key: The raw API key (will be hashed before storage).
            info: The APIKeyInfo to associate with the key.
        """
        raise NotImplementedError

    async def revoke(self, key_id: str) -> bool:
        """Revoke an API key by its ID.

        Args:
            key_id: The unique identifier of the key to revoke.

        Returns:
            True if the key was found and revoked, False otherwise.
        """
        raise NotImplementedError

    async def list_keys(self, organization: str | None = None) -> list[APIKeyInfo]:
        """List all API keys, optionally filtered by organization.

        Args:
            organization: Optional organization to filter by.

        Returns:
            List of APIKeyInfo objects.
        """
        raise NotImplementedError


class InMemoryAPIKeyStore(APIKeyStore):
    """In-memory API key store for development and testing.

    This implementation stores keys in a dictionary, suitable for:
    - Local development
    - Unit and integration testing
    - Single-instance deployments without persistence

    Warning:
        Keys are lost when the process restarts. Use a database-backed
        store for production deployments.
    """

    def __init__(self) -> None:
        """Initialize an empty key store."""
        self._keys: dict[str, APIKeyInfo] = {}

    async def get_by_hash(self, key_hash: str) -> APIKeyInfo | None:
        """Retrieve API key info by hash."""
        return self._keys.get(key_hash)

    async def create(self, api_key: str, info: APIKeyInfo) -> None:
        """Store a new API key."""
        key_hash = hash_api_key(api_key)
        self._keys[key_hash] = info

    async def revoke(self, key_id: str) -> bool:
        """Revoke an API key by marking it as inactive."""
        for key_hash, info in list(self._keys.items()):
            if info.key_id == key_id:
                # Create new info with is_active=False (immutable dataclass)
                revoked_info = APIKeyInfo(
                    key_id=info.key_id,
                    name=info.name,
                    tier=info.tier,
                    organization=info.organization,
                    created_at=info.created_at,
                    expires_at=info.expires_at,
                    rate_limit=info.rate_limit,
                    is_active=False,
                    scopes=info.scopes,
                    metadata=info.metadata,
                )
                self._keys[key_hash] = revoked_info
                return True
        return False

    async def list_keys(self, organization: str | None = None) -> list[APIKeyInfo]:
        """List all API keys, optionally filtered by organization."""
        if organization is None:
            return list(self._keys.values())
        return [info for info in self._keys.values() if info.organization == organization]


def create_anonymous_key_info() -> APIKeyInfo:
    """Create an APIKeyInfo for anonymous (unauthenticated) access.

    Used when authentication is disabled or for public endpoints.

    Returns:
        APIKeyInfo with limited anonymous access.
    """
    return APIKeyInfo(
        key_id="anonymous",
        name="Anonymous",
        tier=APIKeyTier.FREE,
        organization=None,
        created_at=datetime.now(UTC),
        expires_at=None,
        rate_limit="20/minute",
        is_active=True,
        scopes=("chat", "search"),
        metadata={},
    )


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
    *,
    key_store: APIKeyStore | None = None,
    require_auth: bool = False,
) -> APIKeyInfo:
    """Verify API key and return associated info.

    This function serves as a FastAPI dependency for authentication.
    It validates the API key format, checks it against the store,
    and verifies it's active and not expired.

    Args:
        api_key: The API key from the request header (injected by FastAPI).
        key_store: The API key store to check against.
        require_auth: Whether authentication is required.

    Returns:
        APIKeyInfo for the authenticated client.

    Raises:
        HTTPException: 401 if key is missing/invalid, 403 if revoked/expired.
    """
    # If auth not required and no key provided, return anonymous
    if not require_auth and not api_key:
        return create_anonymous_key_info()

    # If auth not required but key provided, validate it
    if key_store is None:
        # No store configured, return anonymous
        return create_anonymous_key_info()

    # Validate key presence when auth is required
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "missing_api_key",
                "message": f"API key required. Include '{API_KEY_HEADER_NAME}' header.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate key format
    if not validate_api_key_format(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_api_key_format",
                "message": "Invalid API key format. Keys must start with 'rgapi_'.",
            },
        )

    # Look up key in store
    key_hash = hash_api_key(api_key)
    key_info = await key_store.get_by_hash(key_hash)

    if key_info is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "invalid_api_key",
                "message": "API key not found or invalid.",
            },
        )

    # Check if active
    if not key_info.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "api_key_revoked",
                "message": "API key has been revoked.",
            },
        )

    # Check expiration
    if key_info.expires_at and datetime.now(UTC) > key_info.expires_at:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "api_key_expired",
                "message": "API key has expired.",
            },
        )

    return key_info
