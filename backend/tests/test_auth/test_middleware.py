"""Tests for authentication middleware."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from requirements_graphrag_api.auth.api_key import (
    APIKeyInfo,
    APIKeyTier,
    InMemoryAPIKeyStore,
    generate_api_key,
)
from requirements_graphrag_api.auth.middleware import (
    AuthMiddleware,
    get_current_client,
    get_current_request_id,
)


class TestAuthMiddlewareAnonymousAccess:
    """Tests for middleware with anonymous access allowed."""

    @pytest.fixture
    def app_anonymous_allowed(self) -> FastAPI:
        """Create app allowing anonymous access."""
        app = FastAPI()
        app.state.api_key_store = InMemoryAPIKeyStore()
        app.add_middleware(AuthMiddleware, require_auth=False)

        @app.get("/test")
        async def test_endpoint():
            client = get_current_client()
            request_id = get_current_request_id()
            return {
                "client_id": client.key_id if client else None,
                "request_id": request_id,
            }

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        return app

    @pytest.fixture
    def client_anonymous(self, app_anonymous_allowed: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app_anonymous_allowed)

    def test_allows_request_without_key(self, client_anonymous: TestClient):
        """Should allow requests without API key."""
        response = client_anonymous.get("/test")
        assert response.status_code == 200
        # Should be anonymous
        assert response.json()["client_id"] == "anonymous"

    def test_includes_request_id_header(self, client_anonymous: TestClient):
        """Should include X-Request-ID in response."""
        response = client_anonymous.get("/test")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 36  # UUID length

    def test_health_endpoint_skips_auth(self, client_anonymous: TestClient):
        """Should skip auth for health endpoint."""
        response = client_anonymous.get("/health")
        assert response.status_code == 200


class TestAuthMiddlewareRequiredAuth:
    """Tests for middleware with required authentication."""

    @pytest.fixture
    def app_auth_required(self) -> FastAPI:
        """Create app requiring authentication."""
        app = FastAPI()
        app.state.api_key_store = InMemoryAPIKeyStore()
        app.add_middleware(AuthMiddleware, require_auth=True)

        @app.get("/test")
        async def test_endpoint():
            client = get_current_client()
            return {
                "client_id": client.key_id if client else None,
                "tier": client.tier if client else None,
            }

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        return app

    @pytest.fixture
    def client_auth_required(self, app_auth_required: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app_auth_required)

    def test_rejects_request_without_key(self, client_auth_required: TestClient):
        """Should reject requests without API key."""
        response = client_auth_required.get("/test")
        assert response.status_code == 401
        assert response.json()["detail"]["error"] == "missing_api_key"

    def test_rejects_invalid_key_format(self, client_auth_required: TestClient):
        """Should reject keys with invalid format."""
        response = client_auth_required.get("/test", headers={"X-API-Key": "invalid_key"})
        assert response.status_code == 401
        assert response.json()["detail"]["error"] == "invalid_api_key_format"

    def test_rejects_unknown_key(self, client_auth_required: TestClient):
        """Should reject unknown keys."""
        unknown_key = generate_api_key()
        response = client_auth_required.get("/test", headers={"X-API-Key": unknown_key})
        assert response.status_code == 403
        assert response.json()["detail"]["error"] == "invalid_api_key"

    def test_health_endpoint_skips_auth(self, client_auth_required: TestClient):
        """Should skip auth for health endpoint even when required."""
        response = client_auth_required.get("/health")
        assert response.status_code == 200


class TestAuthMiddlewareWithValidKey:
    """Tests for middleware with valid API key."""

    @pytest.fixture
    async def app_with_valid_key(self) -> tuple[FastAPI, str]:
        """Create app with a valid API key in store."""
        app = FastAPI()
        store = InMemoryAPIKeyStore()

        # Create valid key
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
            scopes=("chat", "search", "feedback"),
            metadata={},
        )
        await store.create(raw_key, info)

        app.state.api_key_store = store
        app.add_middleware(AuthMiddleware, require_auth=True)

        @app.get("/test")
        async def test_endpoint():
            client = get_current_client()
            return {
                "client_id": client.key_id if client else None,
                "tier": client.tier if client else None,
                "organization": client.organization if client else None,
            }

        return app, raw_key

    @pytest.mark.asyncio
    async def test_allows_valid_key(self, app_with_valid_key: tuple[FastAPI, str]):
        """Should allow requests with valid API key."""
        app, raw_key = app_with_valid_key
        with TestClient(app) as client:
            response = client.get("/test", headers={"X-API-Key": raw_key})
            assert response.status_code == 200
            data = response.json()
            assert data["client_id"] == "valid-key-id"
            assert data["tier"] == "standard"
            assert data["organization"] == "Test Org"

    @pytest.mark.asyncio
    async def test_context_available_in_handler(self, app_with_valid_key: tuple[FastAPI, str]):
        """Should make client available via get_current_client()."""
        app, raw_key = app_with_valid_key
        with TestClient(app) as client:
            response = client.get("/test", headers={"X-API-Key": raw_key})
            assert response.status_code == 200
            # Client info should be available
            assert response.json()["client_id"] is not None


class TestAuthMiddlewareRevokedKey:
    """Tests for middleware with revoked API key."""

    @pytest.fixture
    async def app_with_revoked_key(self) -> tuple[FastAPI, str]:
        """Create app with a revoked API key."""
        app = FastAPI()
        store = InMemoryAPIKeyStore()

        # Create and revoke key
        raw_key = generate_api_key()
        info = APIKeyInfo(
            key_id="revoked-key-id",
            name="Revoked Key",
            tier=APIKeyTier.STANDARD,
            organization=None,
            created_at=datetime.now(UTC),
            expires_at=None,
            rate_limit="50/minute",
            is_active=True,
            scopes=("chat",),
            metadata={},
        )
        await store.create(raw_key, info)
        await store.revoke("revoked-key-id")

        app.state.api_key_store = store
        app.add_middleware(AuthMiddleware, require_auth=True)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        return app, raw_key

    @pytest.mark.asyncio
    async def test_rejects_revoked_key(self, app_with_revoked_key: tuple[FastAPI, str]):
        """Should reject revoked keys."""
        app, raw_key = app_with_revoked_key
        with TestClient(app) as client:
            response = client.get("/test", headers={"X-API-Key": raw_key})
            assert response.status_code == 403
            assert response.json()["detail"]["error"] == "api_key_revoked"


class TestContextVariables:
    """Tests for context variable functions."""

    def test_get_current_client_outside_request(self):
        """Should return None outside request context."""
        client = get_current_client()
        assert client is None

    def test_get_current_request_id_outside_request(self):
        """Should return None outside request context."""
        request_id = get_current_request_id()
        assert request_id is None


class TestSkipAuthPaths:
    """Tests for paths that skip authentication."""

    @pytest.fixture
    def app_with_skip_paths(self) -> FastAPI:
        """Create app with common paths."""
        app = FastAPI(docs_url="/docs", redoc_url="/redoc")
        app.state.api_key_store = InMemoryAPIKeyStore()
        app.add_middleware(AuthMiddleware, require_auth=True)

        @app.get("/")
        async def root():
            return {"status": "ok"}

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        @app.get("/protected")
        async def protected():
            return {"status": "protected"}

        return app

    @pytest.fixture
    def client_skip_paths(self, app_with_skip_paths: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app_with_skip_paths)

    def test_root_skips_auth(self, client_skip_paths: TestClient):
        """Root path should skip auth."""
        response = client_skip_paths.get("/")
        assert response.status_code == 200

    def test_health_skips_auth(self, client_skip_paths: TestClient):
        """Health path should skip auth."""
        response = client_skip_paths.get("/health")
        assert response.status_code == 200

    def test_docs_skips_auth(self, client_skip_paths: TestClient):
        """Docs path should skip auth."""
        response = client_skip_paths.get("/docs")
        # May redirect or return 200
        assert response.status_code in (200, 307)

    def test_protected_requires_auth(self, client_skip_paths: TestClient):
        """Protected path should require auth."""
        response = client_skip_paths.get("/protected")
        assert response.status_code == 401
