"""Tests for admin endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from requirements_graphrag_api.auth import (
    APIKeyInfo,
    APIKeyTier,
    AuthMiddleware,
    InMemoryAPIKeyStore,
    generate_api_key,
)
from requirements_graphrag_api.guardrails.metrics import metrics
from requirements_graphrag_api.routes.admin import router


def create_api_key_info(
    tier: APIKeyTier = APIKeyTier.ENTERPRISE,
    scopes: tuple[str, ...] = ("chat", "search", "feedback", "admin"),
    name: str = "Test Key",
) -> tuple[str, APIKeyInfo]:
    """Helper to create an API key and its info."""
    raw_key = generate_api_key()
    info = APIKeyInfo(
        key_id=f"key_{raw_key[:8]}",
        name=name,
        tier=tier,
        organization=None,
        created_at=datetime.now(UTC),
        expires_at=None,
        rate_limit="1000/minute",
        is_active=True,
        scopes=scopes,
    )
    return raw_key, info


@pytest.fixture
def mock_app() -> FastAPI:
    """Create a test FastAPI app with admin router."""
    app = FastAPI()

    # Set up auth middleware
    api_key_store = InMemoryAPIKeyStore()
    raw_key, info = create_api_key_info()

    # Store the key (need to run async method)
    # Use asyncio.run() instead of get_event_loop() for Python 3.10+ compatibility
    import asyncio

    asyncio.run(api_key_store.create(raw_key, info))

    app.state.api_key_store = api_key_store
    app.add_middleware(AuthMiddleware, require_auth=True)
    app.include_router(router)

    # Store raw key for tests to use
    app.state.test_api_key = raw_key

    return app


@pytest.fixture
def client(mock_app: FastAPI) -> TestClient:
    """Create a test client with auth."""
    return TestClient(mock_app)


@pytest.fixture
def auth_headers(mock_app: FastAPI) -> dict[str, str]:
    """Get auth headers with the test API key."""
    return {"X-API-Key": mock_app.state.test_api_key}


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics before and after each test."""
    metrics.reset()
    yield
    metrics.reset()


class TestGuardrailMetricsEndpoint:
    """Tests for GET /admin/guardrails/metrics."""

    def test_requires_auth(self, client: TestClient) -> None:
        """Test that endpoint requires authentication."""
        response = client.get("/admin/guardrails/metrics")
        assert response.status_code == 401

    def test_returns_metrics(self, client: TestClient, auth_headers: dict) -> None:
        """Test that endpoint returns current metrics."""
        # Record some metrics
        metrics.record_request()
        metrics.record_request(blocked=True)
        metrics.record_prompt_injection(blocked=True)

        response = client.get("/admin/guardrails/metrics", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "summary" in data
        assert "by_type" in data
        assert "performance" in data
        assert data["summary"]["total_requests"] == 2
        assert data["summary"]["requests_blocked"] == 1

    def test_returns_period_info(self, client: TestClient, auth_headers: dict) -> None:
        """Test that metrics include period information."""
        response = client.get("/admin/guardrails/metrics", headers=auth_headers)
        data = response.json()

        assert "period" in data
        assert "start" in data["period"]
        assert data["period"]["is_current"] is True


class TestGuardrailHistoryEndpoint:
    """Tests for GET /admin/guardrails/history."""

    def test_requires_auth(self, client: TestClient) -> None:
        response = client.get("/admin/guardrails/history")
        assert response.status_code == 401

    def test_returns_empty_history(self, client: TestClient, auth_headers: dict) -> None:
        response = client.get("/admin/guardrails/history", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "periods" in data
        assert "count" in data
        assert data["count"] == 0

    def test_returns_history_after_rotation(self, client: TestClient, auth_headers: dict) -> None:
        # Create some metrics and rotate
        metrics.record_request()
        metrics.rotate_period()
        metrics.record_request()
        metrics.record_request()
        metrics.rotate_period()

        response = client.get("/admin/guardrails/history", headers=auth_headers)
        data = response.json()

        assert data["count"] == 2
        assert len(data["periods"]) == 2

    def test_limit_parameter(self, client: TestClient, auth_headers: dict) -> None:
        # Create several periods
        for _ in range(5):
            metrics.record_request()
            metrics.rotate_period()

        response = client.get("/admin/guardrails/history?limit=2", headers=auth_headers)
        data = response.json()

        assert data["count"] == 2


class TestRotateMetricsEndpoint:
    """Tests for POST /admin/guardrails/rotate-metrics."""

    def test_requires_auth(self, client: TestClient) -> None:
        response = client.post("/admin/guardrails/rotate-metrics")
        assert response.status_code == 401

    def test_rotates_metrics(self, client: TestClient, auth_headers: dict) -> None:
        # Record some metrics
        metrics.record_request()
        metrics.record_request(blocked=True)

        response = client.post("/admin/guardrails/rotate-metrics", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "rotated"
        assert "completed_period" in data
        assert data["completed_period"]["total_requests"] == 2
        assert data["completed_period"]["requests_blocked"] == 1

        # Verify current metrics are reset
        current = metrics.get_current_metrics()
        assert current.total_requests == 0


class TestGuardrailConfigEndpoint:
    """Tests for GET /admin/guardrails/config."""

    def test_requires_auth(self, client: TestClient) -> None:
        response = client.get("/admin/guardrails/config")
        assert response.status_code == 401

    def test_returns_config(self, client: TestClient, auth_headers: dict) -> None:
        response = client.get("/admin/guardrails/config", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "features" in data
        assert "prompt_injection" in data["features"]
        assert "pii_detection" in data["features"]
        assert "rate_limiting" in data["features"]


class TestGuardrailSummaryEndpoint:
    """Tests for GET /admin/guardrails/summary."""

    def test_requires_auth(self, client: TestClient) -> None:
        response = client.get("/admin/guardrails/summary")
        assert response.status_code == 401

    def test_returns_summary(self, client: TestClient, auth_headers: dict) -> None:
        response = client.get("/admin/guardrails/summary", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "health_status" in data
        assert "performance_status" in data
        assert "totals" in data
        assert "block_rate" in data

    def test_health_status_good_for_low_blocks(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        # Record 100 requests with 2 blocked (<5%)
        for _ in range(100):
            metrics.record_request()
        for _ in range(2):
            metrics.record_request(blocked=True)

        response = client.get("/admin/guardrails/summary", headers=auth_headers)
        data = response.json()

        assert data["health_status"] == "good"

    def test_top_issues_sorted_by_count(self, client: TestClient, auth_headers: dict) -> None:
        # Record various issues
        for _ in range(10):
            metrics.record_prompt_injection(blocked=True)
        for _ in range(5):
            metrics.record_pii()
        for _ in range(3):
            metrics.record_toxicity()

        response = client.get("/admin/guardrails/summary", headers=auth_headers)
        data = response.json()

        issues = data["top_issues"]
        assert len(issues) > 0
        # Should be sorted by count descending
        counts = [i["count"] for i in issues]
        assert counts == sorted(counts, reverse=True)


class TestAdminScopeRequired:
    """Test that ADMIN scope is required for all endpoints."""

    @pytest.fixture
    def non_admin_client(self) -> TestClient:
        """Create a client with a non-admin API key."""
        import asyncio

        app = FastAPI()

        api_key_store = InMemoryAPIKeyStore()
        raw_key, info = create_api_key_info(
            tier=APIKeyTier.STANDARD,
            scopes=("chat", "search", "feedback"),  # No admin
            name="Non-Admin Key",
        )

        asyncio.run(api_key_store.create(raw_key, info))

        app.state.api_key_store = api_key_store
        app.add_middleware(AuthMiddleware, require_auth=True)
        app.include_router(router)
        app.state.test_api_key = raw_key

        return TestClient(app)

    def test_metrics_requires_admin(self, non_admin_client: TestClient) -> None:
        headers = {"X-API-Key": non_admin_client.app.state.test_api_key}
        response = non_admin_client.get("/admin/guardrails/metrics", headers=headers)
        assert response.status_code == 403
        assert "insufficient_scope" in response.json()["detail"]["error"]

    def test_history_requires_admin(self, non_admin_client: TestClient) -> None:
        headers = {"X-API-Key": non_admin_client.app.state.test_api_key}
        response = non_admin_client.get("/admin/guardrails/history", headers=headers)
        assert response.status_code == 403

    def test_rotate_requires_admin(self, non_admin_client: TestClient) -> None:
        headers = {"X-API-Key": non_admin_client.app.state.test_api_key}
        response = non_admin_client.post("/admin/guardrails/rotate-metrics", headers=headers)
        assert response.status_code == 403
