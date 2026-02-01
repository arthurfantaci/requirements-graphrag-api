"""Tests for request size limit middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from starlette.testclient import TestClient

from requirements_graphrag_api.middleware.size_limit import (
    MAX_REQUEST_SIZE,
    SizeLimitMiddleware,
    check_request_size,
)


@pytest.fixture
def app_with_size_limit():
    """Create a test app with size limit middleware."""
    app = FastAPI()
    app.add_middleware(SizeLimitMiddleware)

    @app.post("/test")
    async def test_endpoint():
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


@pytest.fixture
def client(app_with_size_limit):
    """Create a test client."""
    return TestClient(app_with_size_limit)


class TestSizeLimitMiddleware:
    """Test the SizeLimitMiddleware class."""

    def test_allows_small_requests(self, client):
        response = client.post("/test", json={"data": "small"})
        assert response.status_code == 200

    def test_rejects_oversized_requests_via_header(self, client):
        # Test that oversized Content-Length header triggers rejection
        large_size = MAX_REQUEST_SIZE + 1000
        response = client.post(
            "/test",
            content=b"small",
            headers={"Content-Length": str(large_size)},
        )
        assert response.status_code == 413
        # HTTPException wraps the detail dict
        data = response.json()
        assert data["detail"]["error"] == "request_too_large"

    def test_excludes_health_endpoint(self, client):
        # Health endpoint should be excluded even with large content-length header
        response = client.get(
            "/health",
            headers={"Content-Length": str(MAX_REQUEST_SIZE * 2)},
        )
        assert response.status_code == 200

    def test_allows_requests_without_content_length(self, client):
        # Requests without Content-Length header should pass through
        response = client.post("/test")
        assert response.status_code == 200


class TestSizeLimitMiddlewareCustomLimits:
    """Test middleware with custom size limits."""

    def test_custom_max_request_size(self):
        app = FastAPI()
        app.add_middleware(SizeLimitMiddleware, max_request_size=100)

        @app.post("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # Small request should pass
        response = client.post("/test", json={"a": 1})
        assert response.status_code == 200

        # Request over 100 bytes should fail
        response = client.post(
            "/test",
            content=b"x" * 150,
            headers={"Content-Type": "text/plain", "Content-Length": "150"},
        )
        assert response.status_code == 413

    def test_custom_excluded_paths(self):
        app = FastAPI()
        app.add_middleware(
            SizeLimitMiddleware,
            max_request_size=100,
            excluded_paths=("/custom-exclude",),
        )

        @app.post("/custom-exclude")
        async def excluded():
            return {"status": "ok"}

        @app.post("/not-excluded")
        async def not_excluded():
            return {"status": "ok"}

        client = TestClient(app)

        # Excluded path should allow large requests
        response = client.post(
            "/custom-exclude",
            content=b"x" * 200,
            headers={"Content-Type": "text/plain", "Content-Length": "200"},
        )
        assert response.status_code == 200

        # Non-excluded path should reject
        response = client.post(
            "/not-excluded",
            content=b"x" * 200,
            headers={"Content-Type": "text/plain", "Content-Length": "200"},
        )
        assert response.status_code == 413


class TestCheckRequestSize:
    """Test the check_request_size utility function."""

    def test_passes_for_small_size(self):
        # Should not raise
        check_request_size(1000)

    def test_passes_for_none_size(self):
        # Should not raise
        check_request_size(None)

    def test_raises_for_large_size(self):
        with pytest.raises(HTTPException) as exc_info:
            check_request_size(MAX_REQUEST_SIZE + 1)
        assert exc_info.value.status_code == 413
        assert exc_info.value.detail["error"] == "request_too_large"

    def test_custom_max_size(self):
        with pytest.raises(HTTPException):
            check_request_size(200, max_size=100)


class TestErrorResponse:
    """Test error response format."""

    def test_error_response_format(self, client):
        large_content_length = str(MAX_REQUEST_SIZE + 1)
        response = client.post(
            "/test",
            content=b"small",
            headers={"Content-Length": large_content_length},
        )
        assert response.status_code == 413

        # HTTPException wraps response in "detail"
        data = response.json()
        assert "detail" in data
        detail = data["detail"]
        assert "error" in detail
        assert "message" in detail
        assert "max_size_bytes" in detail
        assert detail["max_size_bytes"] == MAX_REQUEST_SIZE
