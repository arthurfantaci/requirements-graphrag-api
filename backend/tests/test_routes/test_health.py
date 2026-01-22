"""Tests for health check endpoint.

Updated Data Model (2026-01):
- Uses direct Neo4j driver instead of LangChain Neo4jGraph
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from requirements_graphrag_api.routes.health import router


@pytest.fixture
def mock_app() -> FastAPI:
    """Create a test FastAPI app with health router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_driver() -> MagicMock:
    """Create a mock Neo4j driver."""
    driver = MagicMock()
    mock_session = MagicMock()
    mock_session.run = MagicMock(return_value=MagicMock())
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    driver.session.return_value = mock_session
    return driver


@pytest.fixture
def client_with_driver(mock_app: FastAPI, mock_driver: MagicMock) -> TestClient:
    """Create a test client with mocked driver."""
    mock_app.state.driver = mock_driver
    return TestClient(mock_app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_healthy(self, client_with_driver: TestClient) -> None:
        """Test health check when Neo4j is connected."""
        response = client_with_driver.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["neo4j"] == "connected"
        assert "version" in data

    def test_health_check_degraded(self, mock_app: FastAPI) -> None:
        """Test health check when Neo4j connection fails."""
        # Create driver that raises exception on query
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.run = MagicMock(side_effect=Exception("Connection failed"))
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_app.state.driver = mock_driver

        client = TestClient(mock_app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["neo4j"] == "disconnected"

    def test_health_check_no_driver(self, mock_app: FastAPI) -> None:
        """Test health check when driver is not configured."""
        # Don't set driver on app state
        client = TestClient(mock_app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["neo4j"] == "not configured"
