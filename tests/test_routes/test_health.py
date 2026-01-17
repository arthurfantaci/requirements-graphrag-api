"""Tests for health check endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jama_mcp_server_graphrag.routes.health import router


@pytest.fixture
def mock_app() -> FastAPI:
    """Create a test FastAPI app with health router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_graph() -> MagicMock:
    """Create a mock Neo4jGraph."""
    graph = MagicMock()
    graph.query = MagicMock(return_value=[{"connected": 1}])
    return graph


@pytest.fixture
def client_with_graph(mock_app: FastAPI, mock_graph: MagicMock) -> TestClient:
    """Create a test client with mocked graph."""
    mock_app.state.graph = mock_graph
    return TestClient(mock_app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_healthy(self, client_with_graph: TestClient) -> None:
        """Test health check when Neo4j is connected."""
        response = client_with_graph.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["neo4j"] == "connected"
        assert "version" in data

    def test_health_check_degraded(self, mock_app: FastAPI) -> None:
        """Test health check when Neo4j connection fails."""
        # Create graph that raises exception on query
        mock_graph = MagicMock()
        mock_graph.query = MagicMock(side_effect=Exception("Connection failed"))
        mock_app.state.graph = mock_graph

        client = TestClient(mock_app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["neo4j"] == "disconnected"
