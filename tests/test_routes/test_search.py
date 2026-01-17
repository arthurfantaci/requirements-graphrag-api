"""Tests for search endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jama_mcp_server_graphrag.routes.search import router


@pytest.fixture
def mock_app() -> FastAPI:
    """Create a test FastAPI app with search router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def mock_graph() -> MagicMock:
    """Create a mock Neo4jGraph."""
    return MagicMock()


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock Neo4jVector store."""
    return MagicMock()


@pytest.fixture
def client(
    mock_app: FastAPI, mock_graph: MagicMock, mock_vector_store: MagicMock
) -> TestClient:
    """Create a test client with mocked dependencies."""
    mock_app.state.graph = mock_graph
    mock_app.state.vector_store = mock_vector_store
    return TestClient(mock_app)


@pytest.fixture
def sample_search_results() -> list[dict]:
    """Create sample search results."""
    return [
        {
            "content": "Requirements traceability content",
            "score": 0.95,
            "metadata": {"title": "Article 1", "url": "https://example.com/1"},
        },
        {
            "content": "More content about requirements",
            "score": 0.88,
            "metadata": {"title": "Article 2", "url": "https://example.com/2"},
        },
    ]


class TestVectorSearchEndpoint:
    """Tests for POST /api/v1/search/vector endpoint."""

    def test_vector_search_success(
        self, client: TestClient, sample_search_results: list[dict]
    ) -> None:
        """Test successful vector search."""
        with patch(
            "jama_mcp_server_graphrag.routes.search.vector_search"
        ) as mock_search:
            mock_search.return_value = sample_search_results

            response = client.post(
                "/api/v1/search/vector",
                json={"query": "requirements traceability", "limit": 10},
            )

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert data["total"] == 2

    def test_vector_search_validates_query(self, client: TestClient) -> None:
        """Test that empty query is rejected."""
        response = client.post("/api/v1/search/vector", json={"query": "", "limit": 10})

        assert response.status_code == 422  # Validation error

    def test_vector_search_validates_limit(self, client: TestClient) -> None:
        """Test that invalid limit is rejected."""
        response = client.post(
            "/api/v1/search/vector", json={"query": "test", "limit": 100}
        )

        assert response.status_code == 422  # limit > 50


class TestHybridSearchEndpoint:
    """Tests for POST /api/v1/search/hybrid endpoint."""

    def test_hybrid_search_success(
        self, client: TestClient, sample_search_results: list[dict]
    ) -> None:
        """Test successful hybrid search."""
        with patch(
            "jama_mcp_server_graphrag.routes.search.hybrid_search"
        ) as mock_search:
            mock_search.return_value = sample_search_results

            response = client.post(
                "/api/v1/search/hybrid",
                json={"query": "ISO 26262", "limit": 5, "keyword_weight": 0.4},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 2

    def test_hybrid_search_default_keyword_weight(
        self, client: TestClient, sample_search_results: list[dict]
    ) -> None:
        """Test that default keyword_weight is used."""
        with patch(
            "jama_mcp_server_graphrag.routes.search.hybrid_search"
        ) as mock_search:
            mock_search.return_value = sample_search_results

            response = client.post(
                "/api/v1/search/hybrid", json={"query": "test", "limit": 5}
            )

            assert response.status_code == 200
            # Check that hybrid_search was called with default weight
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["keyword_weight"] == 0.3


class TestGraphSearchEndpoint:
    """Tests for POST /api/v1/search/graph endpoint."""

    def test_graph_search_success(
        self, client: TestClient, sample_search_results: list[dict]
    ) -> None:
        """Test successful graph-enriched search."""
        # Add entity data to results
        enriched_results = [
            {**r, "related_entities": ["entity1"], "glossary_terms": ["term1"]}
            for r in sample_search_results
        ]

        with patch(
            "jama_mcp_server_graphrag.routes.search.graph_enriched_search"
        ) as mock_search:
            mock_search.return_value = enriched_results

            response = client.post(
                "/api/v1/search/graph",
                json={"query": "requirements", "limit": 5, "traversal_depth": 2},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 2
            assert "related_entities" in data["results"][0]

    def test_graph_search_validates_traversal_depth(self, client: TestClient) -> None:
        """Test that invalid traversal_depth is rejected."""
        response = client.post(
            "/api/v1/search/graph",
            json={"query": "test", "limit": 5, "traversal_depth": 10},
        )

        assert response.status_code == 422  # traversal_depth > 3
