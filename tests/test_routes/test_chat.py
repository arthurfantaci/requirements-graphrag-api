"""Tests for chat endpoint.

Updated Data Model (2026-01):
- Uses app.state.driver and app.state.retriever instead of graph/vector_store
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jama_mcp_server_graphrag.routes.chat import router


@pytest.fixture
def mock_app() -> FastAPI:
    """Create a test FastAPI app with chat router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock AppConfig."""
    config = MagicMock()
    config.chat_model = "gpt-4o"
    config.openai_api_key = "test-key"
    return config


@pytest.fixture
def client(mock_app: FastAPI, mock_config: MagicMock) -> TestClient:
    """Create a test client with mocked dependencies."""
    mock_app.state.config = mock_config
    mock_app.state.driver = MagicMock()
    mock_app.state.retriever = MagicMock()
    return TestClient(mock_app)


@pytest.fixture
def sample_chat_response() -> dict:
    """Create sample chat response."""
    return {
        "question": "What is requirements traceability?",
        "answer": (
            "Requirements traceability is the ability to trace "
            "requirements throughout the lifecycle."
        ),
        "sources": [
            {
                "title": "Traceability Guide",
                "url": "https://example.com/trace",
                "chunk_id": "chunk-1",
                "relevance_score": 0.95,
            }
        ],
        "entities": ["requirements traceability", "lifecycle"],
        "source_count": 1,
    }


class TestChatEndpoint:
    """Tests for POST /api/v1/chat endpoint."""

    def test_chat_success(self, client: TestClient, sample_chat_response: dict) -> None:
        """Test successful chat request."""
        with patch("jama_mcp_server_graphrag.routes.chat.core_chat") as mock_chat:
            mock_chat.return_value = sample_chat_response

            response = client.post(
                "/api/v1/chat",
                json={"message": "What is requirements traceability?"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "entities" in data

    def test_chat_with_options(self, client: TestClient, sample_chat_response: dict) -> None:
        """Test chat with custom options."""
        with patch("jama_mcp_server_graphrag.routes.chat.core_chat") as mock_chat:
            mock_chat.return_value = sample_chat_response

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test question",
                    "conversation_id": "test-123",
                    "options": {
                        "retrieval_strategy": "graph",
                        "include_sources": True,
                        "max_sources": 3,
                    },
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["conversation_id"] == "test-123"

    def test_chat_validates_message_length(self, client: TestClient) -> None:
        """Test that empty message is rejected."""
        response = client.post("/api/v1/chat", json={"message": ""})

        assert response.status_code == 422

    def test_chat_validates_max_sources(self, client: TestClient) -> None:
        """Test that invalid max_sources is rejected."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Test",
                "options": {"max_sources": 50},  # > 20
            },
        )

        assert response.status_code == 422

    def test_chat_transforms_sources(self, client: TestClient, sample_chat_response: dict) -> None:
        """Test that sources are properly transformed."""
        with patch("jama_mcp_server_graphrag.routes.chat.core_chat") as mock_chat:
            mock_chat.return_value = sample_chat_response

            response = client.post("/api/v1/chat", json={"message": "What is traceability?"})

            data = response.json()
            assert len(data["sources"]) == 1
            assert data["sources"][0]["title"] == "Traceability Guide"
            assert data["sources"][0]["relevance_score"] == 0.95

    def test_chat_transforms_entities(self, client: TestClient, sample_chat_response: dict) -> None:
        """Test that entities are properly transformed."""
        with patch("jama_mcp_server_graphrag.routes.chat.core_chat") as mock_chat:
            mock_chat.return_value = sample_chat_response

            response = client.post("/api/v1/chat", json={"message": "Test"})

            data = response.json()
            assert len(data["entities"]) == 2
            # String entities should be converted to EntityInfo format
            assert data["entities"][0]["name"] == "requirements traceability"
