"""Tests for chat endpoint with SSE streaming.

Updated Data Model (2026-01):
- Uses app.state.driver and app.state.retriever instead of graph/vector_store
- Tests SSE streaming response format
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from requirements_graphrag_api.core.generation import StreamEvent, StreamEventType
from requirements_graphrag_api.routes.chat import router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


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
def mock_guardrail_config() -> MagicMock:
    """Create a mock GuardrailConfig with guardrails disabled."""
    config = MagicMock()
    config.prompt_injection_enabled = False
    config.pii_detection_enabled = False
    config.rate_limiting_enabled = False
    config.injection_block_threshold = "high"
    config.pii_entities = ("EMAIL_ADDRESS", "PHONE_NUMBER")
    config.pii_score_threshold = 0.7
    config.pii_anonymize_type = "replace"
    return config


@pytest.fixture
def client(
    mock_app: FastAPI, mock_config: MagicMock, mock_guardrail_config: MagicMock
) -> TestClient:
    """Create a test client with mocked dependencies."""
    mock_app.state.config = mock_config
    mock_app.state.driver = MagicMock()
    mock_app.state.retriever = MagicMock()
    mock_app.state.guardrail_config = mock_guardrail_config
    return TestClient(mock_app)


def create_mock_stream_events(
    sources: list[dict] | None = None,
    entities: list[str] | None = None,
    images: list[dict] | None = None,
    answer: str = "Requirements traceability is the ability to track requirements.",
) -> list[StreamEvent]:
    """Create a list of mock stream events for testing.

    Args:
        sources: Optional source list.
        entities: Optional entity list.
        images: Optional image list.
        answer: The answer to split into tokens.

    Returns:
        List of StreamEvents in the expected order.
    """
    if sources is None:
        sources = [
            {
                "title": "Traceability Guide",
                "url": "https://example.com/trace",
                "chunk_id": "chunk-1",
                "relevance_score": 0.95,
            }
        ]
    if entities is None:
        entities = ["requirements traceability", "lifecycle"]
    if images is None:
        images = [
            {
                "url": "https://jamasoftware.com/images/traceability-matrix.png",
                "alt_text": "Traceability matrix example",
                "context": "A traceability matrix showing relationships",
                "source_title": "Traceability Guide",
            }
        ]

    events = [
        StreamEvent(
            event_type=StreamEventType.SOURCES,
            data={"sources": sources, "entities": entities, "images": images},
        ),
    ]

    # Split answer into tokens
    words = answer.split()
    for i, word in enumerate(words):
        token = word if i == 0 else f" {word}"
        events.append(
            StreamEvent(
                event_type=StreamEventType.TOKEN,
                data={"token": token},
            )
        )

    events.append(
        StreamEvent(
            event_type=StreamEventType.DONE,
            data={"full_answer": answer, "source_count": len(sources)},
        )
    )

    return events


async def mock_stream_chat_generator(events: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
    """Create an async generator that yields stream events."""
    for event in events:
        yield event


def parse_sse_response(response_text: str) -> list[dict]:
    """Parse SSE response text into a list of events.

    Args:
        response_text: Raw SSE response text.

    Returns:
        List of parsed events with 'event' and 'data' keys.
    """
    events = []
    current_event = {}

    for line in response_text.strip().split("\n"):
        if line.startswith("event: "):
            current_event["event"] = line[7:]
        elif line.startswith("data: "):
            current_event["data"] = json.loads(line[6:])
        elif line == "" and current_event:
            events.append(current_event)
            current_event = {}

    # Handle last event if no trailing newline
    if current_event:
        events.append(current_event)

    return events


class TestChatEndpointStreaming:
    """Tests for POST /api/v1/chat SSE streaming endpoint."""

    def test_chat_returns_sse_content_type(self, client: TestClient) -> None:
        """Test that the endpoint returns text/event-stream content type."""
        events = create_mock_stream_events()

        with patch("requirements_graphrag_api.routes.chat.stream_chat") as mock_stream:
            mock_stream.return_value = mock_stream_chat_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={"message": "What is requirements traceability?"},
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_chat_returns_correct_sse_headers(self, client: TestClient) -> None:
        """Test that SSE-specific headers are set correctly."""
        events = create_mock_stream_events()

        with patch("requirements_graphrag_api.routes.chat.stream_chat") as mock_stream:
            mock_stream.return_value = mock_stream_chat_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={"message": "Test"},
            )

            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["x-accel-buffering"] == "no"

    def test_chat_emits_routing_event_first(self, client: TestClient) -> None:
        """Test that the first SSE event is the routing event."""
        events = create_mock_stream_events()

        with patch("requirements_graphrag_api.routes.chat.stream_chat") as mock_stream:
            mock_stream.return_value = mock_stream_chat_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "What is traceability?",
                    "options": {"force_intent": "explanatory"},
                },
            )

            parsed_events = parse_sse_response(response.text)

            assert len(parsed_events) > 0
            # First event is routing
            assert parsed_events[0]["event"] == "routing"
            assert parsed_events[0]["data"]["intent"] == "explanatory"
            # Second event is sources
            assert parsed_events[1]["event"] == "sources"
            assert "sources" in parsed_events[1]["data"]
            assert "entities" in parsed_events[1]["data"]
            assert "images" in parsed_events[1]["data"]

    def test_chat_emits_token_events(self, client: TestClient) -> None:
        """Test that token events are emitted during streaming."""
        events = create_mock_stream_events(answer="Test answer")

        with patch("requirements_graphrag_api.routes.chat.stream_chat") as mock_stream:
            mock_stream.return_value = mock_stream_chat_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test",
                    "options": {"force_intent": "explanatory"},
                },
            )

            parsed_events = parse_sse_response(response.text)
            token_events = [e for e in parsed_events if e["event"] == "token"]

            assert len(token_events) > 0
            for event in token_events:
                assert "token" in event["data"]

    def test_chat_emits_done_event_last(self, client: TestClient) -> None:
        """Test that the last SSE event is the done event."""
        events = create_mock_stream_events()

        with patch("requirements_graphrag_api.routes.chat.stream_chat") as mock_stream:
            mock_stream.return_value = mock_stream_chat_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test",
                    "options": {"force_intent": "explanatory"},
                },
            )

            parsed_events = parse_sse_response(response.text)

            assert len(parsed_events) > 0
            assert parsed_events[-1]["event"] == "done"
            assert "full_answer" in parsed_events[-1]["data"]
            assert "source_count" in parsed_events[-1]["data"]

    def test_chat_correct_event_sequence(self, client: TestClient) -> None:
        """Test that events are emitted in correct order."""
        events = create_mock_stream_events(answer="One two three")

        with patch("requirements_graphrag_api.routes.chat.stream_chat") as mock_stream:
            mock_stream.return_value = mock_stream_chat_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test",
                    "options": {"force_intent": "explanatory"},
                },
            )

            parsed_events = parse_sse_response(response.text)
            event_types = [e["event"] for e in parsed_events]

            # First should be routing, second sources, last should be done
            assert event_types[0] == "routing"
            assert event_types[1] == "sources"
            assert event_types[-1] == "done"
            # All middle events (after routing and sources, before done) should be tokens
            for event_type in event_types[2:-1]:
                assert event_type == "token"

    def test_chat_with_conversation_history(self, client: TestClient) -> None:
        """Test that conversation history is passed to stream_chat."""
        events = create_mock_stream_events(answer="Follow up answer")

        with patch("requirements_graphrag_api.routes.chat.stream_chat") as mock_stream:
            mock_stream.return_value = mock_stream_chat_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Can you give me an example?",
                    "conversation_history": [
                        {"role": "user", "content": "What is traceability?"},
                        {"role": "assistant", "content": "Traceability is..."},
                    ],
                    "options": {"force_intent": "explanatory"},
                },
            )

            assert response.status_code == 200

            # Verify history was passed to stream_chat
            call_kwargs = mock_stream.call_args[1]
            assert call_kwargs["conversation_history"] is not None
            assert len(call_kwargs["conversation_history"]) == 2

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

    def test_chat_validates_conversation_history_role(self, client: TestClient) -> None:
        """Test that invalid role in conversation history is rejected."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Test",
                "conversation_history": [
                    {"role": "invalid", "content": "Test"},
                ],
            },
        )

        assert response.status_code == 422

    def test_chat_with_custom_options(self, client: TestClient) -> None:
        """Test that custom options are passed to stream_chat."""
        events = create_mock_stream_events()

        with patch("requirements_graphrag_api.routes.chat.stream_chat") as mock_stream:
            mock_stream.return_value = mock_stream_chat_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test question",
                    "conversation_id": "test-123",
                    "options": {
                        "retrieval_strategy": "graph",
                        "include_sources": True,
                        "max_sources": 3,
                        "force_intent": "explanatory",
                    },
                },
            )

            assert response.status_code == 200

            # Verify max_sources was passed
            call_kwargs = mock_stream.call_args[1]
            assert call_kwargs["max_sources"] == 3

    def test_chat_handles_empty_sources(self, client: TestClient) -> None:
        """Test that empty sources list is handled gracefully."""
        events = create_mock_stream_events(sources=[], entities=[], images=[])

        with patch("requirements_graphrag_api.routes.chat.stream_chat") as mock_stream:
            mock_stream.return_value = mock_stream_chat_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test",
                    "options": {"force_intent": "explanatory"},
                },
            )

            assert response.status_code == 200
            parsed_events = parse_sse_response(response.text)

            # First event is routing, second is sources
            assert parsed_events[0]["event"] == "routing"
            sources_event = parsed_events[1]
            assert sources_event["event"] == "sources"
            assert sources_event["data"]["sources"] == []
            assert sources_event["data"]["entities"] == []
            assert sources_event["data"]["images"] == []

    def test_chat_error_event(self, client: TestClient) -> None:
        """Test that error events are properly formatted."""
        error_events = [
            StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": "Something went wrong"},
            )
        ]

        with patch("requirements_graphrag_api.routes.chat.stream_chat") as mock_stream:
            mock_stream.return_value = mock_stream_chat_generator(error_events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test",
                    "options": {"force_intent": "explanatory"},
                },
            )

            assert response.status_code == 200
            parsed_events = parse_sse_response(response.text)

            # First is routing, then error from stream_chat
            assert len(parsed_events) == 2
            assert parsed_events[0]["event"] == "routing"
            assert parsed_events[1]["event"] == "error"
            assert "error" in parsed_events[1]["data"]


class TestChatMessageModel:
    """Tests for ChatMessage Pydantic model validation."""

    def test_valid_user_role(self, client: TestClient) -> None:
        """Test that 'user' role is accepted."""
        events = create_mock_stream_events()

        with patch("requirements_graphrag_api.routes.chat.stream_chat") as mock_stream:
            mock_stream.return_value = mock_stream_chat_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test",
                    "conversation_history": [
                        {"role": "user", "content": "Hello"},
                    ],
                },
            )

            assert response.status_code == 200

    def test_valid_assistant_role(self, client: TestClient) -> None:
        """Test that 'assistant' role is accepted."""
        events = create_mock_stream_events()

        with patch("requirements_graphrag_api.routes.chat.stream_chat") as mock_stream:
            mock_stream.return_value = mock_stream_chat_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test",
                    "conversation_history": [
                        {"role": "assistant", "content": "Hi there"},
                    ],
                },
            )

            assert response.status_code == 200

    def test_invalid_role_rejected(self, client: TestClient) -> None:
        """Test that invalid roles are rejected."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Test",
                "conversation_history": [
                    {"role": "system", "content": "System message"},
                ],
            },
        )

        assert response.status_code == 422

    def test_empty_content_rejected(self, client: TestClient) -> None:
        """Test that empty content in history is rejected."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Test",
                "conversation_history": [
                    {"role": "user", "content": ""},
                ],
            },
        )

        assert response.status_code == 422
