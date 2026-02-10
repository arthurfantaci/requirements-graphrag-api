"""Tests for chat endpoint with SSE streaming.

Updated Data Model (2026-01):
- Uses app.state.driver and app.state.retriever instead of graph/vector_store
- Tests SSE streaming response format

Agentic RAG (2026-02):
- Updated to mock stream_agentic_events instead of stream_chat
- Tests agentic orchestrator integration
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from requirements_graphrag_api.core.definitions import StreamEventType
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
    config.toxicity_enabled = False
    config.topic_guard_enabled = False
    config.output_filter_enabled = False
    config.hallucination_enabled = False
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


def create_mock_agentic_sse_events(
    sources: list[dict] | None = None,
    entities: list[dict] | None = None,
    answer: str = "Requirements traceability is the ability to track requirements.",
) -> list[str]:
    """Create a list of mock SSE event strings for agentic streaming.

    The agentic streaming returns pre-formatted SSE strings, not StreamEvent objects.
    Note: Routing event is NOT included here because it's emitted by _generate_sse_events
    before calling the agentic streaming.

    Args:
        sources: Optional source list.
        entities: Optional entity list.
        answer: The answer to include in tokens and done event.

    Returns:
        List of SSE-formatted strings in the expected order.
    """
    if sources is None:
        sources = [
            {
                "title": "Traceability Guide",
                "content": "Requirements traceability...",
                "score": 0.95,
            }
        ]
    if entities is None:
        entities = [{"name": "requirements traceability", "type": "Concept", "description": "..."}]

    events = []

    # Phase event (agentic-specific) - first event from stream_agentic_events
    phase_rag = {"phase": "rag", "message": "Retrieving relevant context..."}
    events.append(f"data: {json.dumps(phase_rag)}\n\n")

    # Sources event
    events.append(f"data: {json.dumps({'sources': sources})}\n\n")

    # Phase change to synthesis
    phase_synth = {"phase": "synthesis", "message": "Generating answer..."}
    events.append(f"data: {json.dumps(phase_synth)}\n\n")

    # Token events (chunked answer)
    chunk_size = 20
    for i in range(0, len(answer), chunk_size):
        chunk = answer[i : i + chunk_size]
        events.append(f"data: {json.dumps({'token': chunk})}\n\n")

    # Done event
    events.append(f"data: {json.dumps({'full_answer': answer, 'source_count': len(sources)})}\n\n")

    return events


async def mock_agentic_stream_generator(events: list[str]) -> AsyncIterator[str]:
    """Create an async generator that yields SSE strings."""
    for event in events:
        yield event


def parse_sse_response(response_text: str) -> list[dict]:
    """Parse SSE response text into a list of events.

    Args:
        response_text: Raw SSE response text.

    Returns:
        List of parsed data payloads.
    """
    events = []

    for line in response_text.strip().split("\n"):
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                events.append(data)
            except json.JSONDecodeError:
                continue

    return events


class TestChatEndpointStreaming:
    """Tests for POST /api/v1/chat SSE streaming endpoint."""

    def test_chat_returns_sse_content_type(self, client: TestClient) -> None:
        """Test that the endpoint returns text/event-stream content type."""
        events = create_mock_agentic_sse_events()

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={"message": "What is requirements traceability?"},
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_chat_returns_correct_sse_headers(self, client: TestClient) -> None:
        """Test that SSE-specific headers are set correctly."""
        events = create_mock_agentic_sse_events()

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={"message": "Test"},
            )

            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["x-accel-buffering"] == "no"

    def test_chat_emits_routing_event_first(self, client: TestClient) -> None:
        """Test that the first SSE event contains routing info."""
        events = create_mock_agentic_sse_events()

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "What is traceability?",
                    "options": {"force_intent": "explanatory"},
                },
            )

            parsed_events = parse_sse_response(response.text)

            assert len(parsed_events) > 0
            # First event should have intent (routing)
            assert "intent" in parsed_events[0]
            assert parsed_events[0]["intent"] == "explanatory"

    def test_chat_emits_token_events(self, client: TestClient) -> None:
        """Test that token events are emitted during streaming."""
        events = create_mock_agentic_sse_events(answer="Test answer")

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test",
                    "options": {"force_intent": "explanatory"},
                },
            )

            parsed_events = parse_sse_response(response.text)
            token_events = [e for e in parsed_events if "token" in e]

            assert len(token_events) > 0
            for event in token_events:
                assert "token" in event

    def test_chat_emits_done_event_last(self, client: TestClient) -> None:
        """Test that the last SSE event is the done event."""
        events = create_mock_agentic_sse_events()

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test",
                    "options": {"force_intent": "explanatory"},
                },
            )

            parsed_events = parse_sse_response(response.text)

            assert len(parsed_events) > 0
            # Last event should have full_answer (done event)
            assert "full_answer" in parsed_events[-1]
            assert "source_count" in parsed_events[-1]

    def test_chat_correct_event_sequence(self, client: TestClient) -> None:
        """Test that events are emitted in expected order."""
        events = create_mock_agentic_sse_events(answer="One two three")

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test",
                    "options": {"force_intent": "explanatory"},
                },
            )

            parsed_events = parse_sse_response(response.text)

            # Should have: routing, phase(rag), sources, phase(synthesis), tokens..., done
            assert "intent" in parsed_events[0]  # Routing
            assert "full_answer" in parsed_events[-1]  # Done

    def test_chat_with_conversation_history(self, client: TestClient) -> None:
        """Test that conversation history is processed correctly."""
        events = create_mock_agentic_sse_events(answer="Follow up answer")

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(events)

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

            # Verify the call was made with initial state containing messages
            call_args = mock_stream.call_args
            initial_state = call_args[0][1]  # Second positional arg
            assert "messages" in initial_state
            # Should have history messages + current query (3 total)
            assert len(initial_state["messages"]) >= 3

    def test_chat_assistant_messages_become_aimessage(self, client: TestClient) -> None:
        """Test that assistant messages in history are converted to AIMessage (F2 fix)."""
        events = create_mock_agentic_sse_events(answer="Follow up")

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Tell me more",
                    "conversation_history": [
                        {"role": "user", "content": "What is traceability?"},
                        {"role": "assistant", "content": "Traceability is..."},
                    ],
                    "options": {"force_intent": "explanatory"},
                },
            )

            assert response.status_code == 200

            call_args = mock_stream.call_args
            initial_state = call_args[0][1]
            messages = initial_state["messages"]

            # First message should be HumanMessage (from history)
            assert isinstance(messages[0], HumanMessage)
            assert messages[0].content == "What is traceability?"

            # Second message should be AIMessage (from history - F2 fix)
            assert isinstance(messages[1], AIMessage)
            assert messages[1].content == "Traceability is..."

            # Third message should be HumanMessage (current query)
            assert isinstance(messages[2], HumanMessage)

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
        """Test that custom options are handled correctly."""
        events = create_mock_agentic_sse_events()

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(events)

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

    def test_chat_handles_empty_sources(self, client: TestClient) -> None:
        """Test that empty sources list is handled gracefully."""
        events = create_mock_agentic_sse_events(sources=[], entities=[])

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test",
                    "options": {"force_intent": "explanatory"},
                },
            )

            assert response.status_code == 200
            parsed_events = parse_sse_response(response.text)

            # Find the sources event
            sources_events = [e for e in parsed_events if "sources" in e]
            assert len(sources_events) > 0
            assert sources_events[0]["sources"] == []

    def test_chat_error_event(self, client: TestClient) -> None:
        """Test that error events are properly formatted."""
        error_events = [f"data: {json.dumps({'error': 'Something went wrong'})}\n\n"]

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(error_events)

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Test",
                    "options": {"force_intent": "explanatory"},
                },
            )

            assert response.status_code == 200
            parsed_events = parse_sse_response(response.text)

            # Should have routing event, then error from agentic stream
            assert len(parsed_events) >= 1
            # Find error event
            error_events = [e for e in parsed_events if "error" in e]
            assert len(error_events) > 0
            assert "error" in error_events[0]


class TestChatMessageModel:
    """Tests for ChatMessage Pydantic model validation."""

    def test_valid_user_role(self, client: TestClient) -> None:
        """Test that 'user' role is accepted."""
        events = create_mock_agentic_sse_events()

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(events)

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
        events = create_mock_agentic_sse_events()

        with (
            patch("requirements_graphrag_api.routes.chat.create_orchestrator_graph"),
            patch("requirements_graphrag_api.routes.chat.stream_agentic_events") as mock_stream,
        ):
            mock_stream.return_value = mock_agentic_stream_generator(events)

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


class TestConversationalRouting:
    """Tests for CONVERSATIONAL intent routing through the chat endpoint."""

    def test_force_intent_conversational_accepted(self, client: TestClient) -> None:
        """Test force_intent=conversational returns 200 with correct events."""
        with patch(
            "requirements_graphrag_api.routes.chat.stream_conversational_events"
        ) as mock_stream:
            answer = "Your first question was about traceability."

            async def mock_events(*_args, **_kwargs):
                yield f"event: {StreamEventType.TOKEN.value}\n"
                yield f"data: {json.dumps({'token': answer})}\n\n"
                yield f"event: {StreamEventType.DONE.value}\n"
                done = {
                    "full_answer": answer,
                    "source_count": 0,
                    "run_id": "test-id",
                }
                yield f"data: {json.dumps(done)}\n\n"

            mock_stream.return_value = mock_events()

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "What was my first question?",
                    "conversation_history": [
                        {"role": "user", "content": "What is traceability?"},
                        {"role": "assistant", "content": "Traceability is..."},
                    ],
                    "options": {"force_intent": "conversational"},
                },
            )

            assert response.status_code == 200
            parsed = parse_sse_response(response.text)
            # First event: routing with intent=conversational
            assert parsed[0]["intent"] == "conversational"

    def test_force_intent_invalid_rejected(self, client: TestClient) -> None:
        """Test force_intent with invalid value returns 422."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Test",
                "options": {"force_intent": "invalid_intent"},
            },
        )

        assert response.status_code == 422

    def test_conversational_no_cypher_or_phase_events(self, client: TestClient) -> None:
        """Test conversational route emits no cypher/phase events.

        Note: An empty sources event IS emitted so the frontend SSE parser
        proceeds to token rendering (added in chat.py _generate_conversational_events).
        """
        with patch(
            "requirements_graphrag_api.routes.chat.stream_conversational_events"
        ) as mock_stream:

            async def mock_events(*_args, **_kwargs):
                yield f"event: {StreamEventType.TOKEN.value}\n"
                yield f"data: {json.dumps({'token': 'Recall answer'})}\n\n"
                yield f"event: {StreamEventType.DONE.value}\n"
                yield f"data: {json.dumps({'full_answer': 'Recall answer', 'source_count': 0})}\n\n"

            mock_stream.return_value = mock_events()

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "What was my first question?",
                    "conversation_history": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ],
                    "options": {"force_intent": "conversational"},
                },
            )

            parsed = parse_sse_response(response.text)
            for event in parsed:
                assert "cypher" not in event
                assert "phase" not in event

            # Verify the empty sources event is present for frontend compatibility
            sources_events = [e for e in parsed if "sources" in e]
            assert len(sources_events) == 1
            assert sources_events[0]["sources"] == []

    def test_conversational_done_has_run_id(self, client: TestClient) -> None:
        """Test done event includes run_id when available."""
        with patch(
            "requirements_graphrag_api.routes.chat.stream_conversational_events"
        ) as mock_stream:

            async def mock_events(*_args, **_kwargs):
                yield f"event: {StreamEventType.TOKEN.value}\n"
                yield f"data: {json.dumps({'token': 'Answer'})}\n\n"
                yield f"event: {StreamEventType.DONE.value}\n"
                done = {
                    "full_answer": "Answer",
                    "source_count": 0,
                    "run_id": "conv-run-123",
                }
                yield f"data: {json.dumps(done)}\n\n"

            mock_stream.return_value = mock_events()

            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Recap",
                    "conversation_history": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello"},
                    ],
                    "options": {"force_intent": "conversational"},
                },
            )

            parsed = parse_sse_response(response.text)
            done_events = [e for e in parsed if "full_answer" in e]
            assert len(done_events) == 1
            assert done_events[0]["run_id"] == "conv-run-123"


class TestCoreferenceResolution:
    """Tests for _resolve_coreferences() helper function."""

    @pytest.mark.asyncio
    async def test_no_history_returns_original(self) -> None:
        """Test passthrough when no conversation history."""
        from requirements_graphrag_api.routes.chat import _resolve_coreferences

        config = MagicMock()
        result = await _resolve_coreferences(config, "List all webinars", None)
        assert result == "List all webinars"

    @pytest.mark.asyncio
    async def test_empty_history_returns_original(self) -> None:
        """Test passthrough with empty history list."""
        from requirements_graphrag_api.routes.chat import _resolve_coreferences

        config = MagicMock()
        result = await _resolve_coreferences(config, "List all webinars", [])
        assert result == "List all webinars"

    @pytest.mark.asyncio
    async def test_resolves_pronoun_with_history(self) -> None:
        """Test that pronouns are resolved using conversation history."""
        from requirements_graphrag_api.routes.chat import ChatMessage, _resolve_coreferences

        config = MagicMock()
        config.conversational_model = "gpt-4o-mini"
        config.openai_api_key = "test-key"

        history = [
            ChatMessage(role="user", content="What standards apply to aerospace?"),
            ChatMessage(role="assistant", content="DO-178C and ARP 4754A apply to aerospace."),
        ]

        resolved_text = "Are there any webinars related to the aerospace industry?"

        # Mock the LangChain chain: template | llm | StrOutputParser
        mock_final_chain = MagicMock()
        mock_final_chain.ainvoke = AsyncMock(return_value=resolved_text)
        mock_intermediate = MagicMock()
        mock_intermediate.__or__ = MagicMock(return_value=mock_final_chain)
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_intermediate)

        with (
            patch(
                "requirements_graphrag_api.routes.chat.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("langchain_openai.ChatOpenAI"),
        ):
            result = await _resolve_coreferences(
                config, "Are there any webinars related to that industry?", history
            )
            assert result == resolved_text

    @pytest.mark.asyncio
    async def test_llm_error_returns_original(self) -> None:
        """Test graceful fallback on LLM error."""
        from requirements_graphrag_api.routes.chat import ChatMessage, _resolve_coreferences

        config = MagicMock()
        config.conversational_model = "gpt-4o-mini"
        config.openai_api_key = "test-key"

        history = [ChatMessage(role="user", content="Hi")]

        mock_final_chain = MagicMock()
        mock_final_chain.ainvoke = AsyncMock(side_effect=RuntimeError("LLM error"))
        mock_intermediate = MagicMock()
        mock_intermediate.__or__ = MagicMock(return_value=mock_final_chain)
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_intermediate)

        with (
            patch(
                "requirements_graphrag_api.routes.chat.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("langchain_openai.ChatOpenAI"),
        ):
            result = await _resolve_coreferences(config, "those two industries", history)
            assert result == "those two industries"

    @pytest.mark.asyncio
    async def test_empty_llm_response_returns_original(self) -> None:
        """Test fallback when LLM returns empty string."""
        from requirements_graphrag_api.routes.chat import ChatMessage, _resolve_coreferences

        config = MagicMock()
        config.conversational_model = "gpt-4o-mini"
        config.openai_api_key = "test-key"

        history = [ChatMessage(role="user", content="Hi")]

        mock_final_chain = MagicMock()
        mock_final_chain.ainvoke = AsyncMock(return_value="")
        mock_intermediate = MagicMock()
        mock_intermediate.__or__ = MagicMock(return_value=mock_final_chain)
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_intermediate)

        with (
            patch(
                "requirements_graphrag_api.routes.chat.get_prompt",
                new_callable=AsyncMock,
                return_value=mock_prompt,
            ),
            patch("langchain_openai.ChatOpenAI"),
        ):
            result = await _resolve_coreferences(config, "those two industries", history)
            assert result == "those two industries"
