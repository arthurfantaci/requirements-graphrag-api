"""Tests for core generation functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from requirements_graphrag_api.core.generation import (
    StreamEvent,
    StreamEventType,
    generate_answer,
    stream_chat,
)
from tests.conftest import create_llm_mock

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock AppConfig."""
    config = MagicMock()
    config.chat_model = "gpt-4o"
    config.openai_api_key = "test-api-key"
    return config


@pytest.fixture
def mock_graph() -> MagicMock:
    """Create a mock Neo4jGraph."""
    graph = MagicMock()
    graph.query = MagicMock(return_value=[])
    return graph


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock Neo4jVector store."""
    store = MagicMock()
    store.similarity_search_with_score = MagicMock(
        return_value=[
            (
                MagicMock(
                    page_content="Requirements traceability is essential for quality.",
                    metadata={
                        "title": "Traceability Guide",
                        "url": "https://example.com/trace",
                        "chunk_id": "chunk-1",
                    },
                ),
                0.92,
            ),
        ]
    )
    return store


@pytest.fixture
def mock_search_results() -> list[dict]:
    """Create mock graph-enriched search results."""
    return [
        {
            "content": "Requirements traceability enables tracking requirements.",
            "score": 0.95,
            "metadata": {
                "title": "Traceability Article",
                "url": "https://example.com/article1",
                "chunk_id": "chunk-1",
            },
            "entities": ["requirements", "traceability"],
            "glossary_definitions": ["traceability matrix"],
        },
        {
            "content": "Best practices for requirements management.",
            "score": 0.88,
            "metadata": {
                "title": "Best Practices",
                "url": "https://example.com/article2",
                "chunk_id": "chunk-2",
            },
            "entities": ["requirements management"],
            "glossary_definitions": [],
        },
    ]


# =============================================================================
# Generate Answer Tests
# =============================================================================


class TestGenerateAnswer:
    """Tests for generate_answer function."""

    @pytest.mark.asyncio
    async def test_generate_answer_returns_complete_response(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        mock_search_results: list[dict],
    ) -> None:
        """Test that generate_answer returns a complete response."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = mock_search_results
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock(
                "Requirements traceability is the ability to track..."
            )

            result = await generate_answer(
                mock_config,
                mock_graph,
                mock_vector_store,
                "What is requirements traceability?",
            )

            assert "question" in result
            assert "answer" in result
            assert "sources" in result
            assert "entities" in result
            assert result["source_count"] > 0

    @pytest.mark.asyncio
    async def test_generate_answer_includes_sources(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        mock_search_results: list[dict],
    ) -> None:
        """Test that sources are properly formatted."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = mock_search_results
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock("Answer text")

            result = await generate_answer(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Test question",
            )

            assert len(result["sources"]) == 2
            assert result["sources"][0]["title"] == "Traceability Article"
            assert "relevance_score" in result["sources"][0]

    @pytest.mark.asyncio
    async def test_generate_answer_respects_retrieval_limit(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test that retrieval_limit parameter is passed to search."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = []
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock("No context found.")

            await generate_answer(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Test",
                retrieval_limit=10,
            )

            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["limit"] == 10


# =============================================================================
# Stream Chat Tests
# =============================================================================


class TestStreamChat:
    """Tests for stream_chat function."""

    @pytest.mark.asyncio
    async def test_stream_chat_emits_sources_event_first(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        mock_search_results: list[dict],
    ) -> None:
        """Test that the first event is the sources event."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = mock_search_results
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock("Test answer", streaming=True)

            events = []
            async for event in stream_chat(
                mock_config,
                mock_graph,
                mock_vector_store,
                "What is traceability?",
            ):
                events.append(event)

            # First event should be sources
            assert len(events) >= 1
            assert events[0].event_type == StreamEventType.SOURCES
            assert "sources" in events[0].data
            assert "entities" in events[0].data
            assert "images" in events[0].data

    @pytest.mark.asyncio
    async def test_stream_chat_emits_token_events(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        mock_search_results: list[dict],
    ) -> None:
        """Test that token events are emitted during streaming."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = mock_search_results
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock("Test answer tokens", streaming=True)

            events = []
            async for event in stream_chat(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Test question",
            ):
                events.append(event)

            # Find token events
            token_events = [e for e in events if e.event_type == StreamEventType.TOKEN]
            assert len(token_events) > 0
            # Each token event should have a "token" key
            for token_event in token_events:
                assert "token" in token_event.data

    @pytest.mark.asyncio
    async def test_stream_chat_emits_done_event_last(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        mock_search_results: list[dict],
    ) -> None:
        """Test that the last event is the done event with full answer."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = mock_search_results
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock("Complete answer text", streaming=True)

            events = []
            async for event in stream_chat(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Test question",
            ):
                events.append(event)

            # Last event should be done
            assert len(events) >= 1
            assert events[-1].event_type == StreamEventType.DONE
            assert "full_answer" in events[-1].data
            assert "source_count" in events[-1].data
            # The full answer should contain all tokens
            assert "Complete" in events[-1].data["full_answer"]
            assert "answer" in events[-1].data["full_answer"]

    @pytest.mark.asyncio
    async def test_stream_chat_correct_event_sequence(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        mock_search_results: list[dict],
    ) -> None:
        """Test that events are emitted in correct order: sources -> tokens -> done."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = mock_search_results
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock("Answer", streaming=True)

            event_types = []
            async for event in stream_chat(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Test",
            ):
                event_types.append(event.event_type)

            # Check order
            assert event_types[0] == StreamEventType.SOURCES
            assert event_types[-1] == StreamEventType.DONE
            # All middle events should be tokens
            for event_type in event_types[1:-1]:
                assert event_type == StreamEventType.TOKEN

    @pytest.mark.asyncio
    async def test_stream_chat_passes_max_sources(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test that max_sources is passed to graph_enriched_search."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = []
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock("No results", streaming=True)

            events = []
            async for event in stream_chat(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Test",
                max_sources=7,
            ):
                events.append(event)

            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["limit"] == 7

    @pytest.mark.asyncio
    async def test_stream_chat_with_conversation_history(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        mock_search_results: list[dict],
    ) -> None:
        """Test that conversation history is passed to the chain."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = mock_search_results
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock("Follow up answer", streaming=True)

            history = [
                {"role": "user", "content": "What is traceability?"},
                {"role": "assistant", "content": "Traceability is..."},
            ]

            events = []
            async for event in stream_chat(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Can you give me an example?",
                conversation_history=history,
            ):
                events.append(event)

            # Should complete without error
            assert len(events) >= 2
            assert events[0].event_type == StreamEventType.SOURCES
            assert events[-1].event_type == StreamEventType.DONE

    @pytest.mark.asyncio
    async def test_stream_chat_handles_empty_results(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test that streaming works when no results are found."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = []
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock(
                "I could not find relevant information.", streaming=True
            )

            events = []
            async for event in stream_chat(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Unknown topic",
            ):
                events.append(event)

            # Should still emit sources (empty), tokens, and done
            assert events[0].event_type == StreamEventType.SOURCES
            assert events[0].data["sources"] == []
            assert events[-1].event_type == StreamEventType.DONE


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_stream_event_is_frozen(self) -> None:
        """Test that StreamEvent is immutable."""
        event = StreamEvent(
            event_type=StreamEventType.TOKEN,
            data={"token": "test"},
        )
        with pytest.raises(AttributeError):
            event.event_type = StreamEventType.DONE  # type: ignore[misc]

    def test_stream_event_types(self) -> None:
        """Test that all expected event types exist."""
        assert StreamEventType.SOURCES == "sources"
        assert StreamEventType.TOKEN == "token"
        assert StreamEventType.DONE == "done"
        assert StreamEventType.ERROR == "error"
