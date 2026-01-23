"""Tests for core generation functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from requirements_graphrag_api.core.generation import (
    ContextBuildResult,
    Resource,
    StreamEvent,
    StreamEventType,
    _build_context_from_results,
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


@pytest.fixture
def mock_search_results_with_media() -> list[dict]:
    """Create mock graph-enriched search results with media content."""
    return [
        {
            "content": "Requirements traceability enables tracking requirements.",
            "score": 0.95,
            "metadata": {
                "title": "Traceability Article",
                "url": "https://example.com/article1",
                "chunk_id": "chunk-1",
            },
            "entities": [{"name": "requirements", "type": "Concept"}],
            "glossary_definitions": [],
            "media": {
                "images": [
                    {
                        "url": "https://example.com/image1.png",
                        "alt_text": "Traceability diagram showing relationships",
                        "context": "How traceability works",
                    },
                    {
                        "url": "https://example.com/image2.png",
                        "alt_text": "Requirements flow chart",
                        "context": "Requirements flow",
                    },
                ],
                "webinars": [
                    {
                        "title": "Traceability Best Practices",
                        "url": "https://example.com/webinar/traceability",
                    },
                ],
                "videos": [],
            },
        },
        {
            "content": "Best practices for requirements management.",
            "score": 0.88,
            "metadata": {
                "title": "Best Practices",
                "url": "https://example.com/article2",
                "chunk_id": "chunk-2",
            },
            "entities": [{"name": "requirements management", "type": "Concept"}],
            "glossary_definitions": [],
            "media": {
                "images": [
                    {
                        "url": "https://example.com/image3.png",
                        "alt_text": "Management dashboard screenshot",
                        "context": "Dashboard view",
                    },
                ],
                "webinars": [
                    {
                        "title": "Modern Requirements Management",
                        "url": "https://example.com/webinar/modern-rm",
                    },
                ],
                "videos": [
                    {
                        "title": "Quick Start Guide",
                        "url": "https://example.com/video/quickstart",
                    },
                ],
            },
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
            assert "resources" in events[0].data

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


# =============================================================================
# Definition with Acronym Tests
# =============================================================================


class TestDefinitionWithAcronymInContext:
    """Tests for definitions with acronyms in context and sources."""

    @pytest.mark.asyncio
    async def test_definition_with_acronym_in_sources(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test that definitions with acronyms include acronym in source title."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = []
            # Return a definition with an acronym
            mock_terms.return_value = [
                {
                    "term": "Analysis of Alternatives",
                    "definition": "A systematic evaluation of alternatives.",
                    "acronym": "AoA",
                    "url": "https://example.com/glossary#aoa",
                    "score": 1.0,
                }
            ]
            mock_llm_class.return_value = create_llm_mock("Definition answer")

            result = await generate_answer(
                mock_config,
                mock_graph,
                mock_vector_store,
                "What is AoA?",
            )

            # Check that the source title includes the acronym
            assert len(result["sources"]) >= 1
            definition_source = result["sources"][0]
            assert "AoA" in definition_source["title"]
            assert "Analysis of Alternatives" in definition_source["title"]

    @pytest.mark.asyncio
    async def test_definition_without_acronym_in_sources(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test that definitions without acronyms don't have parentheses in title."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = []
            # Return a definition without an acronym
            mock_terms.return_value = [
                {
                    "term": "Requirements Traceability",
                    "definition": "The ability to trace requirements.",
                    "acronym": None,
                    "url": "https://example.com/glossary#trace",
                    "score": 0.9,
                }
            ]
            mock_llm_class.return_value = create_llm_mock("Definition answer")

            result = await generate_answer(
                mock_config,
                mock_graph,
                mock_vector_store,
                "What is requirements traceability?",
            )

            # Check that the source title does NOT have parentheses (no acronym)
            definition_source = result["sources"][0]
            assert definition_source["title"] == "Definition: Requirements Traceability"
            assert "(" not in definition_source["title"]


# =============================================================================
# Resource and ContextBuildResult Dataclass Tests
# =============================================================================


class TestResourceDataclass:
    """Tests for Resource dataclass."""

    def test_resource_is_frozen(self) -> None:
        """Test that Resource is immutable."""
        resource = Resource(
            title="Test Webinar",
            url="https://example.com/webinar",
        )
        with pytest.raises(AttributeError):
            resource.title = "Modified"  # type: ignore[misc]

    def test_resource_with_all_fields(self) -> None:
        """Test Resource with all optional fields."""
        resource = Resource(
            title="Test Image",
            url="https://example.com/image.png",
            alt_text="A test image",
            source_title="Source Article",
        )
        assert resource.title == "Test Image"
        assert resource.url == "https://example.com/image.png"
        assert resource.alt_text == "A test image"
        assert resource.source_title == "Source Article"

    def test_resource_default_values(self) -> None:
        """Test Resource default values for optional fields."""
        resource = Resource(title="Test", url="https://example.com")
        assert resource.alt_text == ""
        assert resource.source_title == ""


class TestContextBuildResultDataclass:
    """Tests for ContextBuildResult dataclass."""

    def test_context_build_result_is_frozen(self) -> None:
        """Test that ContextBuildResult is immutable."""
        result = ContextBuildResult(
            sources=[],
            entities=[],
            context="test",
            entities_str="",
            resources={},
        )
        with pytest.raises(AttributeError):
            result.context = "modified"  # type: ignore[misc]

    def test_context_build_result_with_resources(self) -> None:
        """Test ContextBuildResult with populated resources."""
        webinar = Resource(title="Webinar", url="https://example.com/webinar")
        result = ContextBuildResult(
            sources=[{"title": "Test"}],
            entities=["entity1"],
            context="Context text",
            entities_str="entity1",
            resources={"webinars": [webinar], "images": [], "videos": []},
        )
        assert len(result.resources["webinars"]) == 1
        assert result.resources["webinars"][0].title == "Webinar"


# =============================================================================
# _build_context_from_results Tests
# =============================================================================


class TestBuildContextFromResults:
    """Tests for _build_context_from_results function."""

    def test_extracts_webinars_from_media(self, mock_search_results_with_media: list[dict]) -> None:
        """Verify webinars are extracted from search_results[].media.webinars."""
        result = _build_context_from_results(
            definitions=[],
            search_results=mock_search_results_with_media,
        )

        assert "webinars" in result.resources
        webinars = result.resources["webinars"]
        assert len(webinars) == 2
        titles = [w.title for w in webinars]
        assert "Traceability Best Practices" in titles
        assert "Modern Requirements Management" in titles

    def test_extracts_videos_from_media(self, mock_search_results_with_media: list[dict]) -> None:
        """Verify videos are extracted from search_results[].media.videos."""
        result = _build_context_from_results(
            definitions=[],
            search_results=mock_search_results_with_media,
        )

        assert "videos" in result.resources
        videos = result.resources["videos"]
        assert len(videos) == 1
        assert videos[0].title == "Quick Start Guide"

    def test_extracts_images_with_alt_text(
        self, mock_search_results_with_media: list[dict]
    ) -> None:
        """Verify images include alt_text for LLM reasoning."""
        result = _build_context_from_results(
            definitions=[],
            search_results=mock_search_results_with_media,
        )

        assert "images" in result.resources
        images = result.resources["images"]
        assert len(images) == 3
        # Check that alt_text is preserved
        alt_texts = [img.alt_text for img in images]
        assert "Traceability diagram showing relationships" in alt_texts

    def test_resources_include_source_title(
        self, mock_search_results_with_media: list[dict]
    ) -> None:
        """Verify resources include the source they came from."""
        result = _build_context_from_results(
            definitions=[],
            search_results=mock_search_results_with_media,
        )

        webinars = result.resources["webinars"]
        # First webinar should be from "Traceability Article"
        traceability_webinar = next(w for w in webinars if w.title == "Traceability Best Practices")
        assert traceability_webinar.source_title == "Traceability Article"

    def test_context_string_groups_resources_by_source(
        self, mock_search_results_with_media: list[dict]
    ) -> None:
        """Verify resources appear under their source in context string."""
        result = _build_context_from_results(
            definitions=[],
            search_results=mock_search_results_with_media,
        )

        # Check context contains resource sections
        assert "Resources from this source:" in result.context
        assert "Traceability Best Practices" in result.context
        assert "Modern Requirements Management" in result.context

    def test_context_includes_image_alt_text(
        self, mock_search_results_with_media: list[dict]
    ) -> None:
        """Verify image alt_text appears in context for LLM reasoning."""
        result = _build_context_from_results(
            definitions=[],
            search_results=mock_search_results_with_media,
        )

        assert "Traceability diagram showing relationships" in result.context

    def test_context_includes_resource_urls(
        self, mock_search_results_with_media: list[dict]
    ) -> None:
        """Verify resource URLs are in the context string for LLM."""
        result = _build_context_from_results(
            definitions=[],
            search_results=mock_search_results_with_media,
        )

        assert "https://example.com/webinar/traceability" in result.context
        assert "https://example.com/image1.png" in result.context

    def test_limits_resources_to_three_per_type_per_source(self) -> None:
        """Verify max 3 images, 3 webinars, 3 videos per source."""
        # Create a result with more than 3 of each type
        search_results = [
            {
                "content": "Test content",
                "score": 0.9,
                "metadata": {"title": "Test Article", "url": "https://example.com"},
                "entities": [],
                "glossary_definitions": [],
                "media": {
                    "images": [
                        {"url": f"https://example.com/img{i}.png", "alt_text": f"Image {i}"}
                        for i in range(5)
                    ],
                    "webinars": [
                        {"title": f"Webinar {i}", "url": f"https://example.com/web{i}"}
                        for i in range(5)
                    ],
                    "videos": [
                        {"title": f"Video {i}", "url": f"https://example.com/vid{i}"}
                        for i in range(5)
                    ],
                },
            }
        ]

        result = _build_context_from_results(
            definitions=[],
            search_results=search_results,
        )

        # Each type should be limited to 3
        assert len(result.resources["images"]) == 3
        assert len(result.resources["webinars"]) == 3
        assert len(result.resources["videos"]) == 3

    def test_deduplicates_resources_by_url(self) -> None:
        """Verify same URL from multiple sources appears once."""
        # Same webinar URL in two different sources
        shared_url = "https://example.com/webinar/shared"
        search_results = [
            {
                "content": "Content 1",
                "score": 0.9,
                "metadata": {"title": "Article 1", "url": "https://example.com/1"},
                "entities": [],
                "glossary_definitions": [],
                "media": {
                    "images": [],
                    "webinars": [{"title": "Shared Webinar", "url": shared_url}],
                    "videos": [],
                },
            },
            {
                "content": "Content 2",
                "score": 0.8,
                "metadata": {"title": "Article 2", "url": "https://example.com/2"},
                "entities": [],
                "glossary_definitions": [],
                "media": {
                    "images": [],
                    "webinars": [{"title": "Shared Webinar", "url": shared_url}],
                    "videos": [],
                },
            },
        ]

        result = _build_context_from_results(
            definitions=[],
            search_results=search_results,
        )

        # Should only have one webinar despite appearing in two sources
        assert len(result.resources["webinars"]) == 1
        assert result.resources["webinars"][0].url == shared_url

    def test_handles_missing_media_field(self, mock_search_results: list[dict]) -> None:
        """Verify no error when search result lacks media field."""
        # mock_search_results doesn't have media field
        result = _build_context_from_results(
            definitions=[],
            search_results=mock_search_results,
        )

        # Should have empty resource lists
        assert result.resources["images"] == []
        assert result.resources["webinars"] == []
        assert result.resources["videos"] == []

    def test_handles_empty_search_results(self) -> None:
        """Verify graceful handling of empty results."""
        result = _build_context_from_results(
            definitions=[],
            search_results=[],
        )

        assert result.sources == []
        assert result.entities == []
        assert result.context == "No relevant context found."
        assert result.resources["images"] == []

    def test_omits_empty_resource_sections_in_context(
        self, mock_search_results: list[dict]
    ) -> None:
        """Verify 'Resources from this source' omitted when none exist."""
        result = _build_context_from_results(
            definitions=[],
            search_results=mock_search_results,
        )

        # Context should not contain resource section headers when no resources
        assert "Resources from this source:" not in result.context

    def test_returns_context_build_result_type(
        self, mock_search_results_with_media: list[dict]
    ) -> None:
        """Verify function returns ContextBuildResult dataclass."""
        result = _build_context_from_results(
            definitions=[],
            search_results=mock_search_results_with_media,
        )

        assert isinstance(result, ContextBuildResult)
        assert isinstance(result.sources, list)
        assert isinstance(result.entities, list)
        assert isinstance(result.context, str)
        assert isinstance(result.resources, dict)


# =============================================================================
# Stream Chat Resource Event Tests
# =============================================================================


class TestStreamChatResourceEvents:
    """Tests for resources in stream_chat SOURCES event."""

    @pytest.mark.asyncio
    async def test_sources_event_includes_resources_structure(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        mock_search_results_with_media: list[dict],
    ) -> None:
        """Verify SOURCES event has resources.images/webinars/videos."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = mock_search_results_with_media
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

            # Check SOURCES event structure
            sources_event = events[0]
            assert sources_event.event_type == StreamEventType.SOURCES
            assert "resources" in sources_event.data
            assert "images" in sources_event.data["resources"]
            assert "webinars" in sources_event.data["resources"]
            assert "videos" in sources_event.data["resources"]

    @pytest.mark.asyncio
    async def test_sources_event_resources_have_correct_structure(
        self,
        mock_config: MagicMock,
        mock_graph: MagicMock,
        mock_vector_store: MagicMock,
        mock_search_results_with_media: list[dict],
    ) -> None:
        """Verify resource objects have title, url, alt_text, source_title."""
        with (
            patch("requirements_graphrag_api.core.generation.graph_enriched_search") as mock_search,
            patch("requirements_graphrag_api.core.generation.search_terms") as mock_terms,
            patch("requirements_graphrag_api.core.generation.ChatOpenAI") as mock_llm_class,
        ):
            mock_search.return_value = mock_search_results_with_media
            mock_terms.return_value = []
            mock_llm_class.return_value = create_llm_mock("Test answer", streaming=True)

            events = []
            async for event in stream_chat(
                mock_config,
                mock_graph,
                mock_vector_store,
                "Test question",
            ):
                events.append(event)

            resources = events[0].data["resources"]

            # Check webinar structure
            assert len(resources["webinars"]) > 0
            webinar = resources["webinars"][0]
            assert "title" in webinar
            assert "url" in webinar
            assert "source_title" in webinar

            # Check image structure
            assert len(resources["images"]) > 0
            image = resources["images"][0]
            assert "title" in image
            assert "url" in image
            assert "alt_text" in image
            assert "source_title" in image
