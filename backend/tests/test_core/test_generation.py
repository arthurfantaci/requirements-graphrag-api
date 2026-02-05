"""Tests for core generation shared types and context building."""

from __future__ import annotations

import pytest

from requirements_graphrag_api.core.generation import (
    ContextBuildResult,
    Resource,
    StreamEvent,
    StreamEventType,
    _build_context_from_results,
)

# =============================================================================
# Fixtures
# =============================================================================


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

    def test_filters_webinar_thumbnails_from_images(self) -> None:
        """Verify images matching a webinar thumbnail_url are excluded."""
        thumbnail_url = "https://img.youtube.com/vi/abc123/maxresdefault.jpg"
        search_results = [
            {
                "content": "Content about webinars",
                "score": 0.9,
                "metadata": {"title": "Article 1", "url": "https://example.com/1"},
                "entities": [],
                "glossary_definitions": [],
                "media": {
                    "images": [
                        {"url": thumbnail_url, "alt_text": "Webinar preview"},
                        {"url": "https://example.com/diagram.png", "alt_text": "Diagram"},
                    ],
                    "webinars": [
                        {
                            "title": "Best Practices Webinar",
                            "url": "https://youtube.com/watch?v=abc123",
                            "thumbnail_url": thumbnail_url,
                        },
                    ],
                    "videos": [],
                },
            },
        ]

        result = _build_context_from_results(
            definitions=[],
            search_results=search_results,
        )

        # Thumbnail image should be excluded; non-matching image preserved
        images = result.resources["images"]
        assert len(images) == 1
        assert images[0].url == "https://example.com/diagram.png"

        # Webinar should still be present
        webinars = result.resources["webinars"]
        assert len(webinars) == 1
        assert webinars[0].thumbnail_url == thumbnail_url

    def test_filters_cross_article_webinar_thumbnails(self) -> None:
        """Verify thumbnail filtering works across different search results."""
        thumbnail_url = "https://i.vimeocdn.com/video/12345_640.jpg"
        search_results = [
            {
                "content": "Article with the webinar",
                "score": 0.9,
                "metadata": {"title": "Article A", "url": "https://example.com/a"},
                "entities": [],
                "glossary_definitions": [],
                "media": {
                    "images": [],
                    "webinars": [
                        {
                            "title": "Deep Dive Webinar",
                            "url": "https://vimeo.com/12345",
                            "thumbnail_url": thumbnail_url,
                        },
                    ],
                    "videos": [],
                },
            },
            {
                "content": "Article with the matching image",
                "score": 0.85,
                "metadata": {"title": "Article B", "url": "https://example.com/b"},
                "entities": [],
                "glossary_definitions": [],
                "media": {
                    "images": [
                        {"url": thumbnail_url, "alt_text": "Vimeo thumbnail"},
                        {"url": "https://example.com/photo.png", "alt_text": "Photo"},
                    ],
                    "webinars": [],
                    "videos": [],
                },
            },
        ]

        result = _build_context_from_results(
            definitions=[],
            search_results=search_results,
        )

        # Cross-article thumbnail should be filtered out
        images = result.resources["images"]
        assert len(images) == 1
        assert images[0].url == "https://example.com/photo.png"

        # Webinar from Article A preserved
        webinars = result.resources["webinars"]
        assert len(webinars) == 1
        assert webinars[0].title == "Deep Dive Webinar"

    def test_preserves_images_when_no_webinar_thumbnails(
        self, mock_search_results_with_media: list[dict]
    ) -> None:
        """Verify all images preserved when webinars have no thumbnail_url."""
        # mock_search_results_with_media has webinars without thumbnail_url
        result = _build_context_from_results(
            definitions=[],
            search_results=mock_search_results_with_media,
        )

        # All 3 images should still be present (regression guard)
        assert len(result.resources["images"]) == 3

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
