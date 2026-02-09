"""Tests for core.context — shared context formatting utility."""

from __future__ import annotations

import pytest

from requirements_graphrag_api.core.agentic.state import EntityInfo, RetrievedDocument
from requirements_graphrag_api.core.context import (
    FormattedContext,
    NormalizedDocument,
    format_context,
    format_entity_info_for_synthesis,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def enriched_retrieved_doc() -> RetrievedDocument:
    """A RetrievedDocument with full enrichment metadata."""
    return RetrievedDocument(
        content="Requirements traceability enables tracking requirements.",
        source="Traceability Article",
        score=0.95,
        metadata={
            "title": "Traceability Article",
            "url": "https://example.com/article1",
            "chunk_id": "chunk-1",
            "entities": [
                {
                    "name": "AEC",
                    "type": "Industry",
                    "definition": "Architecture, Engineering & Construction",
                },
                {"name": "Traceability", "type": "Concept", "definition": None},
            ],
            "glossary_definitions": [
                {
                    "term": "Traceability Matrix",
                    "definition": "A document that correlates requirements to tests.",
                },
            ],
            "semantic_relationships": [
                {
                    "from_entity": "Traceability Matrix",
                    "relationship": "REQUIRES",
                    "to_entity": "Requirements Baseline",
                    "to_type": "Concept",
                },
            ],
            "industry_standards": [
                {
                    "standard": "ISO 26262",
                    "organization": "ISO",
                    "standard_definition": "Automotive functional safety",
                },
            ],
            "media": {
                "images": [
                    {"url": "https://example.com/img1.png", "alt_text": "Traceability diagram"},
                ],
                "webinars": [],
                "videos": [],
            },
        },
    )


@pytest.fixture
def enriched_raw_result() -> dict:
    """A raw dict result as returned by graph_enriched_search."""
    return {
        "content": "Best practices for V&V in requirements management.",
        "score": 0.88,
        "metadata": {
            "title": "V&V Best Practices",
            "url": "https://example.com/article2",
            "chunk_id": "chunk-2",
        },
        "entities": [
            {
                "name": "Verification",
                "type": "Concept",
                "definition": "Confirming product meets spec",
            },
            {"name": "Validation", "type": "Concept"},
        ],
        "glossary_definitions": [
            {"term": "V&V", "definition": "Verification and Validation"},
        ],
        "semantic_relationships": [
            {
                "from_entity": "ISO 26262",
                "relationship": "APPLIES_TO",
                "to_entity": "Automotive",
                "to_type": "Industry",
            },
        ],
        "industry_standards": [],
        "media": {},
    }


@pytest.fixture
def minimal_retrieved_doc() -> RetrievedDocument:
    """A RetrievedDocument with no enrichment metadata."""
    return RetrievedDocument(
        content="Simple content.",
        source="Simple Source",
        score=0.5,
    )


# =============================================================================
# NormalizedDocument tests
# =============================================================================


class TestNormalizedDocumentFromRetrievedDocument:
    """Tests for NormalizedDocument.from_retrieved_document."""

    def test_full_metadata(self, enriched_retrieved_doc: RetrievedDocument) -> None:
        """Extracts all enrichment fields from metadata."""
        doc = NormalizedDocument.from_retrieved_document(enriched_retrieved_doc)
        assert doc.content == enriched_retrieved_doc.content
        assert doc.source == "Traceability Article"
        assert doc.score == 0.95
        assert len(doc.entities) == 2
        assert doc.entities[0]["name"] == "AEC"
        assert len(doc.glossary_definitions) == 1
        assert len(doc.semantic_relationships) == 1
        assert len(doc.industry_standards) == 1
        assert doc.url == "https://example.com/article1"
        assert doc.chunk_id == "chunk-1"

    def test_empty_metadata(self, minimal_retrieved_doc: RetrievedDocument) -> None:
        """Handles missing metadata gracefully."""
        doc = NormalizedDocument.from_retrieved_document(minimal_retrieved_doc)
        assert doc.content == "Simple content."
        assert doc.entities == []
        assert doc.glossary_definitions == []
        assert doc.semantic_relationships == []
        assert doc.industry_standards == []
        assert doc.media == {}

    def test_is_frozen(self, enriched_retrieved_doc: RetrievedDocument) -> None:
        """NormalizedDocument is immutable."""
        doc = NormalizedDocument.from_retrieved_document(enriched_retrieved_doc)
        with pytest.raises(AttributeError):
            doc.content = "modified"  # type: ignore[misc]


class TestNormalizedDocumentFromRawResult:
    """Tests for NormalizedDocument.from_raw_result."""

    def test_full_raw_result(self, enriched_raw_result: dict) -> None:
        """Extracts enrichment from top-level keys."""
        doc = NormalizedDocument.from_raw_result(enriched_raw_result)
        assert doc.source == "V&V Best Practices"
        assert doc.score == 0.88
        assert len(doc.entities) == 2
        assert len(doc.glossary_definitions) == 1
        assert len(doc.semantic_relationships) == 1
        assert doc.url == "https://example.com/article2"

    def test_missing_metadata_key(self) -> None:
        """Handles result with no metadata key."""
        result = {"content": "bare content", "score": 0.5}
        doc = NormalizedDocument.from_raw_result(result)
        assert doc.source == "Unknown"
        assert doc.content == "bare content"

    def test_string_entities_coerced(self) -> None:
        """String entity lists are coerced to dicts."""
        result = {
            "content": "text",
            "score": 0.5,
            "metadata": {"title": "T"},
            "entities": ["EntityA", "EntityB"],
        }
        doc = NormalizedDocument.from_raw_result(result)
        assert len(doc.entities) == 2
        assert doc.entities[0] == {"name": "EntityA", "type": "Entity"}


class TestNormalizeRelationships:
    """Tests for _normalize_relationships handling."""

    def test_dict_to_list(self) -> None:
        """A single dict is wrapped in a list."""
        result = {
            "content": "text",
            "score": 0.5,
            "metadata": {"title": "T"},
            "semantic_relationships": {
                "from_entity": "A",
                "relationship": "REL",
                "to_entity": "B",
            },
        }
        doc = NormalizedDocument.from_raw_result(result)
        assert len(doc.semantic_relationships) == 1
        assert doc.semantic_relationships[0]["from_entity"] == "A"

    def test_none_value(self) -> None:
        """None produces empty list."""
        result = {
            "content": "text",
            "score": 0.5,
            "metadata": {"title": "T"},
            "semantic_relationships": None,
        }
        doc = NormalizedDocument.from_raw_result(result)
        assert doc.semantic_relationships == []

    def test_list_with_non_dicts_filtered(self) -> None:
        """Non-dict items in the list are filtered out."""
        result = {
            "content": "text",
            "score": 0.5,
            "metadata": {"title": "T"},
            "semantic_relationships": [
                {"from_entity": "A", "relationship": "REL", "to_entity": "B"},
                "bad_item",
            ],
        }
        doc = NormalizedDocument.from_raw_result(result)
        assert len(doc.semantic_relationships) == 1


# =============================================================================
# format_context tests
# =============================================================================


class TestFormatContext:
    """Tests for the format_context function."""

    def test_hybrid_format_produces_chunks_and_kg_section(
        self,
        enriched_retrieved_doc: RetrievedDocument,
        enriched_raw_result: dict,
    ) -> None:
        """Produces per-chunk sections + KG Context section."""
        docs = [
            NormalizedDocument.from_retrieved_document(enriched_retrieved_doc),
            NormalizedDocument.from_raw_result(enriched_raw_result),
        ]
        result = format_context(docs)

        assert isinstance(result, FormattedContext)
        # Chunk sections
        assert "[Source 1: Traceability Article]" in result.context
        assert "[Source 2: V&V Best Practices]" in result.context
        # Inline entities
        assert "(Entities: AEC, Traceability)" in result.context
        assert "(Entities: Verification, Validation)" in result.context
        # KG section
        assert "## Knowledge Graph Context" in result.context
        assert "### Glossary Definitions" in result.context
        assert "**Traceability Matrix**" in result.context
        assert "### Semantic Relationships" in result.context
        assert "Traceability Matrix -> REQUIRES -> Requirements Baseline" in result.context
        assert "### Industry Standards" in result.context
        assert "ISO 26262" in result.context

    def test_deduplicates_entities_across_chunks(self) -> None:
        """Same entity appearing in two chunks is deduplicated."""
        docs = [
            NormalizedDocument(
                content="Chunk 1",
                source="S1",
                entities=[{"name": "AEC", "type": "Industry"}],
            ),
            NormalizedDocument(
                content="Chunk 2",
                source="S2",
                entities=[{"name": "AEC", "type": "Industry", "definition": "Arch, Eng & Constr"}],
            ),
        ]
        result = format_context(docs)
        # AEC should appear once in entities_by_name with its definition
        assert "AEC" in result.entities_by_name
        assert result.entities_by_name["AEC"]["definition"] == "Arch, Eng & Constr"

    def test_deduplicates_relationships(self) -> None:
        """Same relationship from two chunks appears once in KG section."""
        rel = {
            "from_entity": "A",
            "relationship": "REQUIRES",
            "to_entity": "B",
        }
        docs = [
            NormalizedDocument(content="C1", source="S1", semantic_relationships=[rel]),
            NormalizedDocument(content="C2", source="S2", semantic_relationships=[rel]),
        ]
        result = format_context(docs)
        assert result.context.count("A -> REQUIRES -> B") == 1

    def test_empty_documents(self) -> None:
        """Empty document list returns fallback context."""
        result = format_context([])
        assert result.context == "No relevant context found."
        assert result.sources == []
        assert result.entities_by_name == {}

    def test_max_documents_cap(self) -> None:
        """Only processes up to max_documents."""
        docs = [NormalizedDocument(content=f"Chunk {i}", source=f"S{i}") for i in range(15)]
        result = format_context(docs, max_documents=3)
        assert "[Source 3:" in result.context
        assert "[Source 4:" not in result.context

    def test_sources_metadata(self, enriched_retrieved_doc: RetrievedDocument) -> None:
        """Sources list contains url, chunk_id, relevance_score."""
        docs = [NormalizedDocument.from_retrieved_document(enriched_retrieved_doc)]
        result = format_context(docs)
        assert len(result.sources) == 1
        s = result.sources[0]
        assert s["title"] == "Traceability Article"
        assert s["url"] == "https://example.com/article1"
        assert s["chunk_id"] == "chunk-1"
        assert s["relevance_score"] == 0.95

    def test_media_extracted_not_in_context(
        self, enriched_retrieved_doc: RetrievedDocument
    ) -> None:
        """Media goes to resources dict, not into the context string."""
        docs = [NormalizedDocument.from_retrieved_document(enriched_retrieved_doc)]
        result = format_context(docs)
        assert len(result.resources["images"]) == 1
        assert result.resources["images"][0].url == "https://example.com/img1.png"
        # Images should NOT appear in context string
        assert "img1.png" not in result.context

    def test_entities_str_sorted_and_capped(self) -> None:
        """entities_str is sorted alphabetically and capped at 20."""
        docs = [
            NormalizedDocument(
                content="C",
                source="S",
                entities=[{"name": f"Entity_{chr(65 + i)}", "type": "Concept"} for i in range(25)],
            )
        ]
        result = format_context(docs)
        names = result.entities_str.split(", ")
        assert len(names) == 20
        assert names == sorted(names)

    def test_with_definitions_parameter(self) -> None:
        """Definitions are prepended as definition blocks."""
        definitions = [
            {
                "term": "ALM",
                "acronym": "A.L.M.",
                "definition": "Application Lifecycle Management",
                "url": "https://example.com/alm",
                "score": 0.9,
            },
        ]
        docs = [NormalizedDocument(content="Chunk", source="S1")]
        result = format_context(docs, definitions=definitions)
        assert "[Definition: ALM (A.L.M.)]" in result.context
        assert "Application Lifecycle Management" in result.context

    def test_low_score_definitions_excluded(self) -> None:
        """Definitions below threshold are not included."""
        definitions = [
            {"term": "Low", "definition": "Low relevance", "score": 0.3},
        ]
        result = format_context([], definitions=definitions)
        assert "Low" not in result.context

    def test_includes_relationships_and_standards_new_coverage(self) -> None:
        """Semantic relationships and industry standards appear in context."""
        docs = [
            NormalizedDocument(
                content="Content about automotive safety.",
                source="Safety Article",
                semantic_relationships=[
                    {
                        "from_entity": "FMEA",
                        "relationship": "COMPONENT_OF",
                        "to_entity": "Risk Analysis",
                    },
                ],
                industry_standards=[
                    {
                        "standard": "IEC 62304",
                        "organization": "IEC",
                        "standard_definition": "Medical device software lifecycle",
                    },
                ],
            ),
        ]
        result = format_context(docs)
        assert "FMEA -> COMPONENT_OF -> Risk Analysis" in result.context
        assert "IEC 62304: Medical device software lifecycle (Organization: IEC)" in result.context

    def test_no_kg_section_when_no_enrichment(self) -> None:
        """No KG Context section when documents have no enrichment."""
        docs = [NormalizedDocument(content="Plain text", source="S1")]
        result = format_context(docs)
        assert "## Knowledge Graph Context" not in result.context
        assert "[Source 1: S1]" in result.context


# =============================================================================
# format_entity_info_for_synthesis tests
# =============================================================================


class TestFormatEntityInfoForSynthesis:
    """Tests for format_entity_info_for_synthesis."""

    def test_formats_entities(self) -> None:
        """Produces formatted entity info string."""
        entities = [
            EntityInfo(
                name="Jama Connect",
                entity_type="Tool",
                description="Requirements management tool",
                related_entities=["DOORS", "Polarion"],
                mentioned_in=["Article A", "Article B"],
            ),
            EntityInfo(
                name="Traceability",
                entity_type="Concept",
                description="Ability to trace requirements",
            ),
        ]
        result = format_entity_info_for_synthesis(entities)
        assert "**Jama Connect** (Tool)" in result
        assert "Requirements management tool" in result
        assert "Related: DOORS, Polarion" in result
        assert "Mentioned in: Article A, Article B" in result
        assert "**Traceability** (Concept)" in result

    def test_empty_list(self) -> None:
        """Returns empty string for empty list."""
        assert format_entity_info_for_synthesis([]) == ""

    def test_entity_without_optional_fields(self) -> None:
        """Handles entity with only required fields."""
        entities = [EntityInfo(name="Test", entity_type="Concept")]
        result = format_entity_info_for_synthesis(entities)
        assert "**Test** (Concept)" in result
        assert "Related:" not in result
        assert "Mentioned in:" not in result
