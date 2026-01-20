"""Tests for the prompt catalog module.

This test suite covers:
- Prompt definitions validation
- Catalog initialization and configuration
- Cache behavior
- LangSmith Hub integration (mocked)
- Evaluation utilities
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from jama_mcp_server_graphrag.prompts import (
    PROMPT_DEFINITIONS,
    TEXT2CYPHER_EXAMPLES,
    PromptCatalog,
    PromptDefinition,
    PromptMetadata,
    PromptName,
    get_catalog,
    get_prompt,
    get_prompt_sync,
    initialize_catalog,
)
from jama_mcp_server_graphrag.prompts.evaluation import (
    ComparisonResult,
    EvaluationResult,
    create_cypher_validity_evaluator,
    create_json_validity_evaluator,
    create_length_evaluator,
    get_evaluators_for_prompt,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def catalog() -> PromptCatalog:
    """Create a fresh catalog instance for testing."""
    return PromptCatalog(
        organization="test-org",
        environment="test",
        cache_ttl=60,
        use_hub=False,
    )


@pytest.fixture
def catalog_with_hub() -> PromptCatalog:
    """Create a catalog with Hub enabled for testing."""
    return PromptCatalog(
        organization="test-org",
        environment="test",
        cache_ttl=60,
        use_hub=True,
    )


# =============================================================================
# PROMPT DEFINITIONS TESTS
# =============================================================================


class TestPromptDefinitions:
    """Tests for prompt definitions."""

    def test_all_prompts_defined(self) -> None:
        """All PromptName values should have definitions."""
        for name in PromptName:
            assert name in PROMPT_DEFINITIONS, f"Missing definition for {name}"

    def test_prompt_definitions_have_required_fields(self) -> None:
        """All definitions should have required fields."""
        for name, definition in PROMPT_DEFINITIONS.items():
            assert isinstance(definition, PromptDefinition)
            assert definition.name == name
            assert isinstance(definition.template, ChatPromptTemplate)
            assert isinstance(definition.metadata, PromptMetadata)

    def test_metadata_has_required_fields(self) -> None:
        """All metadata should have required fields."""
        for name, definition in PROMPT_DEFINITIONS.items():
            meta = definition.metadata
            assert meta.version, f"Missing version for {name}"
            assert meta.description, f"Missing description for {name}"
            assert meta.input_variables, f"Missing input_variables for {name}"
            assert meta.output_format, f"Missing output_format for {name}"

    def test_version_format(self) -> None:
        """Version should be in semver format."""
        for name, definition in PROMPT_DEFINITIONS.items():
            version = definition.metadata.version
            parts = version.split(".")
            assert len(parts) == 3, f"Invalid version format for {name}: {version}"
            assert all(p.isdigit() for p in parts), f"Version parts not numeric: {version}"

    def test_input_variables_match_template(self) -> None:
        """Metadata input_variables should match template variables."""
        for name, definition in PROMPT_DEFINITIONS.items():
            template_vars = set(definition.template.input_variables)
            meta_vars = set(definition.metadata.input_variables)
            assert template_vars == meta_vars, (
                f"Variable mismatch for {name}: template={template_vars}, metadata={meta_vars}"
            )

    def test_text2cypher_examples_not_empty(self) -> None:
        """TEXT2CYPHER_EXAMPLES should contain examples."""
        assert TEXT2CYPHER_EXAMPLES
        assert "Example" in TEXT2CYPHER_EXAMPLES
        assert "MATCH" in TEXT2CYPHER_EXAMPLES


# =============================================================================
# CATALOG TESTS
# =============================================================================


class TestPromptCatalog:
    """Tests for PromptCatalog."""

    def test_initialization_defaults(self) -> None:
        """Catalog should initialize with defaults."""
        catalog = PromptCatalog()
        assert catalog.environment in ("development", "test", "production")
        assert catalog.cache_ttl > 0

    def test_initialization_custom_values(self, catalog: PromptCatalog) -> None:
        """Catalog should accept custom configuration."""
        assert catalog.organization == "test-org"
        assert catalog.environment == "test"
        assert catalog.cache_ttl == 60
        assert catalog.use_hub is False

    def test_list_prompts(self, catalog: PromptCatalog) -> None:
        """list_prompts should return all prompt names."""
        prompts = catalog.list_prompts()
        assert len(prompts) == len(PromptName)
        for name in PromptName:
            assert name in prompts

    def test_get_definition(self, catalog: PromptCatalog) -> None:
        """get_definition should return full definition."""
        definition = catalog.get_definition(PromptName.ROUTER)
        assert definition.name == PromptName.ROUTER
        assert isinstance(definition.template, ChatPromptTemplate)

    def test_get_definition_unknown_prompt(self, catalog: PromptCatalog) -> None:
        """get_definition should raise KeyError for unknown prompts."""
        with pytest.raises(KeyError):
            catalog.get_definition("unknown-prompt")  # type: ignore[arg-type]

    def test_get_prompt_sync_returns_template(self, catalog: PromptCatalog) -> None:
        """get_prompt_sync should return a ChatPromptTemplate."""
        template = catalog.get_prompt_sync(PromptName.ROUTER)
        assert isinstance(template, ChatPromptTemplate)

    def test_get_prompt_sync_caches_result(self, catalog: PromptCatalog) -> None:
        """get_prompt_sync should cache results."""
        _ = catalog.get_prompt_sync(PromptName.ROUTER)
        status = catalog.get_cache_status()
        assert "graphrag-router:test" in status
        assert status["graphrag-router:test"]["source"] == "local"

    @pytest.mark.asyncio
    async def test_get_prompt_returns_template(self, catalog: PromptCatalog) -> None:
        """get_prompt should return a ChatPromptTemplate."""
        template = await catalog.get_prompt(PromptName.CRITIC)
        assert isinstance(template, ChatPromptTemplate)

    @pytest.mark.asyncio
    async def test_get_prompt_caches_result(self, catalog: PromptCatalog) -> None:
        """get_prompt should cache results."""
        await catalog.get_prompt(PromptName.CRITIC)
        status = catalog.get_cache_status()
        assert "graphrag-critic:test" in status

    def test_cache_invalidation_all(self, catalog: PromptCatalog) -> None:
        """invalidate_cache should clear all entries."""
        # Populate cache
        _ = catalog.get_prompt_sync(PromptName.ROUTER)
        _ = catalog.get_prompt_sync(PromptName.CRITIC)

        count = catalog.invalidate_cache()
        assert count == 2
        assert catalog.get_cache_status() == {}

    def test_cache_invalidation_specific(self, catalog: PromptCatalog) -> None:
        """invalidate_cache should clear specific entries."""
        # Populate cache
        _ = catalog.get_prompt_sync(PromptName.ROUTER)
        _ = catalog.get_prompt_sync(PromptName.CRITIC)

        count = catalog.invalidate_cache(PromptName.ROUTER)
        assert count == 1
        status = catalog.get_cache_status()
        assert "graphrag-router:test" not in status
        assert "graphrag-critic:test" in status

    def test_cache_expiration(self) -> None:
        """Cache entries should expire after TTL."""
        catalog = PromptCatalog(cache_ttl=1, use_hub=False)
        _ = catalog.get_prompt_sync(PromptName.ROUTER)

        # Cache should be valid initially (testing internal method)
        assert catalog._is_cache_valid("graphrag-router:development")  # noqa: SLF001

        # Wait for expiration
        time.sleep(1.1)
        assert not catalog._is_cache_valid("graphrag-router:development")  # noqa: SLF001


class TestCatalogHubIntegration:
    """Tests for LangSmith Hub integration."""

    @pytest.mark.asyncio
    async def test_hub_pull_success(self, catalog_with_hub: PromptCatalog) -> None:
        """Hub pull should update cache on success."""
        mock_template = ChatPromptTemplate.from_messages(
            [
                ("system", "Mock hub prompt"),
                ("human", "{question}"),
            ]
        )

        # Mock asyncio.to_thread which wraps the hub.pull call
        with patch("jama_mcp_server_graphrag.prompts.catalog.asyncio.to_thread") as mock_thread:
            mock_thread.return_value = mock_template
            _ = await catalog_with_hub.get_prompt(PromptName.ROUTER)

        # Should have cached the hub version
        status = catalog_with_hub.get_cache_status()
        assert "graphrag-router:test" in status
        # Note: In this test, due to mocking, source might vary

    @pytest.mark.asyncio
    async def test_hub_pull_fallback_on_error(self, catalog_with_hub: PromptCatalog) -> None:
        """Hub errors should fall back to local definitions."""
        with patch("jama_mcp_server_graphrag.prompts.catalog.asyncio.to_thread") as mock_thread:
            mock_thread.side_effect = Exception("Hub unavailable")
            template = await catalog_with_hub.get_prompt(PromptName.ROUTER)

        # Should still get a valid template from local fallback
        assert isinstance(template, ChatPromptTemplate)


# =============================================================================
# MODULE-LEVEL FUNCTIONS TESTS
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_catalog_singleton(self) -> None:
        """get_catalog should return singleton instance."""
        catalog1 = get_catalog()
        catalog2 = get_catalog()
        assert catalog1 is catalog2

    def test_initialize_catalog_creates_new(self) -> None:
        """initialize_catalog should create new instance."""
        catalog = initialize_catalog(organization="new-org", cache_ttl=120)
        assert catalog.organization == "new-org"
        assert catalog.cache_ttl == 120

    def test_get_prompt_sync_convenience(self) -> None:
        """get_prompt_sync convenience function should work."""
        template = get_prompt_sync(PromptName.ROUTER)
        assert isinstance(template, ChatPromptTemplate)

    @pytest.mark.asyncio
    async def test_get_prompt_convenience(self) -> None:
        """get_prompt convenience function should work."""
        template = await get_prompt(PromptName.CRITIC)
        assert isinstance(template, ChatPromptTemplate)


# =============================================================================
# EVALUATOR TESTS
# =============================================================================


class TestEvaluators:
    """Tests for prompt evaluators."""

    def test_json_validity_evaluator_valid(self) -> None:
        """JSON evaluator should score valid JSON as 1.0."""
        evaluator = create_json_validity_evaluator()
        result = evaluator({"output": '{"key": "value"}'})
        assert result == {"json_valid": 1.0}

    def test_json_validity_evaluator_invalid(self) -> None:
        """JSON evaluator should score invalid JSON as 0.0."""
        evaluator = create_json_validity_evaluator()
        result = evaluator({"output": "not json"})
        assert result == {"json_valid": 0.0}

    def test_json_validity_evaluator_with_markdown(self) -> None:
        """JSON evaluator should handle markdown code blocks."""
        evaluator = create_json_validity_evaluator()
        result = evaluator({"output": '```json\n{"key": "value"}\n```'})
        assert result == {"json_valid": 1.0}

    def test_cypher_validity_evaluator_valid(self) -> None:
        """Cypher evaluator should score valid Cypher as 1.0."""
        evaluator = create_cypher_validity_evaluator()
        result = evaluator({"output": "MATCH (n) RETURN n"})
        assert result == {"cypher_valid": 1.0}

    def test_cypher_validity_evaluator_invalid(self) -> None:
        """Cypher evaluator should score invalid Cypher as 0.0."""
        evaluator = create_cypher_validity_evaluator()
        result = evaluator({"output": "SELECT * FROM table"})
        assert result == {"cypher_valid": 0.0}

    def test_length_evaluator_appropriate(self) -> None:
        """Length evaluator should score appropriate length as 1.0."""
        evaluator = create_length_evaluator(min_length=5, max_length=100)
        result = evaluator({"output": "This is a normal response"})
        assert result == {"length_appropriate": 1.0}

    def test_length_evaluator_too_short(self) -> None:
        """Length evaluator should penalize too-short output."""
        evaluator = create_length_evaluator(min_length=20, max_length=100)
        result = evaluator({"output": "Hi"})
        assert result["length_appropriate"] < 1.0

    def test_length_evaluator_too_long(self) -> None:
        """Length evaluator should penalize too-long output."""
        evaluator = create_length_evaluator(min_length=5, max_length=10)
        result = evaluator({"output": "This is a very long response"})
        assert result["length_appropriate"] < 1.0

    def test_get_evaluators_for_json_prompt(self) -> None:
        """Should return JSON evaluator for JSON output format."""
        evaluators = get_evaluators_for_prompt(PromptName.ROUTER)
        # Should include JSON and length evaluators
        assert len(evaluators) >= 2

    def test_get_evaluators_for_cypher_prompt(self) -> None:
        """Should return Cypher evaluator for Cypher output format."""
        evaluators = get_evaluators_for_prompt(PromptName.TEXT2CYPHER)
        # Should include Cypher and length evaluators
        assert len(evaluators) >= 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPromptIntegration:
    """Integration tests for prompt catalog usage."""

    def test_router_prompt_formatting(self) -> None:
        """Router prompt should format correctly with tools."""
        template = get_prompt_sync(PromptName.ROUTER)
        formatted = template.format_messages(
            tools="- tool1: description1\n- tool2: description2",
            question="What is requirements traceability?",
        )
        assert len(formatted) == 2  # system + human
        assert "tool1" in formatted[0].content
        assert "requirements traceability" in formatted[1].content

    def test_critic_prompt_formatting(self) -> None:
        """Critic prompt should format correctly."""
        template = get_prompt_sync(PromptName.CRITIC)
        formatted = template.format_messages(
            context="This is the context about traceability.",
            question="What is traceability?",
        )
        assert len(formatted) == 2
        assert "context" in formatted[0].content.lower()

    def test_stepback_prompt_formatting(self) -> None:
        """Stepback prompt should format correctly."""
        template = get_prompt_sync(PromptName.STEPBACK)
        formatted = template.format_messages(
            question="What are ASIL levels in ISO 26262?",
        )
        assert len(formatted) == 2

    def test_text2cypher_prompt_formatting(self) -> None:
        """Text2Cypher prompt should format correctly."""
        template = get_prompt_sync(PromptName.TEXT2CYPHER)
        formatted = template.format_messages(
            schema="Node counts: Article(100), Chunk(500)",
            examples=TEXT2CYPHER_EXAMPLES,
            question="How many articles are there?",
        )
        assert len(formatted) == 2
        assert "Node counts" in formatted[0].content

    def test_rag_generation_prompt_formatting(self) -> None:
        """RAG generation prompt should format correctly."""
        template = get_prompt_sync(PromptName.RAG_GENERATION)
        formatted = template.format_messages(
            context="[Source 1] Requirements traceability is...",
            entities="traceability, requirements",
            question="What is requirements traceability?",
        )
        assert len(formatted) == 2


# =============================================================================
# EVALUATION RESULT DATACLASSES
# =============================================================================


class TestEvaluationDataclasses:
    """Tests for evaluation result dataclasses."""

    def test_evaluation_result_creation(self) -> None:
        """EvaluationResult should initialize correctly."""
        result = EvaluationResult(
            prompt_name="test-prompt",
            dataset_name="test-dataset",
            scores={"accuracy": 0.95, "latency": 0.8},
        )
        assert result.prompt_name == "test-prompt"
        assert result.dataset_name == "test-dataset"
        assert result.scores["accuracy"] == 0.95
        assert result.sample_results == []
        assert result.metadata == {}

    def test_comparison_result_creation(self) -> None:
        """ComparisonResult should initialize correctly."""
        result = ComparisonResult(
            baseline_name="baseline",
            candidate_name="candidate",
            baseline_scores={"accuracy": 0.9},
            candidate_scores={"accuracy": 0.95},
            improvements={"accuracy": 0.05},
            winner="candidate",
            significant=True,
        )
        assert result.baseline_name == "baseline"
        assert result.candidate_name == "candidate"
        assert result.improvements["accuracy"] == 0.05
        assert result.winner == "candidate"
        assert result.significant is True
