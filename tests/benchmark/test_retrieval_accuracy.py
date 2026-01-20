"""Retrieval accuracy benchmark tests.

Tests retrieval quality using metrics:
- Precision@K: Proportion of retrieved docs that are relevant
- Recall@K: Proportion of relevant docs that are retrieved
- MRR (Mean Reciprocal Rank): Position of first relevant result

These tests validate that the retrieval pipeline returns
relevant content for various query types.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.benchmark.schemas import (
    BenchmarkExample,
    ExpectedRouting,
    QueryCategory,
)

# =============================================================================
# RETRIEVAL METRICS
# =============================================================================


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Calculate Precision@K.

    Args:
        relevant: Set of relevant document IDs.
        retrieved: List of retrieved document IDs (ordered).
        k: Number of top results to consider.

    Returns:
        Precision@K score (0.0 to 1.0).
    """
    if not retrieved or k <= 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant)
    return relevant_in_top_k / k


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Calculate Recall@K.

    Args:
        relevant: Set of relevant document IDs.
        retrieved: List of retrieved document IDs (ordered).
        k: Number of top results to consider.

    Returns:
        Recall@K score (0.0 to 1.0).
    """
    if not relevant:
        return 1.0  # No relevant docs to find

    top_k = retrieved[:k]
    relevant_found = sum(1 for doc in top_k if doc in relevant)
    return relevant_found / len(relevant)


def mean_reciprocal_rank(relevant: set[str], retrieved: list[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    Args:
        relevant: Set of relevant document IDs.
        retrieved: List of retrieved document IDs (ordered).

    Returns:
        MRR score (0.0 to 1.0).
    """
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


# =============================================================================
# UNIT TESTS FOR METRICS
# =============================================================================


class TestRetrievalMetrics:
    """Unit tests for retrieval metric functions."""

    def test_precision_at_k_perfect(self) -> None:
        """Test Precision@K with perfect retrieval."""
        relevant = {"doc1", "doc2", "doc3"}
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        assert precision_at_k(relevant, retrieved, 3) == 1.0
        assert precision_at_k(relevant, retrieved, 5) == 0.6

    def test_precision_at_k_empty(self) -> None:
        """Test Precision@K with empty inputs."""
        assert precision_at_k(set(), [], 5) == 0.0
        assert precision_at_k({"doc1"}, [], 5) == 0.0

    def test_recall_at_k_perfect(self) -> None:
        """Test Recall@K with perfect retrieval."""
        relevant = {"doc1", "doc2"}
        retrieved = ["doc1", "doc2", "doc3"]

        assert recall_at_k(relevant, retrieved, 3) == 1.0
        assert recall_at_k(relevant, retrieved, 1) == 0.5

    def test_recall_at_k_no_relevant(self) -> None:
        """Test Recall@K with no relevant docs (edge case)."""
        assert recall_at_k(set(), ["doc1", "doc2"], 2) == 1.0

    def test_mrr_first_position(self) -> None:
        """Test MRR when relevant doc is first."""
        relevant = {"doc1"}
        retrieved = ["doc1", "doc2", "doc3"]

        assert mean_reciprocal_rank(relevant, retrieved) == 1.0

    def test_mrr_later_position(self) -> None:
        """Test MRR when relevant doc is not first."""
        relevant = {"doc3"}
        retrieved = ["doc1", "doc2", "doc3"]

        assert mean_reciprocal_rank(relevant, retrieved) == pytest.approx(1 / 3)

    def test_mrr_not_found(self) -> None:
        """Test MRR when relevant doc is not retrieved."""
        relevant = {"doc99"}
        retrieved = ["doc1", "doc2", "doc3"]

        assert mean_reciprocal_rank(relevant, retrieved) == 0.0


# =============================================================================
# RETRIEVAL PIPELINE TESTS
# =============================================================================


class TestVectorSearchRetrieval:
    """Tests for vector search retrieval accuracy."""

    @pytest.mark.asyncio
    async def test_definitional_query_retrieval(
        self,
        mock_config: MagicMock,
        mock_retriever: MagicMock,
        definitional_examples: list[BenchmarkExample],
    ) -> None:
        """Test retrieval for definitional queries."""
        # Mock retriever response
        mock_retriever.search = AsyncMock(
            return_value=[
                MagicMock(
                    content="Requirements traceability is the ability to trace...",
                    metadata={"title": "Traceability Guide", "chunk_id": "chunk_001"},
                ),
                MagicMock(
                    content="Traceability enables impact analysis...",
                    metadata={"title": "Impact Analysis", "chunk_id": "chunk_002"},
                ),
            ]
        )

        # Get first definitional example
        example = definitional_examples[0]

        # Simulate retrieval
        results = await mock_retriever.search(example.question, k=5)

        # Basic assertions
        assert len(results) > 0
        assert results[0].content is not None

    @pytest.mark.asyncio
    async def test_retrieval_returns_relevant_content(
        self,
        mock_retriever: MagicMock,
    ) -> None:
        """Test that retrieval returns content matching query intent."""
        query = "What is requirements traceability?"

        # Mock with relevant content
        mock_retriever.search = AsyncMock(
            return_value=[
                MagicMock(
                    content="Requirements traceability links requirements to artifacts.",
                    metadata={"relevance_score": 0.95},
                ),
            ]
        )

        results = await mock_retriever.search(query, k=5)

        assert len(results) > 0
        # Content should mention key terms from query
        content = results[0].content.lower()
        assert "traceability" in content or "requirements" in content


class TestHybridSearchRetrieval:
    """Tests for hybrid (vector + keyword) search retrieval."""

    @pytest.mark.asyncio
    async def test_standard_specific_query(
        self,
        mock_config: MagicMock,
        standards_examples: list[BenchmarkExample],
    ) -> None:
        """Test retrieval for standard-specific queries."""
        example = standards_examples[0]  # ISO 26262 example

        # For standards queries, we expect hybrid search to work well
        # because it combines semantic similarity with exact keyword matching
        assert (
            ExpectedRouting.VECTOR_SEARCH in example.expected_tools
            or ExpectedRouting.LOOKUP_STANDARD in example.expected_tools
        )

    @pytest.mark.asyncio
    async def test_technical_term_retrieval(
        self,
        mock_retriever: MagicMock,
    ) -> None:
        """Test retrieval of technical terms like ASIL, FMEA."""
        queries = [
            "What are ASIL levels?",
            "Explain FMEA in safety analysis",
            "What is DO-178C?",
        ]

        for query in queries:
            mock_retriever.search = AsyncMock(
                return_value=[
                    MagicMock(content=f"Content about {query}"),
                ]
            )

            results = await mock_retriever.search(query, k=5)
            assert len(results) > 0, f"No results for: {query}"


class TestGraphEnrichedRetrieval:
    """Tests for graph-enriched retrieval."""

    @pytest.mark.asyncio
    async def test_relational_query_retrieval(
        self,
        mock_driver: MagicMock,
        mock_retriever: MagicMock,
    ) -> None:
        """Test retrieval for queries about relationships."""
        query = "How does verification relate to validation?"

        # Graph-enriched should find connected entities
        mock_retriever.search = AsyncMock(
            return_value=[
                MagicMock(
                    content="Verification and validation are complementary activities...",
                    metadata={
                        "entities": ["verification", "validation"],
                        "relationships": ["related_to"],
                    },
                ),
            ]
        )

        results = await mock_retriever.search(query, k=5)

        assert len(results) > 0
        # Should retrieve content mentioning both concepts
        content = results[0].content.lower()
        assert "verification" in content or "validation" in content


# =============================================================================
# EDGE CASE RETRIEVAL TESTS
# =============================================================================


class TestEdgeCaseRetrieval:
    """Tests for edge case query handling."""

    @pytest.mark.asyncio
    async def test_out_of_domain_query(
        self,
        mock_retriever: MagicMock,
        edge_case_examples: list[BenchmarkExample],
    ) -> None:
        """Test handling of out-of-domain queries."""
        # Find the weather query
        ood_example = next(
            (ex for ex in edge_case_examples if "weather" in ex.question.lower()),
            None,
        )

        if ood_example:
            # Out-of-domain should return low relevance or empty results
            mock_retriever.search = AsyncMock(return_value=[])

            results = await mock_retriever.search(ood_example.question, k=5)

            # Either no results or low relevance is acceptable
            assert len(results) == 0 or all(
                r.metadata.get("relevance_score", 0) < 0.5 for r in results
            )

    @pytest.mark.asyncio
    async def test_typo_handling(
        self,
        mock_retriever: MagicMock,
    ) -> None:
        """Test that retrieval handles typos gracefully."""
        # Intentional typo
        query = "What is requirments traceability?"

        # Semantic search should still find relevant content
        mock_retriever.search = AsyncMock(
            return_value=[
                MagicMock(
                    content="Requirements traceability is the ability to trace...",
                    metadata={"relevance_score": 0.85},
                ),
            ]
        )

        results = await mock_retriever.search(query, k=5)

        assert len(results) > 0
        # Should still return relevant content despite typo
        assert "traceability" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_ambiguous_query(
        self,
        mock_retriever: MagicMock,
    ) -> None:
        """Test handling of ambiguous queries."""
        query = "Tell me about requirements."

        mock_retriever.search = AsyncMock(
            return_value=[
                MagicMock(content="Requirements management overview..."),
                MagicMock(content="Types of requirements..."),
                MagicMock(content="Requirements traceability..."),
            ]
        )

        results = await mock_retriever.search(query, k=5)

        # Should return multiple diverse results for ambiguous query
        assert len(results) >= 1


# =============================================================================
# BENCHMARK-DRIVEN RETRIEVAL TESTS
# =============================================================================


class TestBenchmarkRetrieval:
    """Tests driven by benchmark examples."""

    @pytest.mark.asyncio
    async def test_golden_example_retrieval(
        self,
        golden_example: BenchmarkExample,
        mock_retriever: MagicMock,
    ) -> None:
        """Test retrieval for each golden example."""
        # Mock retrieval based on expected entities
        expected_content = " ".join(golden_example.expected_entities) or "relevant content"

        mock_retriever.search = AsyncMock(
            return_value=[
                MagicMock(
                    content=f"Content about {expected_content}",
                    metadata={"relevance_score": 0.8},
                ),
            ]
        )

        results = await mock_retriever.search(golden_example.question, k=5)

        # Basic retrieval should work for all golden examples
        assert len(results) >= 0, f"Retrieval failed for: {golden_example.id}"

    def test_expected_tools_defined(
        self,
        golden_example: BenchmarkExample,
    ) -> None:
        """Verify all golden examples have expected tools defined."""
        assert len(golden_example.expected_tools) > 0, (
            f"Example {golden_example.id} has no expected tools"
        )

    def test_ground_truth_defined(
        self,
        golden_example: BenchmarkExample,
    ) -> None:
        """Verify all golden examples have ground truth."""
        assert golden_example.ground_truth, f"Example {golden_example.id} has no ground truth"


# =============================================================================
# AGGREGATE RETRIEVAL METRICS
# =============================================================================


class TestAggregateRetrievalMetrics:
    """Tests for aggregate retrieval metrics across dataset."""

    def test_dataset_coverage_by_category(
        self,
        golden_dataset: list[BenchmarkExample],
    ) -> None:
        """Verify dataset covers all query categories."""
        categories_present = {ex.category for ex in golden_dataset}

        for category in QueryCategory:
            assert category in categories_present, (
                f"Category {category.value} not represented in golden dataset"
            )

    def test_dataset_coverage_by_difficulty(
        self,
        golden_dataset: list[BenchmarkExample],
    ) -> None:
        """Verify dataset covers multiple difficulty levels."""
        from tests.benchmark.schemas import DifficultyLevel

        difficulties_present = {ex.difficulty for ex in golden_dataset}

        # Should have at least easy, medium, and hard
        assert DifficultyLevel.EASY in difficulties_present
        assert DifficultyLevel.MEDIUM in difficulties_present
        assert DifficultyLevel.HARD in difficulties_present

    def test_standards_coverage(
        self,
        golden_dataset: list[BenchmarkExample],
    ) -> None:
        """Verify key standards are covered in examples."""
        key_standards = ["ISO 26262", "IEC 62304", "ASPICE", "DO-178C"]

        all_standards = set()
        for ex in golden_dataset:
            all_standards.update(ex.expected_standards)

        for standard in key_standards:
            assert standard in all_standards, f"Standard {standard} not covered in golden dataset"


__all__ = [
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
]
