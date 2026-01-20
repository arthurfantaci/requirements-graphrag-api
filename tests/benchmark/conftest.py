"""Pytest fixtures for benchmark tests.

Provides shared fixtures for evaluation testing including:
- Mock configurations
- Dataset fixtures
- Metric assertions
- Performance benchmarking utilities
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.benchmark.generator import generate_evaluation_dataset
from tests.benchmark.golden_dataset import GOLDEN_DATASET, get_must_pass_examples
from tests.benchmark.schemas import (
    BenchmarkExample,
    DifficultyLevel,
    QueryCategory,
)

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock AppConfig for testing."""
    config = MagicMock()
    config.neo4j_uri = "neo4j://localhost:7687"
    config.neo4j_username = "neo4j"
    config.neo4j_password = "test"
    config.neo4j_database = "neo4j"
    config.openai_api_key = "sk-test"
    config.chat_model = "gpt-4o"
    config.embedding_model = "text-embedding-3-small"
    config.vector_index_name = "chunk_embeddings"
    config.similarity_k = 6
    config.langsmith_tracing_enabled = False
    return config


@pytest.fixture
def mock_driver() -> MagicMock:
    """Create a mock Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=None)
    return driver


@pytest.fixture
def mock_retriever() -> MagicMock:
    """Create a mock VectorRetriever."""
    retriever = MagicMock()
    retriever.search = AsyncMock(return_value=[])
    return retriever


# =============================================================================
# DATASET FIXTURES
# =============================================================================


@pytest.fixture
def golden_dataset() -> list[BenchmarkExample]:
    """Return the complete golden dataset."""
    return GOLDEN_DATASET.copy()


@pytest.fixture
def must_pass_examples() -> list[BenchmarkExample]:
    """Return only must-pass examples for critical testing."""
    return get_must_pass_examples()


@pytest.fixture
def smoke_test_dataset() -> list[BenchmarkExample]:
    """Return a small dataset for smoke testing (10 examples)."""
    # Select representative examples across categories
    smoke_examples = []
    seen_categories: set[QueryCategory] = set()

    for example in GOLDEN_DATASET:
        if example.category not in seen_categories:
            smoke_examples.append(example)
            seen_categories.add(example.category)
        if len(smoke_examples) >= 10:
            break

    return smoke_examples


@pytest.fixture
def generated_dataset() -> list[BenchmarkExample]:
    """Return a generated dataset (50 examples for unit tests)."""
    return generate_evaluation_dataset(total_examples=50, seed=42)


@pytest.fixture
def full_benchmark_dataset() -> list[BenchmarkExample]:
    """Return the full benchmark dataset (250 examples)."""
    return generate_evaluation_dataset(total_examples=250, seed=42)


@pytest.fixture
def definitional_examples() -> list[BenchmarkExample]:
    """Return only definitional examples."""
    return [ex for ex in GOLDEN_DATASET if ex.category == QueryCategory.DEFINITIONAL]


@pytest.fixture
def edge_case_examples() -> list[BenchmarkExample]:
    """Return only edge case examples."""
    return [ex for ex in GOLDEN_DATASET if ex.category == QueryCategory.EDGE_CASE]


@pytest.fixture
def standards_examples() -> list[BenchmarkExample]:
    """Return examples related to standards."""
    return [ex for ex in GOLDEN_DATASET if "standards" in ex.tags]


# =============================================================================
# METRIC ASSERTION HELPERS
# =============================================================================


class MetricAssertions:
    """Helper class for metric-based assertions."""

    @staticmethod
    def assert_above_threshold(
        actual: float,
        threshold: float,
        metric_name: str,
        example_id: str,
    ) -> None:
        """Assert that a metric is above a threshold."""
        assert actual >= threshold, (
            f"{metric_name} for {example_id}: {actual:.3f} < {threshold:.3f}"
        )

    @staticmethod
    def assert_metrics_pass(
        example: BenchmarkExample,
        metrics: dict[str, float],
    ) -> list[str]:
        """Check if metrics pass thresholds, return list of failures."""
        failures = []
        expected = example.expected_metrics

        if metrics.get("faithfulness", 0) < expected.min_faithfulness:
            failures.append(
                f"faithfulness: {metrics.get('faithfulness', 0):.3f} < {expected.min_faithfulness}"
            )

        if metrics.get("answer_relevancy", 0) < expected.min_relevancy:
            score = metrics.get("answer_relevancy", 0)
            failures.append(f"answer_relevancy: {score:.3f} < {expected.min_relevancy}")

        if metrics.get("context_precision", 0) < expected.min_precision:
            score = metrics.get("context_precision", 0)
            failures.append(f"context_precision: {score:.3f} < {expected.min_precision}")

        if metrics.get("context_recall", 0) < expected.min_recall:
            failures.append(
                f"context_recall: {metrics.get('context_recall', 0):.3f} < {expected.min_recall}"
            )

        return failures


@pytest.fixture
def metric_assertions() -> MetricAssertions:
    """Return metric assertion helper."""
    return MetricAssertions()


# =============================================================================
# MOCK RESPONSE FIXTURES
# =============================================================================


@pytest.fixture
def mock_chat_response() -> dict[str, Any]:
    """Create a mock chat response."""
    return {
        "answer": "Requirements traceability is the ability to trace requirements...",
        "sources": [
            {
                "title": "Requirements Traceability Guide",
                "url": "https://example.com/traceability",
                "content": "Detailed content about traceability...",
                "relevance_score": 0.95,
            },
        ],
        "entities": [
            {"name": "traceability", "type": "Concept"},
        ],
    }


@pytest.fixture
def mock_routing_response() -> dict[str, Any]:
    """Create a mock routing response."""
    return {
        "selected_tools": ["graphrag_vector_search"],
        "reasoning": "This is a definitional question about a core concept.",
        "confidence": 0.9,
    }


# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================


@pytest.fixture
def performance_tracker() -> Generator[dict[str, list[float]], None, None]:
    """Track performance metrics across tests."""
    tracker: dict[str, list[float]] = {
        "latencies": [],
        "token_counts": [],
    }
    yield tracker

    # Summary statistics after tests
    if tracker["latencies"]:
        avg_latency = sum(tracker["latencies"]) / len(tracker["latencies"])
        max_latency = max(tracker["latencies"])
        print("\nPerformance Summary:")
        print(f"  Avg Latency: {avg_latency:.2f}ms")
        print(f"  Max Latency: {max_latency:.2f}ms")
        print(f"  Total Queries: {len(tracker['latencies'])}")


# =============================================================================
# CATEGORY PARAMETERIZATION
# =============================================================================


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate parameterized tests for benchmark examples."""
    # Parameterize tests that need all golden examples
    if "golden_example" in metafunc.fixturenames:
        metafunc.parametrize(
            "golden_example",
            GOLDEN_DATASET,
            ids=[ex.id for ex in GOLDEN_DATASET],
        )

    # Parameterize tests by difficulty
    if "difficulty_level" in metafunc.fixturenames:
        metafunc.parametrize(
            "difficulty_level",
            list(DifficultyLevel),
            ids=[d.value for d in DifficultyLevel],
        )

    # Parameterize tests by category
    if "query_category" in metafunc.fixturenames:
        metafunc.parametrize(
            "query_category",
            list(QueryCategory),
            ids=[c.value for c in QueryCategory],
        )


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================


@pytest.fixture
def benchmark_config() -> dict[str, Any]:
    """Configuration for pytest-benchmark."""
    return {
        "min_rounds": 3,
        "max_time": 30,
        "warmup": True,
        "warmup_iterations": 1,
    }


# =============================================================================
# SKIP MARKERS
# =============================================================================


requires_neo4j = pytest.mark.skipif(
    True,  # Will be replaced with actual connection check
    reason="Requires Neo4j connection",
)

requires_openai = pytest.mark.skipif(
    True,  # Will be replaced with actual API key check
    reason="Requires OpenAI API key",
)

requires_langsmith = pytest.mark.skipif(
    True,  # Will be replaced with actual API key check
    reason="Requires LangSmith API key",
)


__all__ = [
    "MetricAssertions",
    "requires_langsmith",
    "requires_neo4j",
    "requires_openai",
]
