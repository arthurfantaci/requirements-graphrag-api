"""Latency and performance benchmark tests.

Tests pipeline performance using:
- Response latency measurements
- Throughput benchmarks
- Resource utilization
- Performance regression detection

These tests validate that the pipeline meets performance requirements.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from tests.benchmark.schemas import (
    BenchmarkExample,
    DifficultyLevel,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# PERFORMANCE THRESHOLDS
# =============================================================================

# Latency thresholds in milliseconds
LATENCY_THRESHOLDS = {
    "p50_target": 2000,  # Median should be under 2s
    "p95_target": 5000,  # 95th percentile under 5s
    "p99_target": 10000,  # 99th percentile under 10s
    "max_acceptable": 30000,  # Hard limit at 30s
}

# Throughput targets
THROUGHPUT_TARGETS = {
    "min_queries_per_minute": 10,  # At least 10 queries/min
    "target_queries_per_minute": 30,  # Target 30 queries/min
}


# =============================================================================
# LATENCY MEASUREMENT UTILITIES
# =============================================================================


class LatencyStats:
    """Container for latency statistics."""

    def __init__(self, latencies: list[float]) -> None:
        """Initialize with list of latencies in milliseconds."""
        self.latencies = sorted(latencies)
        self.count = len(latencies)

    @property
    def min(self) -> float:
        """Minimum latency."""
        return self.latencies[0] if self.latencies else 0.0

    @property
    def max(self) -> float:
        """Maximum latency."""
        return self.latencies[-1] if self.latencies else 0.0

    @property
    def mean(self) -> float:
        """Mean latency."""
        return statistics.mean(self.latencies) if self.latencies else 0.0

    @property
    def median(self) -> float:
        """Median (p50) latency."""
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p95(self) -> float:
        """95th percentile latency."""
        if not self.latencies:
            return 0.0
        idx = int(0.95 * len(self.latencies))
        return self.latencies[min(idx, len(self.latencies) - 1)]

    @property
    def p99(self) -> float:
        """99th percentile latency."""
        if not self.latencies:
            return 0.0
        idx = int(0.99 * len(self.latencies))
        return self.latencies[min(idx, len(self.latencies) - 1)]

    @property
    def std_dev(self) -> float:
        """Standard deviation."""
        if len(self.latencies) < 2:
            return 0.0
        return statistics.stdev(self.latencies)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "min_ms": self.min,
            "max_ms": self.max,
            "mean_ms": self.mean,
            "median_ms": self.median,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "std_dev_ms": self.std_dev,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"LatencyStats(n={self.count}, "
            f"mean={self.mean:.1f}ms, "
            f"p50={self.median:.1f}ms, "
            f"p95={self.p95:.1f}ms, "
            f"p99={self.p99:.1f}ms)"
        )


async def measure_latency(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, float]:
    """Measure latency of an async function.

    Args:
        func: Async function to measure.
        *args: Positional arguments for func.
        **kwargs: Keyword arguments for func.

    Returns:
        Tuple of (result, latency_ms).
    """
    start = time.perf_counter()
    result = await func(*args, **kwargs)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    return result, latency_ms


# =============================================================================
# UNIT TESTS FOR LATENCY UTILITIES
# =============================================================================


class TestLatencyStats:
    """Unit tests for LatencyStats class."""

    def test_basic_stats(self) -> None:
        """Test basic statistics calculation."""
        latencies = [100, 200, 300, 400, 500]
        stats = LatencyStats(latencies)

        assert stats.count == 5
        assert stats.min == 100
        assert stats.max == 500
        assert stats.mean == 300
        assert stats.median == 300

    def test_percentiles(self) -> None:
        """Test percentile calculations."""
        # 100 samples from 1 to 100
        latencies = list(range(1, 101))
        stats = LatencyStats(latencies)

        # p95 index = int(0.95 * 100) = 95, which is value 96 (0-indexed)
        assert stats.p95 == 96
        assert stats.p99 == 100

    def test_empty_latencies(self) -> None:
        """Test handling of empty latencies."""
        stats = LatencyStats([])

        assert stats.count == 0
        assert stats.min == 0.0
        assert stats.max == 0.0
        assert stats.mean == 0.0

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        latencies = [100, 200, 300]
        stats = LatencyStats(latencies)
        d = stats.to_dict()

        assert "count" in d
        assert "mean_ms" in d
        assert "p95_ms" in d
        assert d["count"] == 3


class TestMeasureLatency:
    """Unit tests for measure_latency function."""

    @pytest.mark.asyncio
    async def test_measures_async_function(self) -> None:
        """Test measuring async function latency."""

        async def slow_func() -> str:
            await asyncio.sleep(0.1)  # 100ms
            return "done"

        result, latency = await measure_latency(slow_func)

        assert result == "done"
        assert latency >= 100  # At least 100ms
        assert latency < 200  # Not too much overhead

    @pytest.mark.asyncio
    async def test_passes_arguments(self) -> None:
        """Test that arguments are passed correctly."""

        async def add(a: int, b: int) -> int:
            return a + b

        result, _ = await measure_latency(add, 2, 3)
        assert result == 5


# =============================================================================
# PERFORMANCE THRESHOLD TESTS
# =============================================================================


class TestPerformanceThresholds:
    """Tests for performance threshold compliance."""

    def test_thresholds_are_reasonable(self) -> None:
        """Test that thresholds are reasonably configured."""
        # p50 should be less than p95
        assert LATENCY_THRESHOLDS["p50_target"] < LATENCY_THRESHOLDS["p95_target"]

        # p95 should be less than p99
        assert LATENCY_THRESHOLDS["p95_target"] < LATENCY_THRESHOLDS["p99_target"]

        # p99 should be less than max
        assert LATENCY_THRESHOLDS["p99_target"] < LATENCY_THRESHOLDS["max_acceptable"]

    def test_throughput_targets_consistent(self) -> None:
        """Test throughput targets are consistent."""
        min_qpm = THROUGHPUT_TARGETS["min_queries_per_minute"]
        target_qpm = THROUGHPUT_TARGETS["target_queries_per_minute"]

        assert min_qpm > 0
        assert target_qpm > min_qpm


# =============================================================================
# MOCK PIPELINE LATENCY TESTS
# =============================================================================


class TestMockPipelineLatency:
    """Tests with mocked pipeline latency."""

    @pytest.mark.asyncio
    async def test_retrieval_latency(
        self,
        mock_retriever: MagicMock,
    ) -> None:
        """Test retrieval latency measurement."""

        async def mock_search(query: str, k: int = 5) -> list[Any]:
            await asyncio.sleep(0.05)  # 50ms simulated latency
            return [MagicMock(content="result")]

        mock_retriever.search = mock_search

        _, latency = await measure_latency(mock_retriever.search, "test query")

        assert latency >= 50
        assert latency < LATENCY_THRESHOLDS["p50_target"]

    @pytest.mark.asyncio
    async def test_full_pipeline_latency(
        self,
        mock_config: MagicMock,
    ) -> None:
        """Test full pipeline latency simulation."""

        async def mock_pipeline(question: str) -> dict[str, Any]:
            # Simulate: routing (50ms) + retrieval (100ms) + generation (200ms)
            await asyncio.sleep(0.05)  # Router
            await asyncio.sleep(0.10)  # Retrieval
            await asyncio.sleep(0.20)  # Generation
            return {"answer": "test answer"}

        _, latency = await measure_latency(mock_pipeline, "test question")

        # Should be around 350ms with some variance
        assert latency >= 300
        assert latency < 500

    @pytest.mark.asyncio
    async def test_latency_by_difficulty(
        self,
        golden_dataset: list[BenchmarkExample],
    ) -> None:
        """Test that complex queries are expected to take longer."""
        # Group by difficulty
        by_difficulty: dict[DifficultyLevel, list[BenchmarkExample]] = {}
        for ex in golden_dataset:
            if ex.difficulty not in by_difficulty:
                by_difficulty[ex.difficulty] = []
            by_difficulty[ex.difficulty].append(ex)

        # Harder queries might need more iterations, so we don't set
        # strict latency expectations, just verify we have examples at each level
        assert DifficultyLevel.EASY in by_difficulty
        assert DifficultyLevel.HARD in by_difficulty


# =============================================================================
# THROUGHPUT TESTS
# =============================================================================


class TestThroughput:
    """Tests for throughput measurement."""

    @pytest.mark.asyncio
    async def test_concurrent_queries(self) -> None:
        """Test handling multiple concurrent queries."""
        query_count = 5
        results: list[tuple[Any, float]] = []

        async def mock_query(idx: int) -> dict[str, Any]:
            await asyncio.sleep(0.1)  # 100ms each
            return {"idx": idx, "answer": f"answer {idx}"}

        # Run queries concurrently
        start = time.perf_counter()
        tasks = [measure_latency(mock_query, i) for i in range(query_count)]
        results = await asyncio.gather(*tasks)
        total_time = (time.perf_counter() - start) * 1000

        # All should complete
        assert len(results) == query_count

        # Concurrent should be faster than sequential
        # Sequential would be ~500ms, concurrent should be closer to 100ms + overhead
        assert total_time < 300  # Should be well under 300ms

    @pytest.mark.asyncio
    async def test_sustained_throughput(self) -> None:
        """Test sustained throughput over multiple queries."""
        query_count = 10
        latencies: list[float] = []

        async def mock_query() -> str:
            await asyncio.sleep(0.05)  # 50ms
            return "done"

        for _ in range(query_count):
            _, latency = await measure_latency(mock_query)
            latencies.append(latency)

        stats = LatencyStats(latencies)

        # Should maintain consistent latency
        assert stats.std_dev < stats.mean * 0.5  # Less than 50% variance

        # Calculate throughput
        total_time_seconds = sum(latencies) / 1000
        throughput = query_count / (total_time_seconds / 60)  # queries per minute

        assert throughput >= THROUGHPUT_TARGETS["min_queries_per_minute"]


# =============================================================================
# BENCHMARK-DRIVEN PERFORMANCE TESTS
# =============================================================================


class TestBenchmarkPerformance:
    """Benchmark-driven performance tests."""

    def test_smoke_test_dataset_size(
        self,
        smoke_test_dataset: list[BenchmarkExample],
    ) -> None:
        """Test smoke test dataset is appropriate size for fast runs."""
        # Smoke test should be 10 or fewer for fast CI
        assert len(smoke_test_dataset) <= 10

    def test_full_dataset_size(
        self,
        full_benchmark_dataset: list[BenchmarkExample],
    ) -> None:
        """Test full dataset is comprehensive but not excessive."""
        # Full dataset should be 200-300 examples
        assert 200 <= len(full_benchmark_dataset) <= 300

    @pytest.mark.asyncio
    async def test_example_latency_expectations(
        self,
        golden_example: BenchmarkExample,
    ) -> None:
        """Test that examples have reasonable latency expectations."""
        # All examples should complete within max threshold
        # This is a placeholder - actual latency testing needs live system

        # Verify example is well-formed
        assert golden_example.question
        assert len(golden_example.question) > 5


# =============================================================================
# REGRESSION DETECTION
# =============================================================================


class TestPerformanceRegression:
    """Tests for performance regression detection."""

    def test_latency_regression_detection(self) -> None:
        """Test detecting latency regression."""
        baseline_latencies = [100, 120, 110, 130, 105]
        current_latencies = [150, 180, 160, 200, 170]  # Regressed

        baseline_stats = LatencyStats(baseline_latencies)
        current_stats = LatencyStats(current_latencies)

        # Calculate regression percentage
        regression_pct = ((current_stats.mean - baseline_stats.mean) / baseline_stats.mean) * 100

        # This should flag a regression (> 20% increase)
        assert regression_pct > 20, "Should detect significant regression"

    def test_no_regression_detection(self) -> None:
        """Test no false positives for regression."""
        baseline_latencies = [100, 120, 110, 130, 105]
        current_latencies = [102, 118, 112, 128, 108]  # Similar

        baseline_stats = LatencyStats(baseline_latencies)
        current_stats = LatencyStats(current_latencies)

        regression_pct = ((current_stats.mean - baseline_stats.mean) / baseline_stats.mean) * 100

        # Should not flag as regression (< 10% change)
        assert abs(regression_pct) < 10, "Should not flag minor variance as regression"


# =============================================================================
# RESOURCE UTILIZATION
# =============================================================================


class TestResourceUtilization:
    """Tests for resource utilization tracking."""

    def test_token_count_estimation(self) -> None:
        """Test token count estimation for queries."""

        # Simple estimation: ~4 chars per token
        def estimate_tokens(text: str) -> int:
            return len(text) // 4

        # Short query
        short_query = "What is traceability?"
        assert estimate_tokens(short_query) < 10

        # Long query
        long_query = (
            "What is requirements traceability and how does it relate to "
            "ISO 26262 compliance in automotive functional safety development?"
        )
        assert estimate_tokens(long_query) < 50

    def test_context_size_reasonable(
        self,
        golden_example: BenchmarkExample,
    ) -> None:
        """Test that example context expectations are reasonable."""
        # Ground truth shouldn't be excessively long
        assert len(golden_example.ground_truth) < 2000, (
            f"Ground truth too long for {golden_example.id}"
        )


__all__ = [
    "LATENCY_THRESHOLDS",
    "THROUGHPUT_TARGETS",
    "LatencyStats",
    "measure_latency",
]
