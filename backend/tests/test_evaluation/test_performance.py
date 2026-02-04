"""Tests for performance monitoring utilities.

Tests cover:
- PerformanceTracker execution tracking
- Subgraph metrics calculation
- Optimization hints generation
- Global tracker functions
"""

from __future__ import annotations

import time

from requirements_graphrag_api.evaluation.performance import (
    ExecutionMetrics,
    PerformanceTracker,
    SubgraphMetrics,
    get_global_tracker,
    get_performance_summary,
    reset_global_tracker,
    track_execution,
)

# =============================================================================
# EXECUTION METRICS TESTS
# =============================================================================


class TestExecutionMetrics:
    """Tests for ExecutionMetrics dataclass."""

    def test_duration_calculation(self):
        """Test duration is calculated correctly."""
        metrics = ExecutionMetrics(
            name="test",
            start_time=100.0,
            end_time=100.5,
        )
        assert metrics.duration_ms == 500.0

    def test_duration_incomplete(self):
        """Test duration returns 0 when incomplete."""
        metrics = ExecutionMetrics(
            name="test",
            start_time=100.0,
        )
        assert metrics.duration_ms == 0.0

    def test_complete_sets_end_time(self):
        """Test complete() sets end_time."""
        metrics = ExecutionMetrics(
            name="test",
            start_time=time.perf_counter(),
        )
        assert metrics.end_time is None
        metrics.complete()
        assert metrics.end_time is not None
        assert metrics.duration_ms > 0


# =============================================================================
# SUBGRAPH METRICS TESTS
# =============================================================================


class TestSubgraphMetrics:
    """Tests for SubgraphMetrics aggregation."""

    def test_empty_metrics(self):
        """Test metrics with no executions."""
        metrics = SubgraphMetrics(name="test")
        assert metrics.total_calls == 0
        assert metrics.total_duration_ms == 0.0
        assert metrics.avg_duration_ms == 0.0
        assert metrics.min_duration_ms == 0.0
        assert metrics.max_duration_ms == 0.0

    def test_aggregated_metrics(self):
        """Test metrics with multiple executions."""
        metrics = SubgraphMetrics(name="test")
        metrics.executions = [
            ExecutionMetrics(name="test", start_time=0, end_time=0.1),  # 100ms
            ExecutionMetrics(name="test", start_time=0, end_time=0.2),  # 200ms
            ExecutionMetrics(name="test", start_time=0, end_time=0.3),  # 300ms
        ]
        assert metrics.total_calls == 3
        assert metrics.total_duration_ms == 600.0
        assert metrics.avg_duration_ms == 200.0
        assert metrics.min_duration_ms == 100.0
        assert metrics.max_duration_ms == 300.0


# =============================================================================
# PERFORMANCE TRACKER TESTS
# =============================================================================


class TestPerformanceTracker:
    """Tests for PerformanceTracker."""

    def test_track_context_manager(self):
        """Test tracking with context manager."""
        tracker = PerformanceTracker()

        with tracker.track("test_operation") as metrics:
            time.sleep(0.01)  # 10ms

        assert metrics.name == "test_operation"
        assert metrics.duration_ms >= 10.0

    def test_track_multiple_operations(self):
        """Test tracking multiple operations."""
        tracker = PerformanceTracker()

        with tracker.track("op1"):
            pass

        with tracker.track("op2"):
            pass

        with tracker.track("op1"):  # Second call to op1
            pass

        summary = tracker.get_summary()
        assert "op1" in summary["subgraphs"]
        assert "op2" in summary["subgraphs"]
        assert summary["subgraphs"]["op1"]["total_calls"] == 2
        assert summary["subgraphs"]["op2"]["total_calls"] == 1

    def test_track_with_metadata(self):
        """Test tracking with metadata."""
        tracker = PerformanceTracker()

        with tracker.track("test", metadata={"model": "gpt-4"}) as metrics:
            pass

        assert metrics.metadata == {"model": "gpt-4"}

    def test_record_external_timing(self):
        """Test recording external timing data."""
        tracker = PerformanceTracker()
        tracker.record("external_op", duration_ms=150.0, metadata={"source": "external"})

        metrics = tracker.get_subgraph_metrics("external_op")
        assert metrics is not None
        assert metrics.total_calls == 1
        assert abs(metrics.avg_duration_ms - 150.0) < 1.0  # Allow small float error

    def test_get_summary(self):
        """Test summary generation."""
        tracker = PerformanceTracker()

        with tracker.track("fast_op"):
            time.sleep(0.001)

        with tracker.track("slow_op"):
            time.sleep(0.01)

        summary = tracker.get_summary()

        assert "total_duration_ms" in summary
        assert "subgraphs" in summary
        assert "slowest_operation" in summary
        assert "optimization_hints" in summary
        assert summary["slowest_operation"] in ["fast_op", "slow_op"]

    def test_reset(self):
        """Test tracker reset."""
        tracker = PerformanceTracker()

        with tracker.track("test"):
            pass

        assert len(tracker._metrics) > 0
        tracker.reset()
        assert len(tracker._metrics) == 0


# =============================================================================
# OPTIMIZATION HINTS TESTS
# =============================================================================


class TestOptimizationHints:
    """Tests for optimization hint generation."""

    def test_high_variance_hint(self):
        """Test hint for high variance operations."""
        tracker = PerformanceTracker()

        # Record operations with high variance (max > 3x avg)
        # With [10, 10, 10, 100]: avg = 130/4 = 32.5, max = 100
        # 100 > 32.5 * 3 = 97.5? Yes!
        tracker.record("high_var_op", duration_ms=10)
        tracker.record("high_var_op", duration_ms=10)
        tracker.record("high_var_op", duration_ms=10)
        tracker.record("high_var_op", duration_ms=100)

        summary = tracker.get_summary()
        hints = summary["optimization_hints"]

        assert any("variance" in hint.lower() for hint in hints)

    def test_multiple_calls_hint(self):
        """Test hint for operations called many times."""
        tracker = PerformanceTracker()

        for _ in range(5):
            tracker.record("repeated_op", duration_ms=10)

        summary = tracker.get_summary()
        hints = summary["optimization_hints"]

        assert any("5 times" in hint or "batching" in hint.lower() for hint in hints)


# =============================================================================
# GLOBAL TRACKER TESTS
# =============================================================================


class TestGlobalTracker:
    """Tests for global tracker functions."""

    def test_global_tracker_singleton(self):
        """Test global tracker is a singleton."""
        reset_global_tracker()
        tracker1 = get_global_tracker()
        tracker2 = get_global_tracker()
        assert tracker1 is tracker2

    def test_track_execution_function(self):
        """Test track_execution convenience function."""
        reset_global_tracker()

        with track_execution("test_op") as metrics:
            time.sleep(0.001)

        assert metrics.name == "test_op"

        summary = get_performance_summary()
        assert "test_op" in summary["subgraphs"]

    def test_reset_global_tracker(self):
        """Test resetting global tracker."""
        reset_global_tracker()

        with track_execution("test"):
            pass

        reset_global_tracker()

        # New tracker should have no data
        summary = get_performance_summary()
        assert len(summary["subgraphs"]) == 0


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestPerformanceIntegration:
    """Integration-style tests for performance tracking."""

    def test_nested_tracking(self):
        """Test nested operation tracking."""
        tracker = PerformanceTracker()

        with tracker.track("outer"):
            time.sleep(0.001)
            with tracker.track("inner"):
                time.sleep(0.001)

        summary = tracker.get_summary()
        assert "outer" in summary["subgraphs"]
        assert "inner" in summary["subgraphs"]
        # Outer should be longer than inner
        assert (
            summary["subgraphs"]["outer"]["total_duration_ms"]
            >= summary["subgraphs"]["inner"]["total_duration_ms"]
        )

    def test_exports_from_package(self):
        """Test that performance utilities are exported from package."""
        from requirements_graphrag_api.evaluation import (
            ExecutionMetrics,
            PerformanceTracker,
            SubgraphMetrics,
            get_global_tracker,
            get_performance_summary,
            reset_global_tracker,
            track_execution,
        )

        assert callable(get_global_tracker)
        assert callable(get_performance_summary)
        assert callable(reset_global_tracker)
        assert callable(track_execution)
        assert ExecutionMetrics is not None
        assert PerformanceTracker is not None
        assert SubgraphMetrics is not None
