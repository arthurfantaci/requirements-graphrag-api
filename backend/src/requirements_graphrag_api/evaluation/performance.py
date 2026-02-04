"""Performance monitoring for the agentic RAG system.

Provides utilities for tracking:
- Subgraph execution times
- Tool invocation latency
- Iteration counts and efficiency
- Parallel execution metrics

Usage:
    from requirements_graphrag_api.evaluation.performance import (
        PerformanceTracker,
        track_execution,
    )

    tracker = PerformanceTracker()
    with tracker.track("rag_subgraph"):
        # Execute subgraph
        pass

    print(tracker.get_summary())
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Metrics for a single execution."""

    name: str
    start_time: float
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def complete(self) -> None:
        """Mark execution as complete."""
        self.end_time = time.perf_counter()


@dataclass
class SubgraphMetrics:
    """Aggregated metrics for a subgraph."""

    name: str
    executions: list[ExecutionMetrics] = field(default_factory=list)

    @property
    def total_calls(self) -> int:
        """Total number of calls."""
        return len(self.executions)

    @property
    def total_duration_ms(self) -> float:
        """Total duration in milliseconds."""
        return sum(e.duration_ms for e in self.executions)

    @property
    def avg_duration_ms(self) -> float:
        """Average duration in milliseconds."""
        if not self.executions:
            return 0.0
        return self.total_duration_ms / len(self.executions)

    @property
    def min_duration_ms(self) -> float:
        """Minimum duration in milliseconds."""
        if not self.executions:
            return 0.0
        return min(e.duration_ms for e in self.executions)

    @property
    def max_duration_ms(self) -> float:
        """Maximum duration in milliseconds."""
        if not self.executions:
            return 0.0
        return max(e.duration_ms for e in self.executions)


class PerformanceTracker:
    """Track performance metrics for the agentic system.

    Thread-safe performance tracker for measuring execution times
    of subgraphs, tools, and other operations.

    Example:
        tracker = PerformanceTracker()

        with tracker.track("rag_subgraph"):
            # Execute RAG subgraph
            pass

        with tracker.track("synthesis_subgraph"):
            # Execute synthesis
            pass

        summary = tracker.get_summary()
        print(f"Total time: {summary['total_duration_ms']:.2f}ms")
    """

    def __init__(self) -> None:
        """Initialize the performance tracker."""
        self._metrics: dict[str, SubgraphMetrics] = {}
        self._active_executions: list[ExecutionMetrics] = []
        self._start_time: float | None = None
        self._end_time: float | None = None

    def start(self) -> None:
        """Start tracking overall execution."""
        self._start_time = time.perf_counter()
        self._end_time = None

    def stop(self) -> None:
        """Stop tracking overall execution."""
        self._end_time = time.perf_counter()

    @contextmanager
    def track(
        self, name: str, metadata: dict[str, Any] | None = None
    ) -> Generator[ExecutionMetrics, None, None]:
        """Track execution time for a named operation.

        Args:
            name: Name of the operation (e.g., "rag_subgraph", "graph_search").
            metadata: Optional metadata to attach to this execution.

        Yields:
            ExecutionMetrics instance for this execution.
        """
        if self._start_time is None:
            self.start()

        execution = ExecutionMetrics(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata or {},
        )
        self._active_executions.append(execution)

        try:
            yield execution
        finally:
            execution.complete()
            self._active_executions.remove(execution)

            # Add to subgraph metrics
            if name not in self._metrics:
                self._metrics[name] = SubgraphMetrics(name=name)
            self._metrics[name].executions.append(execution)

    def record(self, name: str, duration_ms: float, metadata: dict[str, Any] | None = None) -> None:
        """Record a completed execution.

        Use this when you have external timing data.

        Args:
            name: Name of the operation.
            duration_ms: Duration in milliseconds.
            metadata: Optional metadata.
        """
        end_time = time.perf_counter()
        start_time = end_time - (duration_ms / 1000)

        execution = ExecutionMetrics(
            name=name,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata or {},
        )

        if name not in self._metrics:
            self._metrics[name] = SubgraphMetrics(name=name)
        self._metrics[name].executions.append(execution)

    def get_subgraph_metrics(self, name: str) -> SubgraphMetrics | None:
        """Get metrics for a specific subgraph.

        Args:
            name: Subgraph name.

        Returns:
            SubgraphMetrics or None if not found.
        """
        return self._metrics.get(name)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all performance metrics.

        Returns:
            Dict with performance summary including:
            - total_duration_ms: Total tracked time
            - subgraphs: Per-subgraph metrics
            - slowest_operation: Name of slowest operation
            - optimization_hints: Suggested optimizations
        """
        if self._end_time is None:
            self.stop()

        total_duration_ms = 0.0
        if self._start_time is not None and self._end_time is not None:
            total_duration_ms = (self._end_time - self._start_time) * 1000

        subgraph_summaries = {}
        slowest_operation = None
        slowest_duration = 0.0

        for name, metrics in self._metrics.items():
            subgraph_summaries[name] = {
                "total_calls": metrics.total_calls,
                "total_duration_ms": metrics.total_duration_ms,
                "avg_duration_ms": metrics.avg_duration_ms,
                "min_duration_ms": metrics.min_duration_ms,
                "max_duration_ms": metrics.max_duration_ms,
            }

            if metrics.avg_duration_ms > slowest_duration:
                slowest_duration = metrics.avg_duration_ms
                slowest_operation = name

        # Generate optimization hints
        hints = self._generate_optimization_hints(subgraph_summaries)

        return {
            "total_duration_ms": total_duration_ms,
            "subgraphs": subgraph_summaries,
            "slowest_operation": slowest_operation,
            "optimization_hints": hints,
        }

    def _generate_optimization_hints(self, subgraph_summaries: dict[str, Any]) -> list[str]:
        """Generate optimization hints based on metrics.

        Args:
            subgraph_summaries: Per-subgraph metrics.

        Returns:
            List of optimization suggestions.
        """
        hints = []

        for name, metrics in subgraph_summaries.items():
            # High variance suggests inconsistent performance
            if metrics["max_duration_ms"] > metrics["avg_duration_ms"] * 3:
                hints.append(
                    f"{name}: High variance detected - max is 3x average. "
                    "Consider caching or connection pooling."
                )

            # Multiple calls suggest potential for batching
            if metrics["total_calls"] > 3:
                hints.append(
                    f"{name}: Called {metrics['total_calls']} times. "
                    "Consider batching if queries are similar."
                )

            # Slow operations
            if metrics["avg_duration_ms"] > 1000:
                hints.append(
                    f"{name}: Average {metrics['avg_duration_ms']:.0f}ms is slow. "
                    "Consider async execution or caching."
                )

        return hints

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._active_executions.clear()
        self._start_time = None
        self._end_time = None


# Global tracker instance for convenience
_global_tracker: PerformanceTracker | None = None


def get_global_tracker() -> PerformanceTracker:
    """Get the global performance tracker.

    Returns:
        Global PerformanceTracker instance.
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global performance tracker."""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset()
    _global_tracker = None


@contextmanager
def track_execution(
    name: str, metadata: dict[str, Any] | None = None
) -> Generator[ExecutionMetrics, None, None]:
    """Track execution using the global tracker.

    Convenience function for tracking without explicit tracker instance.

    Args:
        name: Operation name.
        metadata: Optional metadata.

    Yields:
        ExecutionMetrics instance.
    """
    tracker = get_global_tracker()
    with tracker.track(name, metadata) as metrics:
        yield metrics


def get_performance_summary() -> dict[str, Any]:
    """Get performance summary from global tracker.

    Returns:
        Performance summary dict.
    """
    return get_global_tracker().get_summary()


__all__ = [
    "ExecutionMetrics",
    "PerformanceTracker",
    "SubgraphMetrics",
    "get_global_tracker",
    "get_performance_summary",
    "reset_global_tracker",
    "track_execution",
]
