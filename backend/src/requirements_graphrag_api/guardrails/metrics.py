"""Guardrail metrics collection for compliance dashboard.

This module provides metrics collection and aggregation for all guardrail
operations, enabling compliance monitoring and performance analysis.

Collected Metrics:
    - Request counts (total, blocked, warned)
    - Detection counts by guardrail type
    - Average latency for guardrail processing
    - Period-based aggregation for historical analysis

Usage:
    from requirements_graphrag_api.guardrails.metrics import metrics

    # Record events
    metrics.record_request(blocked=False)
    metrics.record_prompt_injection(blocked=True)
    metrics.record_latency(15.5)

    # Get current metrics
    current = metrics.get_current_metrics()
    print(current.to_dict())

    # Rotate to new period (e.g., hourly)
    metrics.rotate_period()
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


@dataclass
class GuardrailMetrics:
    """Metrics for guardrail performance and compliance.

    This dataclass holds counters and timing data for a single metrics period.
    Use to_dict() to serialize for API responses.

    Attributes:
        total_requests: Total requests processed.
        requests_blocked: Requests blocked by guardrails.
        requests_warned: Requests that triggered warnings.
        prompt_injection_detected: Prompt injection patterns detected.
        prompt_injection_blocked: Requests blocked due to injection.
        pii_detected: PII instances detected.
        pii_redacted: PII instances redacted.
        toxicity_detected: Toxic content detected.
        toxicity_blocked: Requests blocked due to toxicity.
        topic_out_of_scope: Off-topic requests detected.
        rate_limit_exceeded: Rate limit violations.
        hallucination_warnings: Hallucination warnings added.
        conversation_validation_issues: Conversation history issues found.
        avg_guardrail_latency_ms: Average guardrail processing time.
        period_start: Start of this metrics period.
        period_end: End of this metrics period (None if current).
    """

    # Request counters
    total_requests: int = 0
    requests_blocked: int = 0
    requests_warned: int = 0

    # Detection counters by guardrail type
    prompt_injection_detected: int = 0
    prompt_injection_blocked: int = 0
    pii_detected: int = 0
    pii_redacted: int = 0
    toxicity_detected: int = 0
    toxicity_blocked: int = 0
    topic_out_of_scope: int = 0
    rate_limit_exceeded: int = 0
    hallucination_warnings: int = 0
    conversation_validation_issues: int = 0

    # Size/timeout counters
    request_size_exceeded: int = 0
    request_timeout: int = 0

    # Timing
    avg_guardrail_latency_ms: float = 0.0

    # Period bounds
    period_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    period_end: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for API response.

        Returns:
            Dictionary with nested structure for easy consumption.
        """
        block_rate = self.requests_blocked / self.total_requests if self.total_requests > 0 else 0.0
        warn_rate = self.requests_warned / self.total_requests if self.total_requests > 0 else 0.0

        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat() if self.period_end else None,
                "is_current": self.period_end is None,
            },
            "summary": {
                "total_requests": self.total_requests,
                "requests_blocked": self.requests_blocked,
                "requests_warned": self.requests_warned,
                "block_rate": round(block_rate, 4),
                "warn_rate": round(warn_rate, 4),
            },
            "by_type": {
                "prompt_injection": {
                    "detected": self.prompt_injection_detected,
                    "blocked": self.prompt_injection_blocked,
                },
                "pii": {
                    "detected": self.pii_detected,
                    "redacted": self.pii_redacted,
                },
                "toxicity": {
                    "detected": self.toxicity_detected,
                    "blocked": self.toxicity_blocked,
                },
                "topic_guard": {
                    "out_of_scope": self.topic_out_of_scope,
                },
                "rate_limiting": {
                    "exceeded": self.rate_limit_exceeded,
                },
                "hallucination": {
                    "warnings_added": self.hallucination_warnings,
                },
                "conversation": {
                    "validation_issues": self.conversation_validation_issues,
                },
                "resource_limits": {
                    "size_exceeded": self.request_size_exceeded,
                    "timeouts": self.request_timeout,
                },
            },
            "performance": {
                "avg_guardrail_latency_ms": round(self.avg_guardrail_latency_ms, 2),
            },
        }


class MetricsCollector:
    """Thread-safe collector for guardrail metrics.

    Maintains current period metrics and historical periods.
    All public methods are thread-safe.

    Attributes:
        max_history: Maximum number of historical periods to retain.
    """

    def __init__(self, max_history: int = 24) -> None:
        """Initialize the metrics collector.

        Args:
            max_history: Maximum historical periods to retain.
        """
        self._lock = threading.Lock()
        self._current = GuardrailMetrics()
        self._history: list[GuardrailMetrics] = []
        self._latencies: list[float] = []
        self._max_history = max_history

    def record_request(self, blocked: bool = False, warned: bool = False) -> None:
        """Record a processed request.

        Args:
            blocked: Whether the request was blocked.
            warned: Whether a warning was generated.
        """
        with self._lock:
            self._current.total_requests += 1
            if blocked:
                self._current.requests_blocked += 1
            if warned:
                self._current.requests_warned += 1

    def record_prompt_injection(self, blocked: bool = False) -> None:
        """Record a prompt injection detection.

        Args:
            blocked: Whether the request was blocked.
        """
        with self._lock:
            self._current.prompt_injection_detected += 1
            if blocked:
                self._current.prompt_injection_blocked += 1

    def record_pii(self, redacted: bool = True) -> None:
        """Record a PII detection.

        Args:
            redacted: Whether PII was redacted.
        """
        with self._lock:
            self._current.pii_detected += 1
            if redacted:
                self._current.pii_redacted += 1

    def record_toxicity(self, blocked: bool = False) -> None:
        """Record a toxicity detection.

        Args:
            blocked: Whether the request was blocked.
        """
        with self._lock:
            self._current.toxicity_detected += 1
            if blocked:
                self._current.toxicity_blocked += 1

    def record_topic_out_of_scope(self) -> None:
        """Record an off-topic request detection."""
        with self._lock:
            self._current.topic_out_of_scope += 1

    def record_rate_limit_exceeded(self) -> None:
        """Record a rate limit violation."""
        with self._lock:
            self._current.rate_limit_exceeded += 1

    def record_hallucination_warning(self) -> None:
        """Record a hallucination warning being added."""
        with self._lock:
            self._current.hallucination_warnings += 1

    def record_conversation_validation_issue(self) -> None:
        """Record a conversation history validation issue."""
        with self._lock:
            self._current.conversation_validation_issues += 1

    def record_size_exceeded(self) -> None:
        """Record a request size limit violation."""
        with self._lock:
            self._current.request_size_exceeded += 1

    def record_timeout(self) -> None:
        """Record a request timeout."""
        with self._lock:
            self._current.request_timeout += 1

    def record_latency(self, latency_ms: float) -> None:
        """Record guardrail processing latency.

        Args:
            latency_ms: Processing time in milliseconds.
        """
        with self._lock:
            self._latencies.append(latency_ms)
            # Update running average
            if self._latencies:
                self._current.avg_guardrail_latency_ms = sum(self._latencies) / len(self._latencies)

    def get_current_metrics(self) -> GuardrailMetrics:
        """Get the current period's metrics.

        Returns:
            Copy of current metrics (thread-safe snapshot).
        """
        with self._lock:
            # Return a copy to prevent external modification
            return GuardrailMetrics(
                total_requests=self._current.total_requests,
                requests_blocked=self._current.requests_blocked,
                requests_warned=self._current.requests_warned,
                prompt_injection_detected=self._current.prompt_injection_detected,
                prompt_injection_blocked=self._current.prompt_injection_blocked,
                pii_detected=self._current.pii_detected,
                pii_redacted=self._current.pii_redacted,
                toxicity_detected=self._current.toxicity_detected,
                toxicity_blocked=self._current.toxicity_blocked,
                topic_out_of_scope=self._current.topic_out_of_scope,
                rate_limit_exceeded=self._current.rate_limit_exceeded,
                hallucination_warnings=self._current.hallucination_warnings,
                conversation_validation_issues=self._current.conversation_validation_issues,
                request_size_exceeded=self._current.request_size_exceeded,
                request_timeout=self._current.request_timeout,
                avg_guardrail_latency_ms=self._current.avg_guardrail_latency_ms,
                period_start=self._current.period_start,
                period_end=self._current.period_end,
            )

    def get_history(self) -> list[GuardrailMetrics]:
        """Get historical metrics periods.

        Returns:
            List of historical metrics (oldest first).
        """
        with self._lock:
            return list(self._history)

    def rotate_period(self) -> GuardrailMetrics:
        """Rotate to a new metrics period.

        Closes the current period and starts a fresh one.
        Returns the completed period metrics.

        Returns:
            The completed period's metrics.
        """
        with self._lock:
            # Close current period
            self._current.period_end = datetime.now(UTC)
            completed = self._current

            # Archive to history
            self._history.append(completed)

            # Trim history if needed
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

            # Start new period
            self._current = GuardrailMetrics()
            self._latencies = []

            return completed

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._current = GuardrailMetrics()
            self._history = []
            self._latencies = []


# Global metrics collector instance
metrics = MetricsCollector()
