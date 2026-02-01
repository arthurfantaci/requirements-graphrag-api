"""Tests for guardrail metrics collection."""

from __future__ import annotations

import threading

from requirements_graphrag_api.guardrails.metrics import (
    GuardrailMetrics,
    MetricsCollector,
    metrics,
)


class TestGuardrailMetrics:
    """Test the GuardrailMetrics dataclass."""

    def test_defaults(self):
        m = GuardrailMetrics()
        assert m.total_requests == 0
        assert m.requests_blocked == 0
        assert m.requests_warned == 0
        assert m.avg_guardrail_latency_ms == 0.0
        assert m.period_end is None

    def test_to_dict(self):
        m = GuardrailMetrics(total_requests=100, requests_blocked=5, requests_warned=10)
        d = m.to_dict()

        assert "period" in d
        assert "summary" in d
        assert "by_type" in d
        assert "performance" in d

        assert d["summary"]["total_requests"] == 100
        assert d["summary"]["requests_blocked"] == 5
        assert d["summary"]["requests_warned"] == 10
        assert d["summary"]["block_rate"] == 0.05  # 5/100
        assert d["summary"]["warn_rate"] == 0.1  # 10/100

    def test_block_rate_zero_requests(self):
        m = GuardrailMetrics(total_requests=0)
        d = m.to_dict()
        assert d["summary"]["block_rate"] == 0.0

    def test_period_is_current(self):
        m = GuardrailMetrics()
        d = m.to_dict()
        assert d["period"]["is_current"] is True


class TestMetricsCollector:
    """Test the MetricsCollector class."""

    def setup_method(self):
        """Reset collector before each test."""
        self.collector = MetricsCollector()

    def test_record_request(self):
        self.collector.record_request()
        m = self.collector.get_current_metrics()
        assert m.total_requests == 1
        assert m.requests_blocked == 0

    def test_record_blocked_request(self):
        self.collector.record_request(blocked=True)
        m = self.collector.get_current_metrics()
        assert m.total_requests == 1
        assert m.requests_blocked == 1

    def test_record_warned_request(self):
        self.collector.record_request(warned=True)
        m = self.collector.get_current_metrics()
        assert m.total_requests == 1
        assert m.requests_warned == 1

    def test_record_prompt_injection(self):
        self.collector.record_prompt_injection(blocked=False)
        m = self.collector.get_current_metrics()
        assert m.prompt_injection_detected == 1
        assert m.prompt_injection_blocked == 0

    def test_record_prompt_injection_blocked(self):
        self.collector.record_prompt_injection(blocked=True)
        m = self.collector.get_current_metrics()
        assert m.prompt_injection_detected == 1
        assert m.prompt_injection_blocked == 1

    def test_record_pii(self):
        self.collector.record_pii(redacted=True)
        m = self.collector.get_current_metrics()
        assert m.pii_detected == 1
        assert m.pii_redacted == 1

    def test_record_toxicity(self):
        self.collector.record_toxicity(blocked=True)
        m = self.collector.get_current_metrics()
        assert m.toxicity_detected == 1
        assert m.toxicity_blocked == 1

    def test_record_topic_out_of_scope(self):
        self.collector.record_topic_out_of_scope()
        m = self.collector.get_current_metrics()
        assert m.topic_out_of_scope == 1

    def test_record_rate_limit_exceeded(self):
        self.collector.record_rate_limit_exceeded()
        m = self.collector.get_current_metrics()
        assert m.rate_limit_exceeded == 1

    def test_record_hallucination_warning(self):
        self.collector.record_hallucination_warning()
        m = self.collector.get_current_metrics()
        assert m.hallucination_warnings == 1

    def test_record_conversation_validation_issue(self):
        self.collector.record_conversation_validation_issue()
        m = self.collector.get_current_metrics()
        assert m.conversation_validation_issues == 1

    def test_record_size_exceeded(self):
        self.collector.record_size_exceeded()
        m = self.collector.get_current_metrics()
        assert m.request_size_exceeded == 1

    def test_record_timeout(self):
        self.collector.record_timeout()
        m = self.collector.get_current_metrics()
        assert m.request_timeout == 1

    def test_record_latency(self):
        self.collector.record_latency(10.0)
        self.collector.record_latency(20.0)
        m = self.collector.get_current_metrics()
        assert m.avg_guardrail_latency_ms == 15.0  # (10 + 20) / 2

    def test_get_current_metrics_returns_copy(self):
        self.collector.record_request()
        m1 = self.collector.get_current_metrics()
        m2 = self.collector.get_current_metrics()
        # Should be separate objects
        assert m1 is not m2
        assert m1.total_requests == m2.total_requests


class TestMetricsRotation:
    """Test metrics period rotation."""

    def setup_method(self):
        self.collector = MetricsCollector(max_history=3)

    def test_rotate_period(self):
        self.collector.record_request()
        self.collector.record_request()
        completed = self.collector.rotate_period()

        assert completed.total_requests == 2
        assert completed.period_end is not None

        # Current should be reset
        current = self.collector.get_current_metrics()
        assert current.total_requests == 0

    def test_rotate_period_adds_to_history(self):
        self.collector.record_request()
        self.collector.rotate_period()

        history = self.collector.get_history()
        assert len(history) == 1

    def test_history_limited_to_max(self):
        for _i in range(5):
            self.collector.record_request()
            self.collector.rotate_period()

        history = self.collector.get_history()
        assert len(history) == 3  # max_history=3

    def test_reset_clears_everything(self):
        self.collector.record_request()
        self.collector.rotate_period()
        self.collector.record_request()

        self.collector.reset()

        assert self.collector.get_current_metrics().total_requests == 0
        assert len(self.collector.get_history()) == 0


class TestMetricsThreadSafety:
    """Test thread safety of metrics collector."""

    def test_concurrent_recording(self):
        collector = MetricsCollector()
        num_threads = 10
        increments_per_thread = 100

        def record_requests():
            for _ in range(increments_per_thread):
                collector.record_request()

        threads = [threading.Thread(target=record_requests) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        m = collector.get_current_metrics()
        assert m.total_requests == num_threads * increments_per_thread


class TestGlobalMetricsInstance:
    """Test the global metrics instance."""

    def test_global_metrics_exists(self):
        assert metrics is not None
        assert isinstance(metrics, MetricsCollector)

    def test_global_metrics_can_record(self):
        # Reset first
        metrics.reset()
        metrics.record_request()
        m = metrics.get_current_metrics()
        assert m.total_requests >= 1
        # Clean up
        metrics.reset()
