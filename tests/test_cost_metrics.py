"""Tests for cost tracking metrics."""

from __future__ import annotations

from datetime import UTC, datetime

from jama_mcp_server_graphrag.evaluation.cost_metrics import (
    AggregatedCostMetrics,
    CostEfficiencyMetrics,
    CostMetrics,
    CostTracker,
    QueryCostRecord,
    check_query_budget,
    compute_cost_efficiency,
)
from jama_mcp_server_graphrag.token_counter import COST_THRESHOLDS, BudgetStatus


class TestCostMetrics:
    """Tests for CostMetrics dataclass."""

    def test_create_cost_metrics(self):
        """Test creating cost metrics."""
        metrics = CostMetrics(
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            cost_usd=0.0045,
            model="gpt-4o",
        )
        assert metrics.input_tokens == 500
        assert metrics.output_tokens == 200
        assert metrics.total_tokens == 700
        assert metrics.cost_usd == 0.0045
        assert metrics.model == "gpt-4o"

    def test_default_latency(self):
        """Test default latency is zero."""
        metrics = CostMetrics(
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            cost_usd=0.0045,
            model="gpt-4o",
        )
        assert metrics.latency_ms == 0.0

    def test_default_budget_status(self):
        """Test default budget status is OK."""
        metrics = CostMetrics(
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            cost_usd=0.0045,
            model="gpt-4o",
        )
        assert metrics.budget_status == BudgetStatus.OK

    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = CostMetrics(
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            cost_usd=0.0045,
            model="gpt-4o",
            latency_ms=150.5,
            budget_status=BudgetStatus.WARNING,
        )
        result = metrics.to_dict()

        assert result["input_tokens"] == 500
        assert result["output_tokens"] == 200
        assert result["total_tokens"] == 700
        assert result["cost_usd"] == 0.0045
        assert result["model"] == "gpt-4o"
        assert result["latency_ms"] == 150.5
        assert result["budget_status"] == "warning"


class TestCostEfficiencyMetrics:
    """Tests for CostEfficiencyMetrics dataclass."""

    def test_create_efficiency_metrics(self):
        """Test creating efficiency metrics."""
        metrics = CostEfficiencyMetrics(
            quality_score=0.85,
            cost_usd=0.02,
            quality_per_dollar=42.5,
            quality_per_1k_tokens=0.85,
            cost_per_quality_point=0.000235,
        )
        assert metrics.quality_score == 0.85
        assert metrics.cost_usd == 0.02
        assert metrics.quality_per_dollar == 42.5
        assert metrics.quality_per_1k_tokens == 0.85
        assert metrics.cost_per_quality_point == 0.000235

    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = CostEfficiencyMetrics(
            quality_score=0.85,
            cost_usd=0.02,
            quality_per_dollar=42.5,
            quality_per_1k_tokens=0.85,
            cost_per_quality_point=0.000235,
        )
        result = metrics.to_dict()

        assert result["quality_score"] == 0.85
        assert result["cost_usd"] == 0.02
        assert result["quality_per_dollar"] == 42.5
        assert result["quality_per_1k_tokens"] == 0.85
        assert result["cost_per_quality_point"] == 0.000235


class TestAggregatedCostMetrics:
    """Tests for AggregatedCostMetrics dataclass."""

    def test_create_aggregated_metrics(self):
        """Test creating aggregated metrics."""
        metrics = AggregatedCostMetrics(
            query_count=10,
            total_input_tokens=5000,
            total_output_tokens=2000,
            total_tokens=7000,
            total_cost_usd=0.045,
            avg_cost_per_query=0.0045,
            avg_tokens_per_query=700.0,
            avg_latency_ms=150.0,
            budget_used_pct=0.9,
            budget_status=BudgetStatus.OK,
        )
        assert metrics.query_count == 10
        assert metrics.total_tokens == 7000
        assert metrics.budget_status == BudgetStatus.OK

    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = AggregatedCostMetrics(
            query_count=10,
            total_input_tokens=5000,
            total_output_tokens=2000,
            total_tokens=7000,
            total_cost_usd=0.045,
            avg_cost_per_query=0.0045,
            avg_tokens_per_query=700.0,
            avg_latency_ms=150.0,
            budget_used_pct=0.9,
            budget_status=BudgetStatus.OK,
        )
        result = metrics.to_dict()

        assert result["query_count"] == 10
        assert result["total_tokens"] == 7000
        assert result["budget_status"] == "ok"


class TestQueryCostRecord:
    """Tests for QueryCostRecord dataclass."""

    def test_create_record(self):
        """Test creating a query cost record."""
        timestamp = datetime.now(tz=UTC)
        record = QueryCostRecord(
            timestamp=timestamp,
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.0045,
            model="gpt-4o",
        )
        assert record.timestamp == timestamp
        assert record.input_tokens == 500
        assert record.output_tokens == 200
        assert record.cost_usd == 0.0045
        assert record.model == "gpt-4o"

    def test_default_latency(self):
        """Test default latency is zero."""
        record = QueryCostRecord(
            timestamp=datetime.now(tz=UTC),
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.0045,
            model="gpt-4o",
        )
        assert record.latency_ms == 0.0

    def test_default_quality_score(self):
        """Test default quality score is None."""
        record = QueryCostRecord(
            timestamp=datetime.now(tz=UTC),
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.0045,
            model="gpt-4o",
        )
        assert record.quality_score is None

    def test_with_quality_score(self):
        """Test record with quality score."""
        record = QueryCostRecord(
            timestamp=datetime.now(tz=UTC),
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.0045,
            model="gpt-4o",
            quality_score=0.85,
        )
        assert record.quality_score == 0.85


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_default_initialization(self):
        """Test default initialization."""
        tracker = CostTracker()
        assert tracker.budget == COST_THRESHOLDS["benchmark_budget"]
        assert tracker.budget_type == "benchmark"
        assert tracker.model == "gpt-4o"

    def test_custom_budget(self):
        """Test custom budget initialization."""
        tracker = CostTracker(budget=1.00)
        assert tracker.budget == 1.00

    def test_custom_budget_type(self):
        """Test custom budget type initialization."""
        tracker = CostTracker(budget_type="smoke_test")
        assert tracker.budget == COST_THRESHOLDS["smoke_test_budget"]
        assert tracker.budget_type == "smoke_test"

    def test_record_query(self):
        """Test recording a query."""
        tracker = CostTracker()
        metrics = tracker.record_query(input_tokens=500, output_tokens=200)

        assert metrics.input_tokens == 500
        assert metrics.output_tokens == 200
        assert metrics.total_tokens == 700
        assert metrics.cost_usd > 0
        assert tracker.query_count == 1

    def test_record_query_with_latency(self):
        """Test recording a query with latency."""
        tracker = CostTracker()
        metrics = tracker.record_query(
            input_tokens=500,
            output_tokens=200,
            latency_ms=150.5,
        )
        assert metrics.latency_ms == 150.5

    def test_record_query_with_quality_score(self):
        """Test recording a query with quality score."""
        tracker = CostTracker()
        tracker.record_query(
            input_tokens=500,
            output_tokens=200,
            quality_score=0.85,
        )
        assert tracker._records[0].quality_score == 0.85

    def test_record_text_query(self):
        """Test recording a query from text."""
        tracker = CostTracker()
        metrics = tracker.record_text_query(
            input_text="What is requirements management?",
            output_text="Requirements management is the process of...",
        )
        assert metrics.input_tokens > 0
        assert metrics.output_tokens > 0
        assert metrics.cost_usd > 0

    def test_total_cost(self):
        """Test total cost calculation."""
        tracker = CostTracker()
        tracker.record_query(500, 200)
        tracker.record_query(300, 100)
        assert tracker.total_cost > 0

    def test_total_tokens(self):
        """Test total tokens calculation."""
        tracker = CostTracker()
        tracker.record_query(500, 200)
        tracker.record_query(300, 100)
        assert tracker.total_tokens == 1100

    def test_query_count(self):
        """Test query count."""
        tracker = CostTracker()
        assert tracker.query_count == 0
        tracker.record_query(500, 200)
        assert tracker.query_count == 1
        tracker.record_query(300, 100)
        assert tracker.query_count == 2

    def test_is_within_budget_true(self):
        """Test within budget when true."""
        tracker = CostTracker(budget=10.00)
        tracker.record_query(500, 200)
        assert tracker.is_within_budget() is True

    def test_is_within_budget_false(self):
        """Test within budget when false."""
        tracker = CostTracker(budget=0.0001)
        tracker.record_query(500, 200)
        assert tracker.is_within_budget() is False

    def test_get_remaining_budget(self):
        """Test getting remaining budget."""
        tracker = CostTracker(budget=1.00)
        tracker.record_query(500, 200)
        remaining = tracker.get_remaining_budget()
        assert remaining < 1.00
        assert remaining > 0

    def test_get_remaining_budget_not_negative(self):
        """Test remaining budget is not negative."""
        tracker = CostTracker(budget=0.0001)
        tracker.record_query(500, 200)
        remaining = tracker.get_remaining_budget()
        assert remaining == 0.0

    def test_get_aggregated_metrics_empty(self):
        """Test aggregated metrics with no queries."""
        tracker = CostTracker()
        metrics = tracker.get_aggregated_metrics()

        assert metrics.query_count == 0
        assert metrics.total_tokens == 0
        assert metrics.total_cost_usd == 0.0
        assert metrics.budget_status == BudgetStatus.OK

    def test_get_aggregated_metrics(self):
        """Test aggregated metrics with queries."""
        tracker = CostTracker()
        tracker.record_query(500, 200, latency_ms=100.0)
        tracker.record_query(300, 100, latency_ms=50.0)
        metrics = tracker.get_aggregated_metrics()

        assert metrics.query_count == 2
        assert metrics.total_input_tokens == 800
        assert metrics.total_output_tokens == 300
        assert metrics.total_tokens == 1100
        assert metrics.avg_tokens_per_query == 550.0
        assert metrics.avg_latency_ms == 75.0

    def test_get_cost_efficiency(self):
        """Test getting cost efficiency metrics."""
        tracker = CostTracker()
        tracker.record_query(500, 200)
        tracker.record_query(300, 100)
        efficiency = tracker.get_cost_efficiency(avg_quality_score=0.85)

        assert efficiency.quality_score == 0.85
        assert efficiency.cost_usd == tracker.total_cost
        assert efficiency.quality_per_dollar > 0

    def test_reset(self):
        """Test resetting tracker."""
        tracker = CostTracker()
        tracker.record_query(500, 200)
        tracker.record_query(300, 100)
        assert tracker.query_count == 2

        tracker.reset()
        assert tracker.query_count == 0
        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 0


class TestComputeCostEfficiency:
    """Tests for compute_cost_efficiency function."""

    def test_normal_calculation(self):
        """Test normal cost efficiency calculation."""
        efficiency = compute_cost_efficiency(
            quality_score=0.85,
            cost=0.02,
            total_tokens=1000,
        )
        assert efficiency.quality_score == 0.85
        assert efficiency.cost_usd == 0.02
        assert efficiency.quality_per_dollar == 0.85 / 0.02
        assert efficiency.quality_per_1k_tokens == 0.85

    def test_zero_cost(self):
        """Test with zero cost."""
        efficiency = compute_cost_efficiency(
            quality_score=0.85,
            cost=0.0,
            total_tokens=1000,
        )
        assert efficiency.quality_per_dollar == 0.0

    def test_zero_tokens(self):
        """Test with zero tokens."""
        efficiency = compute_cost_efficiency(
            quality_score=0.85,
            cost=0.02,
            total_tokens=0,
        )
        assert efficiency.quality_per_1k_tokens == 0.0

    def test_zero_quality(self):
        """Test with zero quality score."""
        efficiency = compute_cost_efficiency(
            quality_score=0.0,
            cost=0.02,
            total_tokens=1000,
        )
        assert efficiency.cost_per_quality_point == 0.0


class TestCheckQueryBudget:
    """Tests for check_query_budget function."""

    def test_within_target(self):
        """Test query within target budget."""
        within_target, status, cost = check_query_budget(100, 50)
        assert within_target is True
        assert status == BudgetStatus.OK
        assert cost > 0

    def test_exceeds_target(self):
        """Test query exceeding target budget."""
        # Use large token counts to exceed target
        within_target, _status, _cost = check_query_budget(50000, 20000)
        assert within_target is False

    def test_at_hard_limit(self):
        """Test query at hard limit."""
        # Very large token counts
        _within_target, status, _cost = check_query_budget(1_000_000, 500_000)
        assert status == BudgetStatus.EXCEEDED

    def test_returns_cost(self):
        """Test that function returns cost."""
        _within_target, _status, cost = check_query_budget(500, 200, "gpt-4o")
        expected_cost = (500 / 1_000_000) * 2.50 + (200 / 1_000_000) * 10.00
        assert abs(cost - expected_cost) < 0.0001

    def test_different_model(self):
        """Test with different model."""
        _, _, cost_4o = check_query_budget(1000, 500, "gpt-4o")
        _, _, cost_mini = check_query_budget(1000, 500, "gpt-4o-mini")
        # GPT-4o-mini should be cheaper
        assert cost_mini < cost_4o
