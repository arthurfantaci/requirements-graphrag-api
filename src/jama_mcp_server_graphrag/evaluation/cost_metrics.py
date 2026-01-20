"""Cost tracking metrics for RAG evaluation.

Provides cost-aware evaluation metrics that track token usage and costs
alongside quality metrics, enabling budget-constrained evaluation.

Features:
- Per-query cost tracking
- Budget enforcement and alerting
- Cost efficiency metrics (quality per dollar)
- Integration with evaluation runner

Usage:
    from jama_mcp_server_graphrag.evaluation.cost_metrics import (
        CostMetrics,
        CostTracker,
        compute_cost_efficiency,
    )

    # Track costs during evaluation
    tracker = CostTracker(budget=5.00)
    tracker.record_query(input_tokens=500, output_tokens=200)

    # Get cost efficiency metrics
    efficiency = compute_cost_efficiency(quality_score=0.85, cost=0.02)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from jama_mcp_server_graphrag.token_counter import (
    COST_THRESHOLDS,
    BudgetStatus,
    TokenCounter,
    count_tokens,
    estimate_cost,
    get_budget_status,
)

logger = logging.getLogger(__name__)


# =============================================================================
# COST METRICS DATACLASS
# =============================================================================


@dataclass
class CostMetrics:
    """Cost metrics for a single evaluation query.

    Attributes:
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.
        total_tokens: Total tokens (input + output).
        cost_usd: Total cost in USD.
        model: Model used for the query.
        latency_ms: Query latency in milliseconds.
        budget_status: Budget status after this query.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    model: str
    latency_ms: float = 0.0
    budget_status: BudgetStatus = BudgetStatus.OK

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "budget_status": self.budget_status.value,
        }


@dataclass
class CostEfficiencyMetrics:
    """Cost efficiency metrics combining quality and cost.

    Attributes:
        quality_score: Quality metric (0-1).
        cost_usd: Cost in USD.
        quality_per_dollar: Quality score per dollar spent.
        quality_per_1k_tokens: Quality score per 1000 tokens.
        cost_per_quality_point: Cost per 0.01 quality improvement.
    """

    quality_score: float
    cost_usd: float
    quality_per_dollar: float
    quality_per_1k_tokens: float
    cost_per_quality_point: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary format."""
        return {
            "quality_score": round(self.quality_score, 4),
            "cost_usd": round(self.cost_usd, 6),
            "quality_per_dollar": round(self.quality_per_dollar, 2),
            "quality_per_1k_tokens": round(self.quality_per_1k_tokens, 4),
            "cost_per_quality_point": round(self.cost_per_quality_point, 6),
        }


@dataclass
class AggregatedCostMetrics:
    """Aggregated cost metrics across multiple queries.

    Attributes:
        query_count: Number of queries tracked.
        total_input_tokens: Sum of input tokens.
        total_output_tokens: Sum of output tokens.
        total_tokens: Sum of all tokens.
        total_cost_usd: Total cost in USD.
        avg_cost_per_query: Average cost per query.
        avg_tokens_per_query: Average tokens per query.
        avg_latency_ms: Average latency in milliseconds.
        budget_used_pct: Percentage of budget used.
        budget_status: Overall budget status.
    """

    query_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    avg_cost_per_query: float
    avg_tokens_per_query: float
    avg_latency_ms: float
    budget_used_pct: float
    budget_status: BudgetStatus

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "query_count": self.query_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_cost_per_query": round(self.avg_cost_per_query, 6),
            "avg_tokens_per_query": round(self.avg_tokens_per_query, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "budget_used_pct": round(self.budget_used_pct, 2),
            "budget_status": self.budget_status.value,
        }


# =============================================================================
# COST TRACKER
# =============================================================================


@dataclass
class QueryCostRecord:
    """Record of a single query's cost data.

    Attributes:
        timestamp: When the query was executed.
        input_tokens: Input token count.
        output_tokens: Output token count.
        cost_usd: Query cost.
        model: Model used.
        latency_ms: Query latency.
        quality_score: Optional quality score.
    """

    timestamp: datetime
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    latency_ms: float = 0.0
    quality_score: float | None = None


class CostTracker:
    """Tracks costs across multiple evaluation queries.

    Provides budget management, alerting, and aggregated statistics
    for evaluation runs.

    Attributes:
        budget: Budget limit in USD.
        budget_type: Type of budget for threshold calculation.
        model: Default model for cost estimation.
    """

    def __init__(
        self,
        budget: float | None = None,
        budget_type: str = "benchmark",
        model: str = "gpt-4o",
    ) -> None:
        """Initialize the cost tracker.

        Args:
            budget: Budget limit in USD. Defaults to benchmark budget.
            budget_type: Type of budget ("query", "smoke_test", "benchmark", "full_eval").
            model: Default model for cost estimation.
        """
        self.budget = budget or COST_THRESHOLDS.get(
            f"{budget_type}_budget",
            COST_THRESHOLDS["benchmark_budget"],
        )
        self.budget_type = budget_type
        self.model = model
        self._records: list[QueryCostRecord] = []
        self._token_counter = TokenCounter(model=model, budget_limit=self.budget)

    def record_query(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
        latency_ms: float = 0.0,
        quality_score: float | None = None,
    ) -> CostMetrics:
        """Record a query's cost metrics.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model used (defaults to tracker's model).
            latency_ms: Query latency in milliseconds.
            quality_score: Optional quality score for this query.

        Returns:
            CostMetrics for this query.
        """
        model = model or self.model
        cost = estimate_cost(input_tokens, output_tokens, model)

        record = QueryCostRecord(
            timestamp=datetime.now(tz=UTC),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model=model,
            latency_ms=latency_ms,
            quality_score=quality_score,
        )
        self._records.append(record)

        # Also track in token counter
        self._token_counter.add_usage(input_tokens, output_tokens, model)

        # Check budget status
        status = self._get_current_status()

        # Log warnings for budget issues
        if status == BudgetStatus.EXCEEDED:
            logger.warning(
                "Budget exceeded: $%.4f / $%.2f",
                self.total_cost,
                self.budget,
            )
        elif status == BudgetStatus.ALERT:
            logger.warning(
                "Budget alert: $%.4f / $%.2f (%.1f%% used)",
                self.total_cost,
                self.budget,
                (self.total_cost / self.budget) * 100,
            )

        return CostMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            model=model,
            latency_ms=latency_ms,
            budget_status=status,
        )

    def record_text_query(
        self,
        input_text: str,
        output_text: str,
        model: str | None = None,
        latency_ms: float = 0.0,
        quality_score: float | None = None,
    ) -> CostMetrics:
        """Record a query by counting tokens in text.

        Args:
            input_text: Input text.
            output_text: Output text.
            model: Model used.
            latency_ms: Query latency.
            quality_score: Optional quality score.

        Returns:
            CostMetrics for this query.
        """
        model = model or self.model
        input_tokens = count_tokens(input_text, model)
        output_tokens = count_tokens(output_text, model)

        return self.record_query(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            latency_ms=latency_ms,
            quality_score=quality_score,
        )

    @property
    def total_cost(self) -> float:
        """Total cost across all recorded queries."""
        return sum(r.cost_usd for r in self._records)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all recorded queries."""
        return sum(r.input_tokens + r.output_tokens for r in self._records)

    @property
    def query_count(self) -> int:
        """Number of queries recorded."""
        return len(self._records)

    def _get_current_status(self) -> BudgetStatus:
        """Get current budget status."""
        if self.total_cost >= self.budget:
            return BudgetStatus.EXCEEDED
        if self.total_cost >= self.budget * 0.9:
            return BudgetStatus.ALERT
        if self.total_cost >= self.budget * 0.75:
            return BudgetStatus.WARNING
        return BudgetStatus.OK

    def is_within_budget(self) -> bool:
        """Check if still within budget.

        Returns:
            True if total cost is below budget limit.
        """
        return self.total_cost < self.budget

    def get_remaining_budget(self) -> float:
        """Get remaining budget.

        Returns:
            Remaining budget in USD.
        """
        return max(0.0, self.budget - self.total_cost)

    def get_aggregated_metrics(self) -> AggregatedCostMetrics:
        """Get aggregated metrics across all queries.

        Returns:
            AggregatedCostMetrics with summary statistics.
        """
        if not self._records:
            return AggregatedCostMetrics(
                query_count=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                total_cost_usd=0.0,
                avg_cost_per_query=0.0,
                avg_tokens_per_query=0.0,
                avg_latency_ms=0.0,
                budget_used_pct=0.0,
                budget_status=BudgetStatus.OK,
            )

        total_input = sum(r.input_tokens for r in self._records)
        total_output = sum(r.output_tokens for r in self._records)
        total_latency = sum(r.latency_ms for r in self._records)
        count = len(self._records)

        return AggregatedCostMetrics(
            query_count=count,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            total_cost_usd=self.total_cost,
            avg_cost_per_query=self.total_cost / count,
            avg_tokens_per_query=(total_input + total_output) / count,
            avg_latency_ms=total_latency / count,
            budget_used_pct=(self.total_cost / self.budget) * 100,
            budget_status=self._get_current_status(),
        )

    def get_cost_efficiency(self, avg_quality_score: float) -> CostEfficiencyMetrics:
        """Calculate cost efficiency metrics.

        Args:
            avg_quality_score: Average quality score (0-1).

        Returns:
            CostEfficiencyMetrics combining quality and cost.
        """
        return compute_cost_efficiency(
            quality_score=avg_quality_score,
            cost=self.total_cost,
            total_tokens=self.total_tokens,
        )

    def reset(self) -> None:
        """Reset all recorded data."""
        self._records.clear()
        self._token_counter.reset()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_cost_efficiency(
    quality_score: float,
    cost: float,
    total_tokens: int = 0,
) -> CostEfficiencyMetrics:
    """Compute cost efficiency metrics.

    Args:
        quality_score: Quality score (0-1).
        cost: Total cost in USD.
        total_tokens: Total tokens used.

    Returns:
        CostEfficiencyMetrics with efficiency calculations.
    """
    # Avoid division by zero
    quality_per_dollar = quality_score / cost if cost > 0 else 0.0
    quality_per_1k_tokens = (quality_score / (total_tokens / 1000)) if total_tokens > 0 else 0.0
    cost_per_quality_point = (cost / (quality_score * 100)) if quality_score > 0 else 0.0

    return CostEfficiencyMetrics(
        quality_score=quality_score,
        cost_usd=cost,
        quality_per_dollar=quality_per_dollar,
        quality_per_1k_tokens=quality_per_1k_tokens,
        cost_per_quality_point=cost_per_quality_point,
    )


def check_query_budget(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o",
) -> tuple[bool, BudgetStatus, float]:
    """Check if a query is within budget thresholds.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model name.

    Returns:
        Tuple of (within_target, status, cost).
    """
    cost = estimate_cost(input_tokens, output_tokens, model)
    status = get_budget_status(cost, "query")
    within_target = cost <= COST_THRESHOLDS["query_budget_target"]

    return within_target, status, cost


__all__ = [
    "AggregatedCostMetrics",
    "CostEfficiencyMetrics",
    "CostMetrics",
    "CostTracker",
    "QueryCostRecord",
    "check_query_budget",
    "compute_cost_efficiency",
]
