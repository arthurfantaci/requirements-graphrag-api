"""Cost analysis for the agentic RAG system.

Provides utilities for tracking and analyzing LLM costs:
- Token counting per query
- Cost estimation per model
- Comparison between old and new RAG approaches
- Cost optimization recommendations

Usage:
    from requirements_graphrag_api.evaluation.cost_analysis import (
        CostTracker,
        estimate_cost,
    )

    tracker = CostTracker()
    tracker.record_llm_call("gpt-4o-mini", input_tokens=500, output_tokens=200)

    report = tracker.get_report()
    print(f"Total cost: ${report['total_cost']:.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


# Pricing per 1M tokens (as of 2025)
# https://openai.com/pricing
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {
        "input": 2.50,  # $2.50 per 1M input tokens
        "output": 10.00,  # $10.00 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": 0.15,  # $0.15 per 1M input tokens
        "output": 0.60,  # $0.60 per 1M output tokens
    },
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00,
    },
    "gpt-4": {
        "input": 30.00,
        "output": 60.00,
    },
    "gpt-3.5-turbo": {
        "input": 0.50,
        "output": 1.50,
    },
    "text-embedding-3-small": {
        "input": 0.02,
        "output": 0.0,  # No output tokens for embeddings
    },
    "text-embedding-3-large": {
        "input": 0.13,
        "output": 0.0,
    },
    # Claude models (Anthropic)
    "claude-3-5-sonnet": {
        "input": 3.00,
        "output": 15.00,
    },
    "claude-3-opus": {
        "input": 15.00,
        "output": 75.00,
    },
    "claude-3-haiku": {
        "input": 0.25,
        "output": 1.25,
    },
}


@dataclass
class LLMCall:
    """Record of a single LLM API call."""

    model: str
    input_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    operation: str = ""  # e.g., "rag_generation", "query_expansion"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> float:
        """Estimated cost in USD."""
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)


@dataclass
class CostReport:
    """Cost analysis report."""

    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost: float
    cost_by_model: dict[str, float]
    cost_by_operation: dict[str, float]
    calls_by_model: dict[str, int]
    calls_by_operation: dict[str, int]
    avg_tokens_per_call: float
    avg_cost_per_call: float
    recommendations: list[str]


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Estimate cost for a given model and token count.

    Args:
        model: Model name (e.g., "gpt-4o-mini").
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    # Normalize model name
    model_lower = model.lower()

    # Find matching pricing - prefer exact matches and longer model names first
    pricing = None

    # First try exact match
    if model_lower in MODEL_PRICING:
        pricing = MODEL_PRICING[model_lower]
    else:
        # Sort by key length descending to match "gpt-4o-mini" before "gpt-4o"
        sorted_keys = sorted(MODEL_PRICING.keys(), key=len, reverse=True)
        for model_key in sorted_keys:
            if model_key in model_lower or model_lower in model_key:
                pricing = MODEL_PRICING[model_key]
                break

    if pricing is None:
        logger.warning("No pricing found for model: %s, using gpt-4o-mini as default", model)
        pricing = MODEL_PRICING["gpt-4o-mini"]

    # Calculate cost (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


class CostTracker:
    """Track LLM costs across multiple calls.

    Example:
        tracker = CostTracker()

        # Record calls during execution
        tracker.record_llm_call(
            model="gpt-4o-mini",
            input_tokens=500,
            output_tokens=200,
            operation="rag_generation"
        )

        # Get report
        report = tracker.get_report()
        print(f"Total cost: ${report.total_cost:.4f}")
    """

    def __init__(self) -> None:
        """Initialize the cost tracker."""
        self._calls: list[LLMCall] = []

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
        operation: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> LLMCall:
        """Record an LLM API call.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            operation: Operation name (e.g., "rag_generation").
            metadata: Optional metadata.

        Returns:
            The recorded LLMCall.
        """
        call = LLMCall(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            operation=operation,
            metadata=metadata or {},
        )
        self._calls.append(call)
        return call

    def record_from_response(
        self,
        model: str,
        response: Any,
        operation: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> LLMCall | None:
        """Record call from LLM response with usage info.

        Extracts token counts from OpenAI/LangChain response objects.

        Args:
            model: Model name.
            response: LLM response object.
            operation: Operation name.
            metadata: Optional metadata.

        Returns:
            The recorded LLMCall or None if no usage info found.
        """
        # Try to extract usage from various response formats
        input_tokens = 0
        output_tokens = 0

        # OpenAI response format
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "prompt_tokens"):
                input_tokens = usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                output_tokens = usage.completion_tokens

        # LangChain AIMessage format
        elif hasattr(response, "response_metadata"):
            meta = response.response_metadata
            if "token_usage" in meta:
                usage = meta["token_usage"]
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

        # Dict format
        elif isinstance(response, dict):
            if "usage" in response:
                usage = response["usage"]
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

        if input_tokens == 0 and output_tokens == 0:
            logger.warning("Could not extract usage from response")
            return None

        return self.record_llm_call(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            operation=operation,
            metadata=metadata,
        )

    def get_calls(self) -> list[LLMCall]:
        """Get all recorded calls.

        Returns:
            List of LLMCall records.
        """
        return self._calls.copy()

    def get_report(self) -> CostReport:
        """Generate a cost analysis report.

        Returns:
            CostReport with totals, breakdowns, and recommendations.
        """
        if not self._calls:
            return CostReport(
                total_calls=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                total_cost=0.0,
                cost_by_model={},
                cost_by_operation={},
                calls_by_model={},
                calls_by_operation={},
                avg_tokens_per_call=0.0,
                avg_cost_per_call=0.0,
                recommendations=[],
            )

        total_input = sum(c.input_tokens for c in self._calls)
        total_output = sum(c.output_tokens for c in self._calls)
        total_cost = sum(c.cost for c in self._calls)

        # Group by model
        cost_by_model: dict[str, float] = {}
        calls_by_model: dict[str, int] = {}
        for call in self._calls:
            cost_by_model[call.model] = cost_by_model.get(call.model, 0) + call.cost
            calls_by_model[call.model] = calls_by_model.get(call.model, 0) + 1

        # Group by operation
        cost_by_operation: dict[str, float] = {}
        calls_by_operation: dict[str, int] = {}
        for call in self._calls:
            op = call.operation or "unknown"
            cost_by_operation[op] = cost_by_operation.get(op, 0) + call.cost
            calls_by_operation[op] = calls_by_operation.get(op, 0) + 1

        # Generate recommendations
        recommendations = self._generate_recommendations(
            cost_by_model, cost_by_operation, total_cost
        )

        return CostReport(
            total_calls=len(self._calls),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            total_cost=total_cost,
            cost_by_model=cost_by_model,
            cost_by_operation=cost_by_operation,
            calls_by_model=calls_by_model,
            calls_by_operation=calls_by_operation,
            avg_tokens_per_call=(total_input + total_output) / len(self._calls),
            avg_cost_per_call=total_cost / len(self._calls),
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        cost_by_model: dict[str, float],
        cost_by_operation: dict[str, float],
        total_cost: float,
    ) -> list[str]:
        """Generate cost optimization recommendations.

        Args:
            cost_by_model: Cost breakdown by model.
            cost_by_operation: Cost breakdown by operation.
            total_cost: Total cost.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        # Check for expensive models
        for model, cost in cost_by_model.items():
            is_expensive_gpt4o = (
                "gpt-4o" in model.lower()
                and "mini" not in model.lower()
                and cost > total_cost * 0.3
            )
            if is_expensive_gpt4o:
                recommendations.append(
                    f"Model {model} accounts for {cost / total_cost * 100:.1f}% of costs. "
                    "Consider using gpt-4o-mini for non-critical operations."
                )

            if "gpt-4-turbo" in model.lower() or model.lower() == "gpt-4":
                recommendations.append(
                    f"Model {model} is expensive. Consider gpt-4o or gpt-4o-mini."
                )

        # Check for operations that could be optimized
        for op, cost in cost_by_operation.items():
            if cost > total_cost * 0.4:
                recommendations.append(
                    f"Operation '{op}' accounts for {cost / total_cost * 100:.1f}% of costs. "
                    "Consider caching or prompt optimization."
                )

        # General recommendations based on total cost
        if total_cost > 0.10:
            recommendations.append(
                "Query cost exceeds $0.10. Review if all LLM calls are necessary."
            )

        return recommendations

    def compare_with_baseline(self, baseline_tracker: CostTracker) -> dict[str, Any]:
        """Compare costs with a baseline tracker.

        Args:
            baseline_tracker: Tracker with baseline costs (e.g., old RAG).

        Returns:
            Comparison report dict.
        """
        current_report = self.get_report()
        baseline_report = baseline_tracker.get_report()

        cost_diff = current_report.total_cost - baseline_report.total_cost
        cost_ratio = (
            current_report.total_cost / baseline_report.total_cost
            if baseline_report.total_cost > 0
            else float("inf")
        )

        token_diff = current_report.total_tokens - baseline_report.total_tokens
        calls_diff = current_report.total_calls - baseline_report.total_calls

        return {
            "current_cost": current_report.total_cost,
            "baseline_cost": baseline_report.total_cost,
            "cost_difference": cost_diff,
            "cost_ratio": cost_ratio,
            "cost_change_percent": (cost_ratio - 1) * 100 if cost_ratio != float("inf") else None,
            "current_tokens": current_report.total_tokens,
            "baseline_tokens": baseline_report.total_tokens,
            "token_difference": token_diff,
            "current_calls": current_report.total_calls,
            "baseline_calls": baseline_report.total_calls,
            "calls_difference": calls_diff,
            "verdict": (
                "more_expensive" if cost_diff > 0 else "less_expensive" if cost_diff < 0 else "same"
            ),
        }

    def reset(self) -> None:
        """Reset all recorded calls."""
        self._calls.clear()


# Global tracker instance
_global_cost_tracker: CostTracker | None = None


def get_global_cost_tracker() -> CostTracker:
    """Get the global cost tracker.

    Returns:
        Global CostTracker instance.
    """
    global _global_cost_tracker
    if _global_cost_tracker is None:
        _global_cost_tracker = CostTracker()
    return _global_cost_tracker


def reset_global_cost_tracker() -> None:
    """Reset the global cost tracker."""
    global _global_cost_tracker
    if _global_cost_tracker is not None:
        _global_cost_tracker.reset()
    _global_cost_tracker = None


def get_cost_report() -> CostReport:
    """Get cost report from global tracker.

    Returns:
        CostReport instance.
    """
    return get_global_cost_tracker().get_report()


__all__ = [
    "MODEL_PRICING",
    "CostReport",
    "CostTracker",
    "LLMCall",
    "estimate_cost",
    "get_cost_report",
    "get_global_cost_tracker",
    "reset_global_cost_tracker",
]  # Sorted alphabetically
