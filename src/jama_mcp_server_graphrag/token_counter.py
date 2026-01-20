"""Token counting utilities for cost tracking and budget management.

Provides token counting for OpenAI models using tiktoken, with support for:
- Counting tokens in text, messages, and prompts
- Estimating costs based on model pricing
- Budget tracking and alerting
- Integration with LangChain components

Usage:
    from jama_mcp_server_graphrag.token_counter import (
        TokenCounter,
        count_tokens,
        estimate_cost,
        COST_THRESHOLDS,
    )

    # Count tokens in text
    count = count_tokens("Hello, world!", model="gpt-4o")

    # Estimate cost for a query
    cost = estimate_cost(input_tokens=500, output_tokens=200, model="gpt-4o")

    # Check budget status
    counter = TokenCounter()
    counter.add_usage(input_tokens=500, output_tokens=200)
    status = counter.get_budget_status()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

import tiktoken

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


# =============================================================================
# COST THRESHOLDS
# =============================================================================

COST_THRESHOLDS: Final[dict[str, float]] = {
    "query_budget_target": 0.015,  # Target per query ($0.015)
    "query_budget_warning": 0.025,  # Warning threshold ($0.025)
    "query_budget_alert": 0.040,  # Alert threshold ($0.040)
    "query_budget_hard_limit": 0.100,  # Hard limit ($0.10)
    "smoke_test_budget": 0.50,  # CI Tier 2 ($0.50)
    "benchmark_budget": 5.00,  # CI Tier 3 ($5.00)
    "full_eval_budget": 15.00,  # CI Tier 4 ($15.00)
}


# =============================================================================
# MODEL PRICING (per 1M tokens, as of 2024)
# =============================================================================

@dataclass(frozen=True)
class ModelPricing:
    """Pricing information for a model.

    Attributes:
        input_per_million: Cost per million input tokens.
        output_per_million: Cost per million output tokens.
        encoding_name: Tiktoken encoding name for this model.
    """

    input_per_million: float
    output_per_million: float
    encoding_name: str = "cl100k_base"


MODEL_PRICING: Final[dict[str, ModelPricing]] = {
    # GPT-4o family
    "gpt-4o": ModelPricing(2.50, 10.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4o-2024-11-20": ModelPricing(2.50, 10.00),
    "gpt-4o-2024-08-06": ModelPricing(2.50, 10.00),
    # GPT-4 Turbo
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-preview": ModelPricing(10.00, 30.00),
    # GPT-4
    "gpt-4": ModelPricing(30.00, 60.00),
    "gpt-4-32k": ModelPricing(60.00, 120.00),
    # GPT-3.5
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-16k": ModelPricing(3.00, 4.00),
    # Embeddings
    "text-embedding-3-small": ModelPricing(0.02, 0.0),
    "text-embedding-3-large": ModelPricing(0.13, 0.0),
    "text-embedding-ada-002": ModelPricing(0.10, 0.0),
}

# Default model for token counting
DEFAULT_MODEL: Final[str] = "gpt-4o"


class BudgetStatus(StrEnum):
    """Budget status levels."""

    OK = "ok"
    WARNING = "warning"
    ALERT = "alert"
    EXCEEDED = "exceeded"


# =============================================================================
# TOKEN COUNTING FUNCTIONS
# =============================================================================


def get_encoding(model: str = DEFAULT_MODEL) -> tiktoken.Encoding:
    """Get the tiktoken encoding for a model.

    Args:
        model: Model name.

    Returns:
        Tiktoken Encoding instance.
    """
    pricing = MODEL_PRICING.get(model)
    encoding_name = pricing.encoding_name if pricing else "cl100k_base"

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """Count tokens in a text string.

    Args:
        text: Text to count tokens in.
        model: Model name for tokenizer selection.

    Returns:
        Number of tokens.
    """
    encoding = get_encoding(model)
    return len(encoding.encode(text))


def count_message_tokens(
    messages: list[dict[str, str] | BaseMessage],
    model: str = DEFAULT_MODEL,
) -> int:
    """Count tokens in a list of chat messages.

    Accounts for message formatting overhead used by OpenAI models.

    Args:
        messages: List of messages (dicts or BaseMessage objects).
        model: Model name.

    Returns:
        Total token count including formatting overhead.
    """
    encoding = get_encoding(model)

    # Message formatting overhead varies by model
    # For gpt-4o and gpt-3.5-turbo, each message adds ~4 tokens
    tokens_per_message = 4
    tokens_per_name = -1  # If name is present, role is omitted

    total_tokens = 0
    for message in messages:
        total_tokens += tokens_per_message

        # Handle both dict and BaseMessage formats
        if hasattr(message, "content"):
            content = message.content
            role = getattr(message, "type", "user")
        else:
            content = message.get("content", "")
            role = message.get("role", "user")

        total_tokens += len(encoding.encode(str(content)))
        total_tokens += len(encoding.encode(role))

        # Check for name field
        name = message.get("name") if isinstance(message, dict) else None
        if name:
            total_tokens += len(encoding.encode(name))
            total_tokens += tokens_per_name

    # Every reply is primed with assistant
    total_tokens += 3

    return total_tokens


def count_prompt_tokens(
    prompt: ChatPromptTemplate,
    variables: dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> int:
    """Count tokens in a formatted prompt template.

    Args:
        prompt: ChatPromptTemplate to format.
        variables: Variables to format the prompt with.
        model: Model name.

    Returns:
        Token count of the formatted prompt.
    """
    try:
        messages = prompt.format_messages(**variables)
        return count_message_tokens(messages, model)
    except Exception as e:
        logger.warning("Failed to count prompt tokens: %s", e)
        # Fallback: estimate from template string
        return count_tokens(str(prompt), model)


# =============================================================================
# COST ESTIMATION
# =============================================================================


def estimate_cost(
    input_tokens: int,
    output_tokens: int = 0,
    model: str = DEFAULT_MODEL,
) -> float:
    """Estimate cost for token usage.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model name.

    Returns:
        Estimated cost in USD.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        logger.warning("Unknown model pricing for %s, using default", model)
        pricing = MODEL_PRICING[DEFAULT_MODEL]

    input_cost = (input_tokens / 1_000_000) * pricing.input_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_million

    return input_cost + output_cost


def get_budget_status(cost: float, budget_type: str = "query") -> BudgetStatus:  # noqa: PLR0911
    """Get budget status for a cost amount.

    Args:
        cost: Cost in USD.
        budget_type: Type of budget ("query", "smoke_test", "benchmark", "full_eval").

    Returns:
        BudgetStatus indicating the cost level.
    """
    if budget_type == "query":
        if cost >= COST_THRESHOLDS["query_budget_hard_limit"]:
            return BudgetStatus.EXCEEDED
        if cost >= COST_THRESHOLDS["query_budget_alert"]:
            return BudgetStatus.ALERT
        if cost >= COST_THRESHOLDS["query_budget_warning"]:
            return BudgetStatus.WARNING
        return BudgetStatus.OK

    # For other budget types, compare against their limit
    limit_key = f"{budget_type}_budget"
    limit = COST_THRESHOLDS.get(limit_key, COST_THRESHOLDS["query_budget_hard_limit"])

    if cost >= limit:
        return BudgetStatus.EXCEEDED
    if cost >= limit * 0.9:
        return BudgetStatus.ALERT
    if cost >= limit * 0.75:
        return BudgetStatus.WARNING
    return BudgetStatus.OK


# =============================================================================
# TOKEN COUNTER CLASS
# =============================================================================


@dataclass
class UsageRecord:
    """Record of token usage for a single operation.

    Attributes:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model used.
        cost: Estimated cost.
        operation: Description of the operation.
    """

    input_tokens: int
    output_tokens: int
    model: str
    cost: float
    operation: str = ""


@dataclass
class TokenCounter:
    """Tracks token usage and costs across multiple operations.

    Provides cumulative tracking with budget status monitoring.

    Attributes:
        model: Default model for pricing.
        budget_limit: Budget limit in USD.
        budget_type: Type of budget for status calculation.
    """

    model: str = DEFAULT_MODEL
    budget_limit: float = COST_THRESHOLDS["query_budget_hard_limit"]
    budget_type: str = "query"
    _records: list[UsageRecord] = field(default_factory=list)

    def add_usage(
        self,
        input_tokens: int,
        output_tokens: int = 0,
        model: str | None = None,
        operation: str = "",
    ) -> UsageRecord:
        """Record token usage.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model used (defaults to counter's model).
            operation: Description of the operation.

        Returns:
            UsageRecord for this operation.
        """
        model = model or self.model
        cost = estimate_cost(input_tokens, output_tokens, model)

        record = UsageRecord(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            cost=cost,
            operation=operation,
        )
        self._records.append(record)

        return record

    def add_text_usage(
        self,
        input_text: str,
        output_text: str = "",
        model: str | None = None,
        operation: str = "",
    ) -> UsageRecord:
        """Record usage by counting tokens in text.

        Args:
            input_text: Input text.
            output_text: Output text.
            model: Model used.
            operation: Description of the operation.

        Returns:
            UsageRecord for this operation.
        """
        model = model or self.model
        input_tokens = count_tokens(input_text, model)
        output_tokens = count_tokens(output_text, model) if output_text else 0

        return self.add_usage(input_tokens, output_tokens, model, operation)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all records."""
        return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all records."""
        return sum(r.output_tokens for r in self._records)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all records."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost(self) -> float:
        """Total cost across all records."""
        return sum(r.cost for r in self._records)

    @property
    def record_count(self) -> int:
        """Number of usage records."""
        return len(self._records)

    def get_budget_status(self) -> BudgetStatus:
        """Get current budget status.

        Returns:
            BudgetStatus based on total cost vs budget limit.
        """
        return get_budget_status(self.total_cost, self.budget_type)

    def is_within_budget(self) -> bool:
        """Check if total cost is within budget.

        Returns:
            True if within budget, False otherwise.
        """
        return self.total_cost < self.budget_limit

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of token usage.

        Returns:
            Dictionary with usage statistics.
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "record_count": self.record_count,
            "budget_limit": self.budget_limit,
            "budget_remaining": round(self.budget_limit - self.total_cost, 6),
            "budget_status": self.get_budget_status().value,
            "within_budget": self.is_within_budget(),
        }

    def reset(self) -> None:
        """Reset all usage records."""
        self._records.clear()


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_counter: TokenCounter | None = None


def get_token_counter() -> TokenCounter:
    """Get the global token counter singleton.

    Returns:
        TokenCounter instance.
    """
    global _counter  # noqa: PLW0603
    if _counter is None:
        _counter = TokenCounter()
    return _counter


def reset_token_counter() -> None:
    """Reset the global token counter."""
    if _counter is not None:
        _counter.reset()


__all__ = [
    "COST_THRESHOLDS",
    "DEFAULT_MODEL",
    "MODEL_PRICING",
    "BudgetStatus",
    "ModelPricing",
    "TokenCounter",
    "UsageRecord",
    "count_message_tokens",
    "count_prompt_tokens",
    "count_tokens",
    "estimate_cost",
    "get_budget_status",
    "get_encoding",
    "get_token_counter",
    "reset_token_counter",
]
