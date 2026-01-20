"""Tests for token counting utilities."""

from __future__ import annotations

import pytest

from jama_mcp_server_graphrag.token_counter import (
    COST_THRESHOLDS,
    DEFAULT_MODEL,
    MODEL_PRICING,
    BudgetStatus,
    ModelPricing,
    TokenCounter,
    UsageRecord,
    count_message_tokens,
    count_tokens,
    estimate_cost,
    get_budget_status,
    get_encoding,
    get_token_counter,
    reset_token_counter,
)


class TestCostThresholds:
    """Tests for COST_THRESHOLDS constant."""

    def test_query_budget_target_exists(self):
        """Test query budget target is defined."""
        assert "query_budget_target" in COST_THRESHOLDS
        assert COST_THRESHOLDS["query_budget_target"] == 0.015

    def test_query_budget_warning_exists(self):
        """Test query budget warning threshold is defined."""
        assert "query_budget_warning" in COST_THRESHOLDS
        assert COST_THRESHOLDS["query_budget_warning"] == 0.025

    def test_query_budget_alert_exists(self):
        """Test query budget alert threshold is defined."""
        assert "query_budget_alert" in COST_THRESHOLDS
        assert COST_THRESHOLDS["query_budget_alert"] == 0.040

    def test_query_budget_hard_limit_exists(self):
        """Test query budget hard limit is defined."""
        assert "query_budget_hard_limit" in COST_THRESHOLDS
        assert COST_THRESHOLDS["query_budget_hard_limit"] == 0.100

    def test_smoke_test_budget_exists(self):
        """Test smoke test budget is defined."""
        assert "smoke_test_budget" in COST_THRESHOLDS
        assert COST_THRESHOLDS["smoke_test_budget"] == 0.50

    def test_benchmark_budget_exists(self):
        """Test benchmark budget is defined."""
        assert "benchmark_budget" in COST_THRESHOLDS
        assert COST_THRESHOLDS["benchmark_budget"] == 5.00

    def test_full_eval_budget_exists(self):
        """Test full eval budget is defined."""
        assert "full_eval_budget" in COST_THRESHOLDS
        assert COST_THRESHOLDS["full_eval_budget"] == 15.00

    def test_thresholds_are_ordered(self):
        """Test that thresholds are in ascending order."""
        assert COST_THRESHOLDS["query_budget_target"] < COST_THRESHOLDS["query_budget_warning"]
        assert COST_THRESHOLDS["query_budget_warning"] < COST_THRESHOLDS["query_budget_alert"]
        assert COST_THRESHOLDS["query_budget_alert"] < COST_THRESHOLDS["query_budget_hard_limit"]


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    def test_create_pricing(self):
        """Test creating a model pricing instance."""
        pricing = ModelPricing(input_per_million=2.50, output_per_million=10.00)
        assert pricing.input_per_million == 2.50
        assert pricing.output_per_million == 10.00
        assert pricing.encoding_name == "cl100k_base"

    def test_custom_encoding(self):
        """Test custom encoding name."""
        pricing = ModelPricing(
            input_per_million=1.00,
            output_per_million=2.00,
            encoding_name="custom_encoding",
        )
        assert pricing.encoding_name == "custom_encoding"

    def test_pricing_is_frozen(self):
        """Test that pricing is immutable."""
        pricing = ModelPricing(input_per_million=2.50, output_per_million=10.00)
        with pytest.raises(AttributeError):
            pricing.input_per_million = 5.00


class TestModelPricingDict:
    """Tests for MODEL_PRICING dictionary."""

    def test_gpt4o_pricing(self):
        """Test GPT-4o pricing is defined."""
        assert "gpt-4o" in MODEL_PRICING
        assert MODEL_PRICING["gpt-4o"].input_per_million == 2.50
        assert MODEL_PRICING["gpt-4o"].output_per_million == 10.00

    def test_gpt4o_mini_pricing(self):
        """Test GPT-4o-mini pricing is defined."""
        assert "gpt-4o-mini" in MODEL_PRICING
        assert MODEL_PRICING["gpt-4o-mini"].input_per_million == 0.15
        assert MODEL_PRICING["gpt-4o-mini"].output_per_million == 0.60

    def test_embedding_models_have_no_output_cost(self):
        """Test embedding models have zero output cost."""
        embedding_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]
        for model in embedding_models:
            assert model in MODEL_PRICING
            assert MODEL_PRICING[model].output_per_million == 0.0

    def test_default_model_has_pricing(self):
        """Test default model has pricing defined."""
        assert DEFAULT_MODEL in MODEL_PRICING


class TestBudgetStatus:
    """Tests for BudgetStatus enum."""

    def test_ok_value(self):
        """Test OK status value."""
        assert BudgetStatus.OK.value == "ok"

    def test_warning_value(self):
        """Test WARNING status value."""
        assert BudgetStatus.WARNING.value == "warning"

    def test_alert_value(self):
        """Test ALERT status value."""
        assert BudgetStatus.ALERT.value == "alert"

    def test_exceeded_value(self):
        """Test EXCEEDED status value."""
        assert BudgetStatus.EXCEEDED.value == "exceeded"


class TestGetEncoding:
    """Tests for get_encoding function."""

    def test_returns_encoding_for_known_model(self):
        """Test encoding is returned for known model."""
        encoding = get_encoding("gpt-4o")
        assert encoding is not None

    def test_returns_encoding_for_unknown_model(self):
        """Test fallback encoding for unknown model."""
        encoding = get_encoding("unknown-model")
        assert encoding is not None

    def test_default_model_encoding(self):
        """Test encoding for default model."""
        encoding = get_encoding()
        assert encoding is not None


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_empty_string(self):
        """Test counting tokens in empty string."""
        count = count_tokens("")
        assert count == 0

    def test_simple_text(self):
        """Test counting tokens in simple text."""
        count = count_tokens("Hello, world!")
        assert count > 0

    def test_longer_text(self):
        """Test that longer text has more tokens."""
        short_count = count_tokens("Hello")
        long_count = count_tokens("Hello, this is a much longer piece of text.")
        assert long_count > short_count

    def test_consistent_counts(self):
        """Test that same text gives same count."""
        text = "The quick brown fox jumps over the lazy dog."
        count1 = count_tokens(text)
        count2 = count_tokens(text)
        assert count1 == count2


class TestCountMessageTokens:
    """Tests for count_message_tokens function."""

    def test_single_message_dict(self):
        """Test counting tokens in single message dict."""
        messages = [{"role": "user", "content": "Hello!"}]
        count = count_message_tokens(messages)
        assert count > 0

    def test_multiple_messages(self):
        """Test counting tokens in multiple messages."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = count_message_tokens(messages)
        assert count > count_message_tokens([messages[0]])

    def test_empty_messages(self):
        """Test counting tokens in empty message list."""
        count = count_message_tokens([])
        # Should still have priming tokens
        assert count == 3  # Every reply is primed with assistant

    def test_message_with_name(self):
        """Test message with name field."""
        messages = [{"role": "user", "content": "Hello!", "name": "Alice"}]
        count = count_message_tokens(messages)
        assert count > 0


class TestEstimateCost:
    """Tests for estimate_cost function."""

    def test_zero_tokens(self):
        """Test cost with zero tokens."""
        cost = estimate_cost(0, 0)
        assert cost == 0.0

    def test_input_only_cost(self):
        """Test cost with only input tokens."""
        cost = estimate_cost(1_000_000, 0, "gpt-4o")
        assert cost == 2.50  # $2.50 per 1M input tokens

    def test_output_only_cost(self):
        """Test cost with only output tokens."""
        cost = estimate_cost(0, 1_000_000, "gpt-4o")
        assert cost == 10.00  # $10.00 per 1M output tokens

    def test_combined_cost(self):
        """Test combined input and output cost."""
        cost = estimate_cost(1_000_000, 1_000_000, "gpt-4o")
        assert cost == 12.50  # $2.50 + $10.00

    def test_small_token_count(self):
        """Test cost for small token counts."""
        cost = estimate_cost(1000, 500, "gpt-4o")
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(cost - expected) < 0.0001

    def test_unknown_model_uses_default(self):
        """Test unknown model falls back to default pricing."""
        cost = estimate_cost(1_000_000, 0, "unknown-model")
        # Should use default model (gpt-4o) pricing
        assert cost == 2.50


class TestGetBudgetStatus:
    """Tests for get_budget_status function."""

    def test_ok_status_for_low_cost(self):
        """Test OK status for cost below target."""
        status = get_budget_status(0.01, "query")
        assert status == BudgetStatus.OK

    def test_warning_status(self):
        """Test WARNING status for cost at warning threshold."""
        status = get_budget_status(0.025, "query")
        assert status == BudgetStatus.WARNING

    def test_alert_status(self):
        """Test ALERT status for cost at alert threshold."""
        status = get_budget_status(0.040, "query")
        assert status == BudgetStatus.ALERT

    def test_exceeded_status(self):
        """Test EXCEEDED status for cost at hard limit."""
        status = get_budget_status(0.100, "query")
        assert status == BudgetStatus.EXCEEDED

    def test_other_budget_type_ok(self):
        """Test OK status for other budget types."""
        status = get_budget_status(1.00, "benchmark")
        assert status == BudgetStatus.OK

    def test_other_budget_type_warning(self):
        """Test WARNING status for other budget types."""
        # 75% of $5.00 benchmark budget = $3.75
        status = get_budget_status(3.75, "benchmark")
        assert status == BudgetStatus.WARNING

    def test_other_budget_type_exceeded(self):
        """Test EXCEEDED status for other budget types."""
        status = get_budget_status(5.00, "benchmark")
        assert status == BudgetStatus.EXCEEDED


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""

    def test_create_record(self):
        """Test creating a usage record."""
        record = UsageRecord(
            input_tokens=500,
            output_tokens=200,
            model="gpt-4o",
            cost=0.0045,
            operation="test",
        )
        assert record.input_tokens == 500
        assert record.output_tokens == 200
        assert record.model == "gpt-4o"
        assert record.cost == 0.0045
        assert record.operation == "test"

    def test_default_operation(self):
        """Test default operation is empty string."""
        record = UsageRecord(
            input_tokens=500,
            output_tokens=200,
            model="gpt-4o",
            cost=0.0045,
        )
        assert record.operation == ""


class TestTokenCounter:
    """Tests for TokenCounter class."""

    def test_default_initialization(self):
        """Test default initialization."""
        counter = TokenCounter()
        assert counter.model == DEFAULT_MODEL
        assert counter.budget_limit == COST_THRESHOLDS["query_budget_hard_limit"]
        assert counter.budget_type == "query"

    def test_custom_initialization(self):
        """Test custom initialization."""
        counter = TokenCounter(
            model="gpt-4o-mini",
            budget_limit=1.00,
            budget_type="benchmark",
        )
        assert counter.model == "gpt-4o-mini"
        assert counter.budget_limit == 1.00
        assert counter.budget_type == "benchmark"

    def test_add_usage(self):
        """Test adding usage record."""
        counter = TokenCounter()
        record = counter.add_usage(500, 200)
        assert record.input_tokens == 500
        assert record.output_tokens == 200
        assert record.cost > 0
        assert counter.record_count == 1

    def test_add_usage_with_operation(self):
        """Test adding usage record with operation name."""
        counter = TokenCounter()
        record = counter.add_usage(500, 200, operation="embedding")
        assert record.operation == "embedding"

    def test_add_text_usage(self):
        """Test adding usage by text."""
        counter = TokenCounter()
        record = counter.add_text_usage("Hello, world!", "Hi there!")
        assert record.input_tokens > 0
        assert record.output_tokens > 0
        assert record.cost > 0

    def test_total_input_tokens(self):
        """Test total input tokens calculation."""
        counter = TokenCounter()
        counter.add_usage(500, 100)
        counter.add_usage(300, 50)
        assert counter.total_input_tokens == 800

    def test_total_output_tokens(self):
        """Test total output tokens calculation."""
        counter = TokenCounter()
        counter.add_usage(500, 100)
        counter.add_usage(300, 50)
        assert counter.total_output_tokens == 150

    def test_total_tokens(self):
        """Test total tokens calculation."""
        counter = TokenCounter()
        counter.add_usage(500, 100)
        counter.add_usage(300, 50)
        assert counter.total_tokens == 950

    def test_total_cost(self):
        """Test total cost calculation."""
        counter = TokenCounter()
        counter.add_usage(500, 100)
        counter.add_usage(300, 50)
        assert counter.total_cost > 0

    def test_record_count(self):
        """Test record count."""
        counter = TokenCounter()
        assert counter.record_count == 0
        counter.add_usage(500, 100)
        assert counter.record_count == 1
        counter.add_usage(300, 50)
        assert counter.record_count == 2

    def test_get_budget_status(self):
        """Test getting budget status."""
        counter = TokenCounter()
        status = counter.get_budget_status()
        assert status == BudgetStatus.OK

    def test_is_within_budget_true(self):
        """Test within budget check when true."""
        counter = TokenCounter()
        counter.add_usage(100, 50)
        assert counter.is_within_budget() is True

    def test_is_within_budget_false(self):
        """Test within budget check when false."""
        counter = TokenCounter(budget_limit=0.0001)
        counter.add_usage(1000, 500)
        assert counter.is_within_budget() is False

    def test_get_summary(self):
        """Test getting usage summary."""
        counter = TokenCounter()
        counter.add_usage(500, 200)
        summary = counter.get_summary()

        assert "total_input_tokens" in summary
        assert "total_output_tokens" in summary
        assert "total_tokens" in summary
        assert "total_cost" in summary
        assert "record_count" in summary
        assert "budget_limit" in summary
        assert "budget_remaining" in summary
        assert "budget_status" in summary
        assert "within_budget" in summary

    def test_reset(self):
        """Test resetting counter."""
        counter = TokenCounter()
        counter.add_usage(500, 200)
        counter.add_usage(300, 100)
        assert counter.record_count == 2

        counter.reset()
        assert counter.record_count == 0
        assert counter.total_tokens == 0
        assert counter.total_cost == 0.0


class TestGlobalTokenCounter:
    """Tests for global token counter singleton."""

    def test_get_token_counter_returns_instance(self):
        """Test getting global counter returns instance."""
        counter = get_token_counter()
        assert isinstance(counter, TokenCounter)

    def test_get_token_counter_returns_same_instance(self):
        """Test getting global counter returns same instance."""
        counter1 = get_token_counter()
        counter2 = get_token_counter()
        assert counter1 is counter2

    def test_reset_token_counter(self):
        """Test resetting global counter."""
        counter = get_token_counter()
        counter.add_usage(500, 200)
        assert counter.record_count > 0

        reset_token_counter()
        assert counter.record_count == 0
