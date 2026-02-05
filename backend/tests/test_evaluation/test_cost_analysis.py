"""Tests for cost analysis utilities.

Tests cover:
- Cost estimation for different models
- CostTracker recording and reporting
- Cost comparison functionality
- Recommendation generation
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from requirements_graphrag_api.evaluation.cost_analysis import (
    MODEL_PRICING,
    CostTracker,
    LLMCall,
    estimate_cost,
    get_cost_report,
    get_global_cost_tracker,
    reset_global_cost_tracker,
)

# =============================================================================
# COST ESTIMATION TESTS
# =============================================================================


class TestEstimateCost:
    """Tests for estimate_cost function."""

    def test_gpt4o_mini_cost(self):
        """Test cost estimation for gpt-4o-mini."""
        # 1M tokens at $0.15 input + $0.60 output
        cost = estimate_cost("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(0.75, rel=0.01)  # $0.15 + $0.60

    def test_gpt4o_cost(self):
        """Test cost estimation for gpt-4o."""
        # 1M tokens at $2.50 input + $10.00 output
        cost = estimate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(12.50, rel=0.01)

    def test_small_token_count(self):
        """Test cost for typical query size."""
        # 1000 input + 500 output tokens
        cost = estimate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert cost == pytest.approx(expected, rel=0.01)

    def test_embedding_model_cost(self):
        """Test cost for embedding model (no output tokens)."""
        cost = estimate_cost("text-embedding-3-small", input_tokens=1_000_000)
        assert cost == pytest.approx(0.02, rel=0.01)

    def test_unknown_model_fallback(self):
        """Test fallback for unknown model."""
        # Should use gpt-4o-mini pricing as default
        cost = estimate_cost("unknown-model-xyz", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(0.75, rel=0.01)

    def test_case_insensitive_matching(self):
        """Test model name matching is case insensitive."""
        cost1 = estimate_cost("GPT-4O-MINI", input_tokens=1000)
        cost2 = estimate_cost("gpt-4o-mini", input_tokens=1000)
        assert cost1 == cost2


# =============================================================================
# LLM CALL TESTS
# =============================================================================


class TestLLMCall:
    """Tests for LLMCall dataclass."""

    def test_total_tokens(self):
        """Test total token calculation."""
        call = LLMCall(model="gpt-4o-mini", input_tokens=500, output_tokens=200)
        assert call.total_tokens == 700

    def test_cost_property(self):
        """Test cost property calculation."""
        call = LLMCall(model="gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
        assert call.cost == pytest.approx(0.75, rel=0.01)

    def test_timestamp_auto_set(self):
        """Test timestamp is automatically set."""
        call = LLMCall(model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        assert call.timestamp is not None


# =============================================================================
# COST TRACKER TESTS
# =============================================================================


class TestCostTracker:
    """Tests for CostTracker."""

    def test_record_llm_call(self):
        """Test recording an LLM call."""
        tracker = CostTracker()
        call = tracker.record_llm_call(
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
            operation="test_op",
        )

        assert call.model == "gpt-4o-mini"
        assert call.input_tokens == 1000
        assert call.output_tokens == 500
        assert call.operation == "test_op"

    def test_record_multiple_calls(self):
        """Test recording multiple calls."""
        tracker = CostTracker()

        tracker.record_llm_call("gpt-4o-mini", 1000, 500, "op1")
        tracker.record_llm_call("gpt-4o-mini", 2000, 1000, "op2")

        calls = tracker.get_calls()
        assert len(calls) == 2

    def test_record_from_response_openai_format(self):
        """Test recording from OpenAI response format."""
        tracker = CostTracker()

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 500
        mock_response.usage.completion_tokens = 200

        call = tracker.record_from_response("gpt-4o-mini", mock_response, "test")

        assert call is not None
        assert call.input_tokens == 500
        assert call.output_tokens == 200

    def test_record_from_response_dict_format(self):
        """Test recording from dict response format."""
        tracker = CostTracker()

        response = {"usage": {"prompt_tokens": 300, "completion_tokens": 150}}

        call = tracker.record_from_response("gpt-4o-mini", response, "test")

        assert call is not None
        assert call.input_tokens == 300
        assert call.output_tokens == 150

    def test_record_from_response_no_usage(self):
        """Test recording returns None when no usage info."""
        tracker = CostTracker()
        call = tracker.record_from_response("gpt-4o-mini", {"data": "no usage"}, "test")
        assert call is None


# =============================================================================
# COST REPORT TESTS
# =============================================================================


class TestCostReport:
    """Tests for CostReport generation."""

    def test_empty_report(self):
        """Test report with no calls."""
        tracker = CostTracker()
        report = tracker.get_report()

        assert report.total_calls == 0
        assert report.total_cost == 0.0
        assert report.total_tokens == 0

    def test_report_totals(self):
        """Test report totals calculation."""
        tracker = CostTracker()
        tracker.record_llm_call("gpt-4o-mini", 1000, 500, "op1")
        tracker.record_llm_call("gpt-4o-mini", 2000, 1000, "op2")

        report = tracker.get_report()

        assert report.total_calls == 2
        assert report.total_input_tokens == 3000
        assert report.total_output_tokens == 1500
        assert report.total_tokens == 4500

    def test_report_by_model(self):
        """Test report breakdown by model."""
        tracker = CostTracker()
        # Use larger token counts to make cost differences more apparent
        tracker.record_llm_call("gpt-4o-mini", 100_000, 50_000)
        tracker.record_llm_call("gpt-4o", 100_000, 50_000)

        report = tracker.get_report()

        assert "gpt-4o-mini" in report.cost_by_model
        assert "gpt-4o" in report.cost_by_model
        # gpt-4o should be more expensive than gpt-4o-mini
        assert report.cost_by_model["gpt-4o"] > report.cost_by_model["gpt-4o-mini"]

    def test_report_by_operation(self):
        """Test report breakdown by operation."""
        tracker = CostTracker()
        tracker.record_llm_call("gpt-4o-mini", 1000, 500, "rag_generation")
        tracker.record_llm_call("gpt-4o-mini", 2000, 1000, "rag_generation")
        tracker.record_llm_call("gpt-4o-mini", 500, 200, "query_expansion")

        report = tracker.get_report()

        assert "rag_generation" in report.cost_by_operation
        assert "query_expansion" in report.cost_by_operation
        assert report.calls_by_operation["rag_generation"] == 2
        assert report.calls_by_operation["query_expansion"] == 1

    def test_report_averages(self):
        """Test report average calculations."""
        tracker = CostTracker()
        tracker.record_llm_call("gpt-4o-mini", 1000, 500)
        tracker.record_llm_call("gpt-4o-mini", 2000, 1000)

        report = tracker.get_report()

        assert report.avg_tokens_per_call == 2250  # (1500 + 3000) / 2
        assert report.avg_cost_per_call > 0


# =============================================================================
# RECOMMENDATION TESTS
# =============================================================================


class TestRecommendations:
    """Tests for cost recommendation generation."""

    def test_expensive_model_recommendation(self):
        """Test recommendation for expensive model usage."""
        tracker = CostTracker()
        # Heavy usage of expensive model
        tracker.record_llm_call("gpt-4o", 10000, 5000, "heavy_op")

        report = tracker.get_report()

        assert any("gpt-4o-mini" in r.lower() for r in report.recommendations)

    def test_high_cost_operation_recommendation(self):
        """Test recommendation for high-cost operations."""
        tracker = CostTracker()
        # One operation dominates costs
        tracker.record_llm_call("gpt-4o", 50000, 25000, "expensive_op")
        tracker.record_llm_call("gpt-4o-mini", 1000, 500, "cheap_op")

        report = tracker.get_report()

        assert any("expensive_op" in r for r in report.recommendations)


# =============================================================================
# COMPARISON TESTS
# =============================================================================


class TestCostComparison:
    """Tests for cost comparison functionality."""

    def test_compare_with_baseline(self):
        """Test comparing current costs with baseline."""
        baseline = CostTracker()
        baseline.record_llm_call("gpt-4o-mini", 1000, 500, "old_rag")

        current = CostTracker()
        current.record_llm_call("gpt-4o-mini", 2000, 1000, "new_agentic")

        comparison = current.compare_with_baseline(baseline)

        assert comparison["current_cost"] > comparison["baseline_cost"]
        assert comparison["verdict"] == "more_expensive"

    def test_compare_more_efficient(self):
        """Test comparison when current is more efficient."""
        baseline = CostTracker()
        # gpt-4o: $2.50/1M input + $10/1M output = much more expensive
        baseline.record_llm_call("gpt-4o", 100_000, 50_000, "old")

        current = CostTracker()
        # gpt-4o-mini: $0.15/1M input + $0.60/1M output = much cheaper
        current.record_llm_call("gpt-4o-mini", 100_000, 50_000, "new")

        comparison = current.compare_with_baseline(baseline)

        assert comparison["current_cost"] < comparison["baseline_cost"]
        assert comparison["verdict"] == "less_expensive"

    def test_compare_same_cost(self):
        """Test comparison when costs are equal."""
        baseline = CostTracker()
        baseline.record_llm_call("gpt-4o-mini", 1000, 500)

        current = CostTracker()
        current.record_llm_call("gpt-4o-mini", 1000, 500)

        comparison = current.compare_with_baseline(baseline)

        assert comparison["cost_difference"] == 0
        assert comparison["verdict"] == "same"


# =============================================================================
# GLOBAL TRACKER TESTS
# =============================================================================


class TestGlobalCostTracker:
    """Tests for global cost tracker functions."""

    def test_global_tracker_singleton(self):
        """Test global tracker is a singleton."""
        reset_global_cost_tracker()
        tracker1 = get_global_cost_tracker()
        tracker2 = get_global_cost_tracker()
        assert tracker1 is tracker2

    def test_get_cost_report_function(self):
        """Test get_cost_report convenience function."""
        reset_global_cost_tracker()
        tracker = get_global_cost_tracker()
        tracker.record_llm_call("gpt-4o-mini", 1000, 500, "test")

        report = get_cost_report()
        assert report.total_calls == 1

    def test_reset_global_cost_tracker(self):
        """Test resetting global tracker."""
        reset_global_cost_tracker()
        tracker = get_global_cost_tracker()
        tracker.record_llm_call("gpt-4o-mini", 1000, 500)

        reset_global_cost_tracker()
        report = get_cost_report()
        assert report.total_calls == 0


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestCostAnalysisIntegration:
    """Integration-style tests for cost analysis."""

    def test_exports_from_package(self):
        """Test that cost analysis utilities are exported from package."""
        from requirements_graphrag_api.evaluation import (
            CostReport,
            CostTracker,
            LLMCall,
            estimate_cost,
            get_cost_report,
            get_global_cost_tracker,
            reset_global_cost_tracker,
        )

        assert callable(estimate_cost)
        assert callable(get_cost_report)
        assert callable(get_global_cost_tracker)
        assert callable(reset_global_cost_tracker)
        assert CostReport is not None
        assert CostTracker is not None
        assert LLMCall is not None

    def test_full_tracking_workflow(self):
        """Test a complete cost tracking workflow."""
        tracker = CostTracker()

        # Simulate agentic RAG workflow
        tracker.record_llm_call("gpt-4o-mini", 500, 200, "query_expansion")
        tracker.record_llm_call("gpt-4o-mini", 1000, 500, "rag_generation")
        tracker.record_llm_call("gpt-4o-mini", 800, 300, "synthesis")
        tracker.record_llm_call("gpt-4o-mini", 400, 150, "critique")

        report = tracker.get_report()

        assert report.total_calls == 4
        assert "query_expansion" in report.cost_by_operation
        assert "rag_generation" in report.cost_by_operation
        assert "synthesis" in report.cost_by_operation
        assert "critique" in report.cost_by_operation
        assert report.total_cost > 0

    def test_model_pricing_coverage(self):
        """Test that common models have pricing."""
        expected_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "text-embedding-3-small",
            "claude-3-5-sonnet",
            "claude-3-opus",
            "claude-3-haiku",
        ]

        for model in expected_models:
            assert model in MODEL_PRICING, f"Missing pricing for {model}"
            assert "input" in MODEL_PRICING[model]
            assert "output" in MODEL_PRICING[model]
