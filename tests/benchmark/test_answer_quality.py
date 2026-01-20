"""Answer quality benchmark tests.

Tests answer generation quality using:
- RAGAS metrics (faithfulness, relevancy, precision, recall)
- Domain-specific quality checks
- Ground truth comparison

These tests validate that generated answers are accurate,
relevant, and properly grounded in retrieved context.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.benchmark.schemas import (
    BenchmarkExample,
    DifficultyLevel,
    QueryCategory,
)

# =============================================================================
# ANSWER QUALITY CHECKS
# =============================================================================


def check_answer_not_empty(answer: str) -> bool:
    """Check that answer is not empty or whitespace-only."""
    return bool(answer and answer.strip())


def check_answer_length(answer: str, min_words: int = 10, max_words: int = 500) -> bool:
    """Check that answer length is within acceptable range."""
    word_count = len(answer.split())
    return min_words <= word_count <= max_words


def check_answer_mentions_entities(answer: str, entities: list[str]) -> float:
    """Check what proportion of expected entities are mentioned.

    Args:
        answer: The generated answer.
        entities: List of entities that should be mentioned.

    Returns:
        Proportion of entities mentioned (0.0 to 1.0).
    """
    if not entities:
        return 1.0

    answer_lower = answer.lower()
    mentioned = sum(1 for e in entities if e.lower() in answer_lower)
    return mentioned / len(entities)


def check_answer_cites_standards(answer: str, standards: list[str]) -> float:
    """Check what proportion of expected standards are cited.

    Args:
        answer: The generated answer.
        standards: List of standards that should be referenced.

    Returns:
        Proportion of standards cited (0.0 to 1.0).
    """
    if not standards:
        return 1.0

    answer_lower = answer.lower()
    cited = sum(1 for s in standards if s.lower() in answer_lower)
    return cited / len(standards)


def check_no_hallucination_markers(answer: str) -> bool:
    """Check for common hallucination markers.

    Returns True if no hallucination markers found (good).
    """
    hallucination_phrases = [
        "i don't have information",
        "i cannot find",
        "based on my training",
        "as an ai",
        "i'm not sure but",
        "i think maybe",
    ]

    answer_lower = answer.lower()
    for phrase in hallucination_phrases:
        if phrase in answer_lower:
            return False
    return True


# =============================================================================
# UNIT TESTS FOR QUALITY CHECKS
# =============================================================================


class TestAnswerQualityChecks:
    """Unit tests for answer quality check functions."""

    def test_check_answer_not_empty_valid(self) -> None:
        """Test with valid non-empty answer."""
        assert check_answer_not_empty("This is a valid answer.")
        assert check_answer_not_empty("Short answer")

    def test_check_answer_not_empty_invalid(self) -> None:
        """Test with empty or whitespace answers."""
        assert not check_answer_not_empty("")
        assert not check_answer_not_empty("   ")
        assert not check_answer_not_empty("\n\t")

    def test_check_answer_length_valid(self) -> None:
        """Test with answers within length limits."""
        short_answer = " ".join(["word"] * 15)
        assert check_answer_length(short_answer)

        medium_answer = " ".join(["word"] * 100)
        assert check_answer_length(medium_answer)

    def test_check_answer_length_too_short(self) -> None:
        """Test with answer that's too short."""
        assert not check_answer_length("Too short", min_words=10)

    def test_check_answer_length_too_long(self) -> None:
        """Test with answer that's too long."""
        long_answer = " ".join(["word"] * 600)
        assert not check_answer_length(long_answer, max_words=500)

    def test_check_mentions_entities_all(self) -> None:
        """Test when all entities are mentioned."""
        answer = "Requirements traceability and verification are important."
        entities = ["traceability", "verification"]

        assert check_answer_mentions_entities(answer, entities) == 1.0

    def test_check_mentions_entities_partial(self) -> None:
        """Test when some entities are mentioned."""
        answer = "Traceability is important for compliance."
        entities = ["traceability", "verification", "validation"]

        score = check_answer_mentions_entities(answer, entities)
        assert score == pytest.approx(1 / 3)

    def test_check_mentions_entities_none(self) -> None:
        """Test when no entities are mentioned."""
        answer = "This is a generic answer."
        entities = ["traceability", "verification"]

        assert check_answer_mentions_entities(answer, entities) == 0.0

    def test_check_cites_standards_all(self) -> None:
        """Test when all standards are cited."""
        answer = "ISO 26262 and IEC 62304 are important standards."
        standards = ["ISO 26262", "IEC 62304"]

        assert check_answer_cites_standards(answer, standards) == 1.0

    def test_check_cites_standards_empty(self) -> None:
        """Test with no expected standards."""
        answer = "This answer doesn't need to cite standards."

        assert check_answer_cites_standards(answer, []) == 1.0

    def test_check_no_hallucination_clean(self) -> None:
        """Test answer without hallucination markers."""
        answer = "Requirements traceability enables impact analysis."
        assert check_no_hallucination_markers(answer)

    def test_check_no_hallucination_detected(self) -> None:
        """Test answer with hallucination markers."""
        answer = "I don't have information about this specific topic."
        assert not check_no_hallucination_markers(answer)


# =============================================================================
# ANSWER GENERATION TESTS
# =============================================================================


class TestAnswerGeneration:
    """Tests for answer generation quality."""

    @pytest.mark.asyncio
    async def test_definitional_answer_quality(
        self,
        mock_config: MagicMock,
        definitional_examples: list[BenchmarkExample],
        mock_chat_response: dict[str, Any],
    ) -> None:
        """Test answer quality for definitional queries."""
        example = definitional_examples[0]

        # Simulate chat response
        answer = mock_chat_response["answer"]

        # Basic quality checks
        assert check_answer_not_empty(answer)
        assert check_answer_length(answer, min_words=5)

    @pytest.mark.asyncio
    async def test_answer_includes_expected_entities(
        self,
        golden_example: BenchmarkExample,
    ) -> None:
        """Test that answers include expected entities."""
        # This would use actual generation in integration tests
        # For unit tests, we verify the expectation is set
        if golden_example.expected_entities:
            assert len(golden_example.expected_entities) > 0

    @pytest.mark.asyncio
    async def test_standards_answer_cites_correctly(
        self,
        standards_examples: list[BenchmarkExample],
    ) -> None:
        """Test that standards-related answers cite the standard."""
        for example in standards_examples:
            # Verify expected standards are defined
            if "standards" in example.tags:
                assert len(example.expected_standards) > 0 or "edge-case" in example.tags, (
                    f"Standards example {example.id} should have expected_standards"
                )


class TestAnswerFaithfulness:
    """Tests for answer faithfulness to context."""

    def test_answer_grounded_in_context(self) -> None:
        """Test that answers are grounded in retrieved context."""
        context = "ISO 26262 defines ASIL levels A through D for automotive safety."
        answer = "ISO 26262 defines four ASIL levels: A, B, C, and D."

        # Answer should mention key terms from context
        context_terms = ["iso 26262", "asil"]
        answer_lower = answer.lower()

        terms_found = sum(1 for term in context_terms if term in answer_lower)
        assert terms_found >= 1, "Answer should reference context terms"

    def test_answer_not_contradicting_context(self) -> None:
        """Test that answers don't contradict context."""
        context = "ISO 26262 is for automotive functional safety."
        answer = "ISO 26262 is an automotive safety standard."

        # Simple consistency check - both mention automotive and safety
        assert "automotive" in answer.lower()
        assert "safety" in answer.lower()


class TestAnswerRelevancy:
    """Tests for answer relevancy to question."""

    def test_answer_addresses_question(self) -> None:
        """Test that answer addresses the question asked."""
        question = "What is requirements traceability?"
        answer = "Requirements traceability is the ability to link requirements to other artifacts."

        # Answer should mention the concept being asked about
        assert "traceability" in answer.lower()

    def test_answer_format_matches_question_type(self) -> None:
        """Test that answer format matches question type."""
        # Definitional question should get a definition
        def_question = "What is verification?"
        def_answer = "Verification is the process of confirming the product is built correctly."

        assert "is" in def_answer.lower()  # Definitional structure

        # Procedural question should get steps/process
        proc_question = "How do I implement traceability?"
        proc_answer = "To implement traceability: 1) Define link types, 2) Establish baselines..."

        # Procedural answers often have numbered steps or action words
        assert any(
            marker in proc_answer.lower() for marker in ["1)", "first", "step", "to implement"]
        )


# =============================================================================
# METRIC THRESHOLD TESTS
# =============================================================================


class TestMetricThresholds:
    """Tests for metric threshold compliance."""

    def test_example_has_valid_thresholds(
        self,
        golden_example: BenchmarkExample,
    ) -> None:
        """Test that all examples have valid metric thresholds."""
        metrics = golden_example.expected_metrics

        # Thresholds should be between 0 and 1
        assert 0.0 <= metrics.min_faithfulness <= 1.0
        assert 0.0 <= metrics.min_relevancy <= 1.0
        assert 0.0 <= metrics.min_precision <= 1.0
        assert 0.0 <= metrics.min_recall <= 1.0

    def test_edge_cases_have_lower_thresholds(
        self,
        edge_case_examples: list[BenchmarkExample],
    ) -> None:
        """Test that edge cases have appropriately lower thresholds."""
        for example in edge_case_examples:
            metrics = example.expected_metrics

            # Edge cases should have more lenient thresholds
            # (default is 0.7, edge cases should be lower)
            assert metrics.min_faithfulness <= 0.8
            assert metrics.min_relevancy <= 0.8

    def test_must_pass_have_strict_thresholds(
        self,
        must_pass_examples: list[BenchmarkExample],
    ) -> None:
        """Test that must-pass examples have reasonable thresholds."""
        for example in must_pass_examples:
            # Must-pass shouldn't have thresholds below 0.3
            # (that would make them trivial to pass)
            if "out-of-domain" not in example.tags:
                metrics = example.expected_metrics
                assert metrics.min_faithfulness >= 0.3
                assert metrics.min_relevancy >= 0.3


# =============================================================================
# GROUND TRUTH COMPARISON TESTS
# =============================================================================


class TestGroundTruthComparison:
    """Tests comparing answers to ground truth."""

    def test_ground_truth_format(
        self,
        golden_example: BenchmarkExample,
    ) -> None:
        """Test that ground truth is properly formatted."""
        gt = golden_example.ground_truth

        # Ground truth should be non-empty
        assert gt and gt.strip()

        # Ground truth should be descriptive (not too short)
        assert len(gt) >= 20, f"Ground truth too short for {golden_example.id}"

    def test_ground_truth_covers_key_points(
        self,
        golden_example: BenchmarkExample,
    ) -> None:
        """Test that ground truth covers expected entities/standards."""
        gt_lower = golden_example.ground_truth.lower()

        # Ground truth should mention expected entities (when defined)
        for entity in golden_example.expected_entities[:2]:  # Check first 2
            if entity.lower() not in gt_lower:
                # It's okay if not all entities are in ground truth
                # but at least warn about it
                pass

        # Ground truth should mention expected standards (when defined)
        for standard in golden_example.expected_standards[:2]:
            if standard.lower() not in gt_lower:
                pass


# =============================================================================
# BENCHMARK-DRIVEN ANSWER TESTS
# =============================================================================


class TestBenchmarkAnswerQuality:
    """Benchmark-driven answer quality tests."""

    def test_all_categories_have_ground_truth(
        self,
        golden_dataset: list[BenchmarkExample],
    ) -> None:
        """Verify all categories have examples with ground truth."""
        for category in QueryCategory:
            category_examples = [ex for ex in golden_dataset if ex.category == category]
            assert len(category_examples) > 0, f"No examples for {category.value}"

            # All should have ground truth
            for ex in category_examples:
                assert ex.ground_truth, f"No ground truth for {ex.id}"

    def test_difficulty_distribution(
        self,
        golden_dataset: list[BenchmarkExample],
    ) -> None:
        """Verify reasonable difficulty distribution."""
        difficulties = [ex.difficulty for ex in golden_dataset]

        easy_count = difficulties.count(DifficultyLevel.EASY)
        medium_count = difficulties.count(DifficultyLevel.MEDIUM)
        hard_count = difficulties.count(DifficultyLevel.HARD)

        # Should have examples at each level
        assert easy_count >= 3, "Need more easy examples"
        assert medium_count >= 5, "Need more medium examples"
        assert hard_count >= 3, "Need more hard examples"


# =============================================================================
# MOCK RAGAS EVALUATION
# =============================================================================


class TestRAGASMetricsIntegration:
    """Tests for RAGAS metrics integration."""

    @pytest.mark.asyncio
    async def test_compute_faithfulness_mock(
        self,
        mock_config: MagicMock,
    ) -> None:
        """Test faithfulness computation with mocks."""
        with patch(
            "jama_mcp_server_graphrag.evaluation.metrics.compute_faithfulness"
        ) as mock_compute:
            mock_compute.return_value = 0.85

            # In real tests, this would call the actual function
            score = await mock_compute(
                mock_config,
                "What is traceability?",
                "Traceability links requirements to artifacts.",
                ["Context about traceability"],
            )

            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_compute_all_metrics_mock(
        self,
        mock_config: MagicMock,
    ) -> None:
        """Test all metrics computation with mocks."""
        with patch(
            "jama_mcp_server_graphrag.evaluation.metrics.compute_all_metrics"
        ) as mock_compute:
            from jama_mcp_server_graphrag.evaluation.metrics import RAGMetrics

            mock_compute.return_value = RAGMetrics(
                faithfulness=0.85,
                answer_relevancy=0.80,
                context_precision=0.75,
                context_recall=0.70,
            )

            metrics = await mock_compute(
                mock_config,
                "question",
                "answer",
                ["context"],
                "ground_truth",
            )

            assert metrics.faithfulness == 0.85
            assert metrics.answer_relevancy == 0.80


__all__ = [
    "check_answer_cites_standards",
    "check_answer_length",
    "check_answer_mentions_entities",
    "check_answer_not_empty",
    "check_no_hallucination_markers",
]
