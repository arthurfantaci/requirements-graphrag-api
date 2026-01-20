"""Tests for human feedback loop scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add scripts to path
SCRIPTS_PATH = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_PATH))

from export_for_annotation import (  # noqa: E402
    AnnotationCandidate,
    ExportConfig,
    calculate_confidence_score,
    determine_annotation_reason,
    extract_metrics_from_run,
)
from import_feedback import (  # noqa: E402
    AnnotationFeedback,
    generate_evaluation_example,
    process_annotations,
    validate_feedback,
)
from update_datasets import (  # noqa: E402
    DatasetExample,
    calculate_similarity,
    enrich_example,
    extract_entities,
    extract_standards,
    find_duplicates,
    infer_category,
    infer_difficulty,
    merge_datasets,
)

# =============================================================================
# Tests for export_for_annotation.py
# =============================================================================


class TestAnnotationCandidate:
    """Tests for AnnotationCandidate dataclass."""

    def test_create_candidate(self):
        """Test creating an annotation candidate."""
        candidate = AnnotationCandidate(
            run_id="run_123",
            question="What is traceability?",
            answer="Traceability is...",
            contexts=["context1", "context2"],
            ground_truth="Expected answer",
            metrics={"faithfulness": 0.5},
            confidence_score=0.5,
            timestamp="2024-01-01T00:00:00Z",
            reason="Low confidence",
        )
        assert candidate.run_id == "run_123"
        assert candidate.confidence_score == 0.5

    def test_to_dict(self):
        """Test converting to dictionary."""
        candidate = AnnotationCandidate(
            run_id="run_123",
            question="What is traceability?",
            answer="Traceability is...",
            contexts=[],
            ground_truth=None,
            metrics={},
            confidence_score=0.5,
            timestamp="",
            reason="Low confidence",
        )
        result = candidate.to_dict()
        assert result["run_id"] == "run_123"
        assert result["confidence_score"] == 0.5


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExportConfig()
        assert config.confidence_threshold == 0.7
        assert config.faithfulness_threshold == 0.6
        assert config.max_runs == 100
        assert config.days_back == 7

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExportConfig(
            confidence_threshold=0.8,
            max_runs=50,
        )
        assert config.confidence_threshold == 0.8
        assert config.max_runs == 50


class TestCalculateConfidenceScore:
    """Tests for calculate_confidence_score function."""

    def test_empty_metrics(self):
        """Test with empty metrics."""
        score = calculate_confidence_score({})
        assert score == 0.0

    def test_full_metrics(self):
        """Test with all metrics present."""
        metrics = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.9,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
        score = calculate_confidence_score(metrics)
        # Weighted average: 0.8*0.3 + 0.9*0.3 + 0.7*0.2 + 0.6*0.2 = 0.77
        assert abs(score - 0.77) < 0.01

    def test_partial_metrics(self):
        """Test with partial metrics."""
        metrics = {"faithfulness": 0.8, "answer_relevancy": 0.6}
        score = calculate_confidence_score(metrics)
        # Only faithfulness and relevancy: (0.8*0.3 + 0.6*0.3) / 0.6 = 0.7
        assert abs(score - 0.7) < 0.01


class TestDetermineAnnotationReason:
    """Tests for determine_annotation_reason function."""

    def test_no_reason_when_above_threshold(self):
        """Test no reason when all metrics are good."""
        config = ExportConfig(confidence_threshold=0.7)
        reason = determine_annotation_reason(
            metrics={"faithfulness": 0.9, "answer_relevancy": 0.9},
            confidence=0.9,
            config=config,
        )
        assert reason is None

    def test_low_confidence_reason(self):
        """Test reason for low confidence."""
        config = ExportConfig(confidence_threshold=0.7)
        reason = determine_annotation_reason(
            metrics={},
            confidence=0.5,
            config=config,
        )
        assert reason is not None
        assert "Low confidence" in reason

    def test_low_faithfulness_reason(self):
        """Test reason for low faithfulness."""
        config = ExportConfig(faithfulness_threshold=0.6)
        reason = determine_annotation_reason(
            metrics={"faithfulness": 0.4},
            confidence=0.8,
            config=config,
        )
        assert reason is not None
        assert "Low faithfulness" in reason

    def test_high_latency_reason(self):
        """Test reason for high latency."""
        config = ExportConfig(include_high_latency=True, latency_threshold_ms=1000)
        reason = determine_annotation_reason(
            metrics={},
            confidence=0.8,
            config=config,
            latency_ms=5000,
        )
        assert reason is not None
        assert "High latency" in reason


class TestExtractMetricsFromRun:
    """Tests for extract_metrics_from_run function."""

    def test_metrics_from_outputs(self):
        """Test extracting metrics from run outputs."""
        run = MagicMock()
        run.outputs = {"metrics": {"faithfulness": 0.8}}
        run.feedback_stats = None

        metrics = extract_metrics_from_run(run)
        assert metrics.get("faithfulness") == 0.8

    def test_metrics_from_feedback(self):
        """Test extracting metrics from feedback stats."""
        run = MagicMock()
        run.outputs = {}
        run.feedback_stats = {"quality": {"avg": 0.7}}

        metrics = extract_metrics_from_run(run)
        assert metrics.get("quality") == 0.7


# =============================================================================
# Tests for import_feedback.py
# =============================================================================


class TestAnnotationFeedback:
    """Tests for AnnotationFeedback dataclass."""

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "run_id": "run_123",
            "question": "What is X?",
            "original_answer": "X is...",
            "is_correct": True,
            "quality_score": 4,
        }
        feedback = AnnotationFeedback.from_dict(data)
        assert feedback.run_id == "run_123"
        assert feedback.is_correct is True
        assert feedback.quality_score == 4

    def test_to_dict(self):
        """Test converting to dictionary."""
        feedback = AnnotationFeedback(
            run_id="run_123",
            question="What is X?",
            original_answer="X is...",
            is_correct=True,
        )
        result = feedback.to_dict()
        assert result["run_id"] == "run_123"
        assert result["is_correct"] is True


class TestValidateFeedback:
    """Tests for validate_feedback function."""

    def test_valid_correct_feedback(self):
        """Test validation of correct feedback."""
        feedback = AnnotationFeedback(
            run_id="run_123",
            question="What is X?",
            original_answer="X is...",
            is_correct=True,
        )
        errors = validate_feedback(feedback)
        assert len(errors) == 0

    def test_valid_incorrect_with_correction(self):
        """Test validation of incorrect feedback with correction."""
        feedback = AnnotationFeedback(
            run_id="run_123",
            question="What is X?",
            original_answer="X is...",
            is_correct=False,
            corrected_answer="X is actually...",
        )
        errors = validate_feedback(feedback)
        assert len(errors) == 0

    def test_invalid_missing_question(self):
        """Test validation with missing question."""
        feedback = AnnotationFeedback(
            run_id="run_123",
            question="",
            original_answer="X is...",
            is_correct=True,
        )
        errors = validate_feedback(feedback)
        assert "Missing question" in errors

    def test_invalid_incorrect_without_correction(self):
        """Test validation of incorrect without correction."""
        feedback = AnnotationFeedback(
            run_id="run_123",
            question="What is X?",
            original_answer="X is...",
            is_correct=False,
            corrected_answer=None,
        )
        errors = validate_feedback(feedback)
        assert any("no correction" in e for e in errors)

    def test_invalid_quality_score(self):
        """Test validation of invalid quality score."""
        feedback = AnnotationFeedback(
            run_id="run_123",
            question="What is X?",
            original_answer="X is...",
            is_correct=True,
            quality_score=10,
        )
        errors = validate_feedback(feedback)
        assert any("out of range" in e for e in errors)


class TestGenerateEvaluationExample:
    """Tests for generate_evaluation_example function."""

    def test_generate_from_correct(self):
        """Test generating example from correct feedback."""
        feedback = AnnotationFeedback(
            run_id="run_123",
            question="What is X?",
            original_answer="X is a thing.",
            is_correct=True,
            quality_score=5,
        )
        example = generate_evaluation_example(feedback, "ex_001")

        assert example is not None
        assert example.question == "What is X?"
        assert example.ground_truth == "X is a thing."
        assert "verified-correct" in example.tags

    def test_generate_from_correction(self):
        """Test generating example from corrected feedback."""
        feedback = AnnotationFeedback(
            run_id="run_123",
            question="What is X?",
            original_answer="X is wrong.",
            is_correct=False,
            corrected_answer="X is actually correct.",
        )
        example = generate_evaluation_example(feedback, "ex_001")

        assert example is not None
        assert example.ground_truth == "X is actually correct."
        assert "human-corrected" in example.tags

    def test_no_example_without_ground_truth(self):
        """Test no example generated without ground truth."""
        feedback = AnnotationFeedback(
            run_id="run_123",
            question="What is X?",
            original_answer="X is wrong.",
            is_correct=False,
            corrected_answer=None,
        )
        example = generate_evaluation_example(feedback, "ex_001")
        assert example is None


class TestProcessAnnotations:
    """Tests for process_annotations function."""

    def test_process_valid_annotations(self):
        """Test processing valid annotations."""
        annotations = [
            AnnotationFeedback(
                run_id="run_1",
                question="Q1",
                original_answer="A1",
                is_correct=True,
                quality_score=4,
            ),
            AnnotationFeedback(
                run_id="run_2",
                question="Q2",
                original_answer="A2",
                is_correct=False,
                corrected_answer="Corrected A2",
                quality_score=3,
            ),
        ]

        _examples, stats = process_annotations(annotations)

        assert stats.total_annotations == 2
        assert stats.correct_answers == 1
        assert stats.incorrect_answers == 1
        assert stats.examples_generated == 2
        assert abs(stats.average_quality_score - 3.5) < 0.01

    def test_process_with_validation_errors(self):
        """Test processing with validation errors."""
        annotations = [
            AnnotationFeedback(
                run_id="run_1",
                question="",  # Invalid
                original_answer="A1",
                is_correct=True,
            ),
        ]

        _examples, stats = process_annotations(annotations)

        assert stats.validation_errors == 1
        assert stats.examples_generated == 0


# =============================================================================
# Tests for update_datasets.py
# =============================================================================


class TestDatasetExample:
    """Tests for DatasetExample dataclass."""

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "id": "ex_001",
            "question": "What is X?",
            "ground_truth": "X is...",
            "category": "definitional",
        }
        example = DatasetExample.from_dict(data)
        assert example.id == "ex_001"
        assert example.category == "definitional"

    def test_to_dict(self):
        """Test converting to dictionary."""
        example = DatasetExample(
            id="ex_001",
            question="What is X?",
            ground_truth="X is...",
        )
        result = example.to_dict()
        assert result["id"] == "ex_001"


class TestCalculateSimilarity:
    """Tests for calculate_similarity function."""

    def test_identical_strings(self):
        """Test similarity of identical strings."""
        sim = calculate_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_different_strings(self):
        """Test similarity of different strings."""
        sim = calculate_similarity("hello", "goodbye")
        assert sim < 0.5

    def test_similar_strings(self):
        """Test similarity of similar strings."""
        sim = calculate_similarity(
            "What is requirements traceability?",
            "What is requirement traceability?",
        )
        assert sim > 0.9

    def test_case_insensitive(self):
        """Test case insensitivity."""
        sim = calculate_similarity("HELLO", "hello")
        assert sim == 1.0


class TestFindDuplicates:
    """Tests for find_duplicates function."""

    def test_find_exact_duplicate(self):
        """Test finding exact duplicate."""
        new = [DatasetExample(id="new_1", question="What is X?", ground_truth="...")]
        existing = [DatasetExample(id="ex_1", question="What is X?", ground_truth="...")]

        duplicates = find_duplicates(new, existing)
        assert "new_1" in duplicates

    def test_find_similar_duplicate(self):
        """Test finding similar duplicate."""
        new = [
            DatasetExample(
                id="new_1", question="What is requirements traceability?", ground_truth="..."
            )
        ]
        existing = [
            DatasetExample(
                id="ex_1", question="What is requirement traceability?", ground_truth="..."
            )
        ]

        duplicates = find_duplicates(new, existing, similarity_threshold=0.85)
        assert "new_1" in duplicates

    def test_no_duplicates(self):
        """Test when no duplicates exist."""
        new = [
            DatasetExample(
                id="new_1",
                question="What is requirements traceability?",
                ground_truth="...",
            )
        ]
        existing = [
            DatasetExample(
                id="ex_1",
                question="How do I implement ISO 26262 compliance?",
                ground_truth="...",
            )
        ]

        duplicates = find_duplicates(new, existing)
        assert len(duplicates) == 0


class TestInferCategory:
    """Tests for infer_category function."""

    def test_definitional_what_is(self):
        """Test inferring definitional from 'what is'."""
        cat = infer_category("What is requirements traceability?")
        assert cat == "definitional"

    def test_procedural_how(self):
        """Test inferring procedural from 'how'."""
        cat = infer_category("How do I implement traceability?")
        assert cat == "procedural"

    def test_comparison(self):
        """Test inferring comparison."""
        cat = infer_category("Compare verification and validation")
        assert cat == "comparison"

    def test_analytical_why(self):
        """Test inferring analytical from 'why'."""
        cat = infer_category("Why is traceability important?")
        assert cat == "analytical"


class TestInferDifficulty:
    """Tests for infer_difficulty function."""

    def test_easy_short(self):
        """Test inferring easy for short Q&A."""
        diff = infer_difficulty("What is X?", "X is a simple thing.")
        assert diff == "easy"

    def test_hard_complex(self):
        """Test inferring hard for complex Q&A."""
        long_answer = " ".join(["word"] * 150)
        diff = infer_difficulty("Complex question and another part", long_answer)
        assert diff == "hard"


class TestExtractStandards:
    """Tests for extract_standards function."""

    def test_extract_iso(self):
        """Test extracting ISO standards."""
        standards = extract_standards("ISO 26262 and ISO 14971 are important")
        assert "ISO 26262" in standards
        assert "ISO 14971" in standards

    def test_extract_iec(self):
        """Test extracting IEC standards."""
        standards = extract_standards("IEC 62304 covers medical devices")
        assert "IEC 62304" in standards

    def test_extract_do178(self):
        """Test extracting DO-178C."""
        standards = extract_standards("Aviation uses DO-178C")
        assert "DO-178C" in standards


class TestExtractEntities:
    """Tests for extract_entities function."""

    def test_extract_traceability(self):
        """Test extracting traceability entity."""
        entities = extract_entities("Requirements traceability is important")
        assert "traceability" in entities

    def test_extract_multiple(self):
        """Test extracting multiple entities."""
        entities = extract_entities("Verification and validation in the V-model")
        assert "verification" in entities
        assert "validation" in entities
        assert "V-model" in entities


class TestEnrichExample:
    """Tests for enrich_example function."""

    def test_adds_tags(self):
        """Test that enrichment adds tags."""
        example = DatasetExample(
            id="ex_1",
            question="What is X?",
            ground_truth="X is...",
        )
        enriched = enrich_example(example)
        assert "human-feedback" in enriched.tags
        assert "auto-enriched" in enriched.tags

    def test_extracts_standards(self):
        """Test that enrichment extracts standards."""
        example = DatasetExample(
            id="ex_1",
            question="What is ISO 26262?",
            ground_truth="ISO 26262 is a safety standard.",
        )
        enriched = enrich_example(example)
        assert "ISO 26262" in enriched.expected_standards


class TestMergeDatasets:
    """Tests for merge_datasets function."""

    def test_merge_no_duplicates(self):
        """Test merging without duplicates."""
        existing = [DatasetExample(id="ex_1", question="Q1", ground_truth="A1")]
        new = [
            DatasetExample(
                id="new_1",
                question="Q2",
                ground_truth="A2",
                metadata={"confidence": 0.9},
            )
        ]

        _merged, stats = merge_datasets(existing, new)

        assert stats.existing_examples == 1
        assert stats.examples_added == 1
        assert stats.final_total == 2

    def test_merge_with_duplicates(self):
        """Test merging with duplicates."""
        existing = [DatasetExample(id="ex_1", question="What is X?", ground_truth="A1")]
        new = [
            DatasetExample(
                id="new_1",
                question="What is X?",  # Duplicate
                ground_truth="A2",
                metadata={"confidence": 0.9},
            )
        ]

        _merged, stats = merge_datasets(existing, new)

        assert stats.duplicates_found == 1
        assert stats.examples_added == 0
        assert stats.final_total == 1

    def test_merge_low_confidence_skipped(self):
        """Test that low confidence examples are skipped."""
        existing = []
        new = [
            DatasetExample(
                id="new_1",
                question="Q1",
                ground_truth="A1",
                metadata={"confidence": 0.5},
            )
        ]

        _merged, stats = merge_datasets(existing, new, min_confidence=0.8)

        assert stats.low_confidence_skipped == 1
        assert stats.examples_added == 0
