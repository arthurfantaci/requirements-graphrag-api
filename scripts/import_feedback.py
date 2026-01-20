#!/usr/bin/env python3
"""Import human feedback annotations for evaluation improvement.

This script imports annotations from human reviewers and processes them
for use in improving the evaluation dataset and model performance.

Workflow:
1. Read annotations from LangSmith or local JSON
2. Validate and normalize feedback data
3. Generate corrected examples for the golden dataset
4. Output statistics on annotation quality

Usage:
    # Import from local JSON file
    python scripts/import_feedback.py --input annotations_completed.json

    # Import from LangSmith feedback
    python scripts/import_feedback.py --from-langsmith --project my-project

    # Import and generate new dataset examples
    python scripts/import_feedback.py --input feedback.json --generate-examples

    # Validate annotations without importing
    python scripts/import_feedback.py --input feedback.json --validate-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Constants for quality score validation
MIN_QUALITY_SCORE = 1
MAX_QUALITY_SCORE = 5
HIGH_QUALITY_THRESHOLD = 4
LOW_QUALITY_THRESHOLD = 2
LANGSMITH_SCORE_THRESHOLD = 0.5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AnnotationFeedback:
    """Human feedback on an evaluation run.

    Attributes:
        run_id: Original LangSmith run ID.
        question: The input question.
        original_answer: The model's original answer.
        is_correct: Whether the answer was correct.
        corrected_answer: Human-provided corrected answer if incorrect.
        quality_score: Human-assigned quality score (1-5).
        feedback_notes: Reviewer's notes.
        missing_information: What information was missing.
        factual_errors: Any factual errors identified.
        annotator_id: ID of the human annotator.
        annotation_timestamp: When the annotation was made.
        metadata: Additional annotation metadata.
    """

    run_id: str
    question: str
    original_answer: str
    is_correct: bool
    corrected_answer: str | None = None
    quality_score: int | None = None
    feedback_notes: str = ""
    missing_information: list[str] = field(default_factory=list)
    factual_errors: list[str] = field(default_factory=list)
    annotator_id: str = ""
    annotation_timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationFeedback:
        """Create from dictionary."""
        return cls(
            run_id=data.get("run_id", ""),
            question=data.get("question", ""),
            original_answer=data.get("original_answer", data.get("answer", "")),
            is_correct=data.get("is_correct", False),
            corrected_answer=data.get("corrected_answer"),
            quality_score=data.get("quality_score"),
            feedback_notes=data.get("feedback_notes", data.get("notes", "")),
            missing_information=data.get("missing_information", []),
            factual_errors=data.get("factual_errors", []),
            annotator_id=data.get("annotator_id", ""),
            annotation_timestamp=data.get("annotation_timestamp", ""),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "question": self.question,
            "original_answer": self.original_answer,
            "is_correct": self.is_correct,
            "corrected_answer": self.corrected_answer,
            "quality_score": self.quality_score,
            "feedback_notes": self.feedback_notes,
            "missing_information": self.missing_information,
            "factual_errors": self.factual_errors,
            "annotator_id": self.annotator_id,
            "annotation_timestamp": self.annotation_timestamp,
            "metadata": self.metadata,
        }


@dataclass
class NewEvaluationExample:
    """A new evaluation example generated from feedback.

    Attributes:
        id: Unique identifier for the example.
        question: The question.
        ground_truth: Corrected/verified answer.
        source_run_id: Original run ID this was derived from.
        confidence: Confidence in this example's correctness.
        tags: Tags for categorization.
        metadata: Additional metadata.
    """

    id: str
    question: str
    ground_truth: str
    source_run_id: str
    confidence: float
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format compatible with golden dataset."""
        return {
            "id": self.id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "source_run_id": self.source_run_id,
            "confidence": self.confidence,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class ImportStats:
    """Statistics from feedback import.

    Attributes:
        total_annotations: Total annotations processed.
        correct_answers: Number marked as correct.
        incorrect_answers: Number marked as incorrect.
        with_corrections: Number with corrected answers.
        average_quality_score: Average quality score.
        examples_generated: Number of new examples generated.
        validation_errors: Number of validation errors.
    """

    total_annotations: int = 0
    correct_answers: int = 0
    incorrect_answers: int = 0
    with_corrections: int = 0
    average_quality_score: float = 0.0
    examples_generated: int = 0
    validation_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_annotations": self.total_annotations,
            "correct_answers": self.correct_answers,
            "incorrect_answers": self.incorrect_answers,
            "with_corrections": self.with_corrections,
            "average_quality_score": round(self.average_quality_score, 2),
            "examples_generated": self.examples_generated,
            "validation_errors": self.validation_errors,
            "accuracy_rate": (
                round(self.correct_answers / self.total_annotations, 2)
                if self.total_annotations > 0
                else 0.0
            ),
        }


def validate_feedback(feedback: AnnotationFeedback) -> list[str]:
    """Validate a feedback annotation.

    Args:
        feedback: The feedback to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    if not feedback.question:
        errors.append("Missing question")

    if not feedback.original_answer:
        errors.append("Missing original answer")

    if not feedback.is_correct and not feedback.corrected_answer:
        errors.append("Incorrect answer marked but no correction provided")

    score_is_set = feedback.quality_score is not None
    score_in_range = MIN_QUALITY_SCORE <= (feedback.quality_score or 0) <= MAX_QUALITY_SCORE
    if score_is_set and not score_in_range:
        errors.append(
            f"Quality score {feedback.quality_score} out of range "
            f"({MIN_QUALITY_SCORE}-{MAX_QUALITY_SCORE})"
        )

    return errors


def load_annotations_from_json(input_path: Path) -> list[AnnotationFeedback]:
    """Load annotations from a JSON file.

    Args:
        input_path: Path to JSON file.

    Returns:
        List of AnnotationFeedback instances.
    """
    logger.info("Loading annotations from %s", input_path)

    with input_path.open() as f:
        data = json.load(f)

    # Handle both flat list and nested structure
    if isinstance(data, list):
        annotations_data = data
    elif isinstance(data, dict):
        annotations_data = data.get("annotations", data.get("feedback", data.get("candidates", [])))
    else:
        annotations_data = []

    annotations = []
    for item in annotations_data:
        try:
            feedback = AnnotationFeedback.from_dict(item)
            annotations.append(feedback)
        except Exception as e:
            logger.warning("Failed to parse annotation: %s", e)

    logger.info("Loaded %d annotations", len(annotations))
    return annotations


def load_annotations_from_langsmith(
    project: str,
    days_back: int = 7,
) -> list[AnnotationFeedback]:
    """Load annotations from LangSmith feedback.

    Args:
        project: LangSmith project name.
        days_back: Number of days to look back.

    Returns:
        List of AnnotationFeedback instances.
    """
    try:
        from langsmith import Client  # noqa: PLC0415
    except ImportError as e:
        msg = "langsmith package is required for LangSmith import"
        raise ImportError(msg) from e

    import os  # noqa: PLC0415

    if not os.getenv("LANGSMITH_API_KEY"):
        msg = "LANGSMITH_API_KEY environment variable is not set"
        raise ValueError(msg)

    client = Client()
    logger.info("Querying feedback from LangSmith project '%s'", project)

    annotations = []

    try:
        # Query runs with feedback
        from datetime import timedelta  # noqa: PLC0415

        start_time = datetime.now(tz=UTC) - timedelta(days=days_back)

        runs = list(
            client.list_runs(
                project_name=project,
                start_time=start_time,
                has_feedback=True,
            )
        )

        for run in runs:
            # Get feedback for this run
            feedbacks = list(client.list_feedback(run_ids=[run.id]))

            for fb in feedbacks:
                # Convert LangSmith feedback to our format
                has_passing_score = fb.score is not None and fb.score >= LANGSMITH_SCORE_THRESHOLD
                is_correct = has_passing_score

                annotation = AnnotationFeedback(
                    run_id=str(run.id),
                    question=run.inputs.get("question", "") if run.inputs else "",
                    original_answer=run.outputs.get("answer", "") if run.outputs else "",
                    is_correct=is_correct,
                    corrected_answer=fb.comment if not is_correct and fb.comment else None,
                    quality_score=int(fb.score * 5) if fb.score is not None else None,
                    feedback_notes=fb.comment or "",
                    annotator_id=str(fb.created_by) if hasattr(fb, "created_by") else "",
                    annotation_timestamp=fb.created_at.isoformat() if fb.created_at else "",
                )
                annotations.append(annotation)

    except Exception as e:
        logger.error("Failed to load from LangSmith: %s", e)

    logger.info("Loaded %d annotations from LangSmith", len(annotations))
    return annotations


def generate_evaluation_example(
    feedback: AnnotationFeedback,
    example_id: str,
) -> NewEvaluationExample | None:
    """Generate a new evaluation example from feedback.

    Args:
        feedback: The annotation feedback.
        example_id: ID to assign to the example.

    Returns:
        NewEvaluationExample if valid, None otherwise.
    """
    # Determine the ground truth answer
    if feedback.is_correct:
        ground_truth = feedback.original_answer
        confidence = 0.9  # High confidence for confirmed correct
    elif feedback.corrected_answer:
        ground_truth = feedback.corrected_answer
        confidence = 0.95  # Higher confidence for human-corrected
    else:
        return None  # Can't generate without ground truth

    # Build tags
    tags = ["human-feedback", "generated"]
    if feedback.is_correct:
        tags.append("verified-correct")
    else:
        tags.append("human-corrected")

    if feedback.quality_score is not None:
        if feedback.quality_score >= HIGH_QUALITY_THRESHOLD:
            tags.append("high-quality")
        elif feedback.quality_score <= LOW_QUALITY_THRESHOLD:
            tags.append("needs-review")

    return NewEvaluationExample(
        id=example_id,
        question=feedback.question,
        ground_truth=ground_truth,
        source_run_id=feedback.run_id,
        confidence=confidence,
        tags=tags,
        metadata={
            "original_is_correct": feedback.is_correct,
            "quality_score": feedback.quality_score,
            "annotator_id": feedback.annotator_id,
            "annotation_timestamp": feedback.annotation_timestamp,
            "factual_errors": feedback.factual_errors,
            "missing_information": feedback.missing_information,
        },
    )


def process_annotations(
    annotations: list[AnnotationFeedback],
    *,
    generate_examples: bool = True,
    validate_only: bool = False,
) -> tuple[list[NewEvaluationExample], ImportStats]:
    """Process annotations and generate statistics.

    Args:
        annotations: List of feedback annotations.
        generate_examples: Whether to generate new evaluation examples.
        validate_only: Only validate, don't generate examples.

    Returns:
        Tuple of (generated examples, import statistics).
    """
    stats = ImportStats()
    examples = []
    quality_scores = []

    for i, feedback in enumerate(annotations):
        stats.total_annotations += 1

        # Validate
        errors = validate_feedback(feedback)
        if errors:
            stats.validation_errors += 1
            logger.warning(
                "Validation errors for run %s: %s",
                feedback.run_id,
                ", ".join(errors),
            )
            continue

        # Count statistics
        if feedback.is_correct:
            stats.correct_answers += 1
        else:
            stats.incorrect_answers += 1
            if feedback.corrected_answer:
                stats.with_corrections += 1

        if feedback.quality_score is not None:
            quality_scores.append(feedback.quality_score)

        # Generate example
        if generate_examples and not validate_only:
            example_id = f"feedback_{i + 1:04d}"
            example = generate_evaluation_example(feedback, example_id)
            if example:
                examples.append(example)
                stats.examples_generated += 1

    # Calculate average quality score
    if quality_scores:
        stats.average_quality_score = sum(quality_scores) / len(quality_scores)

    return examples, stats


def save_examples(
    examples: list[NewEvaluationExample],
    output_path: Path,
) -> None:
    """Save generated examples to JSON file.

    Args:
        examples: List of examples to save.
        output_path: Output file path.
    """
    data = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "total_examples": len(examples),
        "examples": [e.to_dict() for e in examples],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Saved %d examples to %s", len(examples), output_path)


def save_stats(
    stats: ImportStats,
    output_path: Path,
) -> None:
    """Save import statistics to JSON file.

    Args:
        stats: Import statistics.
        output_path: Output file path.
    """
    data = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        **stats.to_dict(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved statistics to %s", output_path)


def main() -> int:  # noqa: PLR0911, PLR0915
    """Main entry point for feedback import.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Import human feedback annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Input JSON file with annotations",
    )
    parser.add_argument(
        "--from-langsmith",
        action="store_true",
        help="Import from LangSmith feedback instead of file",
    )
    parser.add_argument(
        "--project",
        "-p",
        default="jama-mcp-graphrag",
        help="LangSmith project name (for --from-langsmith)",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=7,
        help="Days to look back (for --from-langsmith)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/feedback/new_examples.json"),
        help="Output file for generated examples",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("data/feedback/import_stats.json"),
        help="Output file for import statistics",
    )
    parser.add_argument(
        "--generate-examples",
        action="store_true",
        default=True,
        help="Generate new evaluation examples from feedback",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Don't generate examples, just compute statistics",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate annotations without processing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine generate_examples flag
    generate_examples = args.generate_examples and not args.no_generate

    try:
        # Load annotations
        if args.from_langsmith:
            annotations = load_annotations_from_langsmith(args.project, args.days)
        elif args.input:
            annotations = load_annotations_from_json(args.input)
        else:
            logger.error("Either --input or --from-langsmith is required")
            return 1

        if not annotations:
            logger.info("No annotations found")
            return 0

        # Process annotations
        examples, stats = process_annotations(
            annotations,
            generate_examples=generate_examples,
            validate_only=args.validate_only,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("IMPORT SUMMARY")
        print("=" * 60)
        print(f"Total annotations:     {stats.total_annotations}")
        print(f"Correct answers:       {stats.correct_answers}")
        print(f"Incorrect answers:     {stats.incorrect_answers}")
        print(f"With corrections:      {stats.with_corrections}")
        print(f"Validation errors:     {stats.validation_errors}")
        print(f"Average quality score: {stats.average_quality_score:.2f}")
        if stats.total_annotations > 0:
            accuracy = stats.correct_answers / stats.total_annotations * 100
            print(f"Model accuracy:        {accuracy:.1f}%")
        print(f"Examples generated:    {stats.examples_generated}")
        print("=" * 60 + "\n")

        if args.validate_only:
            print("VALIDATE ONLY - No files written")
            return 0

        # Save outputs
        if examples:
            save_examples(examples, args.output)
            print(f"Saved {len(examples)} examples to {args.output}")

        save_stats(stats, args.stats_output)
        print(f"Saved statistics to {args.stats_output}")

        return 0  # noqa: TRY300

    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        return 1
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON: %s", e)
        return 1
    except Exception:
        logger.exception("Import failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
