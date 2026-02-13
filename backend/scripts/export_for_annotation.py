#!/usr/bin/env python3
"""Export low-confidence evaluation runs for human annotation.

This script queries evaluation runs from LangSmith and exports those
with low confidence scores to an annotation queue for human review.

Note:
    Annotation queues (Phase 5) now automatically route negative user
    feedback to intent-specific queues in LangSmith. This manual
    export/import workflow complements that by surfacing low-confidence
    runs that users did not explicitly flag.

Workflow:
1. Query LangSmith for recent evaluation runs
2. Filter runs below confidence thresholds
3. Export to annotation queue (LangSmith) or local JSON

Usage:
    # Export to local JSON file
    python scripts/export_for_annotation.py --output annotations.json

    # Export runs below threshold
    python scripts/export_for_annotation.py --threshold 0.7

    # Export from specific project
    python scripts/export_for_annotation.py --project my-project

    # Dry run to see what would be exported
    python scripts/export_for_annotation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langsmith import Client

# Constants
PREVIEW_LIMIT = 10

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AnnotationCandidate:
    """A candidate run for human annotation.

    Attributes:
        run_id: LangSmith run ID.
        question: The input question.
        answer: The generated answer.
        contexts: Retrieved contexts used.
        ground_truth: Expected answer if available.
        metrics: Computed evaluation metrics.
        confidence_score: Overall confidence score.
        timestamp: When the run was executed.
        reason: Why this run was flagged for annotation.
        metadata: Additional run metadata.
    """

    run_id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None
    metrics: dict[str, float]
    confidence_score: float
    timestamp: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ExportConfig:
    """Configuration for annotation export.

    Attributes:
        project: LangSmith project name.
        confidence_threshold: Minimum confidence to exclude from annotation.
        faithfulness_threshold: Minimum faithfulness score.
        relevancy_threshold: Minimum answer relevancy score.
        max_runs: Maximum number of runs to export.
        days_back: Number of days to look back for runs.
        include_high_latency: Whether to include high-latency runs.
        latency_threshold_ms: Latency threshold in milliseconds.
    """

    project: str = "jama-mcp-graphrag"
    confidence_threshold: float = 0.7
    faithfulness_threshold: float = 0.6
    relevancy_threshold: float = 0.6
    max_runs: int = 100
    days_back: int = 7
    include_high_latency: bool = True
    latency_threshold_ms: float = 5000.0


def get_langsmith_client() -> Client:
    """Get LangSmith client with error handling.

    Returns:
        LangSmith Client instance.

    Raises:
        ImportError: If langsmith is not installed.
        ValueError: If API key is not configured.
    """
    try:
        from langsmith import Client as LangSmithClient
    except ImportError as e:
        msg = "langsmith package is required. Install with: pip install langsmith"
        raise ImportError(msg) from e

    if not os.getenv("LANGSMITH_API_KEY"):
        msg = "LANGSMITH_API_KEY environment variable is not set"
        raise ValueError(msg)

    return LangSmithClient()


def query_evaluation_runs(
    client: Any,
    config: ExportConfig,
) -> list[dict[str, Any]]:
    """Query LangSmith for recent evaluation runs.

    Args:
        client: LangSmith client.
        config: Export configuration.

    Returns:
        List of run dictionaries.
    """
    logger.info(
        "Querying runs from project '%s' (last %d days)",
        config.project,
        config.days_back,
    )

    start_time = datetime.now(tz=UTC) - timedelta(days=config.days_back)

    try:
        # Query runs from the project
        runs = list(
            client.list_runs(
                project_name=config.project,
                start_time=start_time,
                run_type="chain",
                limit=config.max_runs * 2,  # Get extra to account for filtering
            )
        )
        logger.info("Found %d total runs", len(runs))
        return runs
    except Exception as e:
        logger.error("Failed to query runs: %s", e)
        return []


def extract_metrics_from_run(run: Any) -> dict[str, float]:
    """Extract evaluation metrics from a run.

    Args:
        run: LangSmith run object.

    Returns:
        Dictionary of metric name to value.
    """
    metrics = {}

    # Check outputs for metrics
    if hasattr(run, "outputs") and run.outputs:
        outputs = run.outputs
        if isinstance(outputs, dict):
            # Look for metrics in various locations
            if "metrics" in outputs:
                metrics.update(outputs["metrics"])
            # Direct metric fields
            for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if key in outputs:
                    metrics[key] = float(outputs[key])

    # Check feedback for metrics
    if hasattr(run, "feedback_stats") and run.feedback_stats:
        for key, value in run.feedback_stats.items():
            if isinstance(value, dict) and "avg" in value and value["avg"] is not None:
                metrics[key] = float(value["avg"])
            elif isinstance(value, int | float):
                metrics[key] = float(value)

    return metrics


def calculate_confidence_score(metrics: dict[str, float]) -> float:
    """Calculate overall confidence score from metrics.

    Args:
        metrics: Dictionary of evaluation metrics.

    Returns:
        Confidence score between 0 and 1.
    """
    if not metrics:
        return 0.0

    # Weight the metrics
    weights = {
        "faithfulness": 0.3,
        "answer_relevancy": 0.3,
        "context_precision": 0.2,
        "context_recall": 0.2,
    }

    total_weight = 0.0
    weighted_sum = 0.0

    for metric, weight in weights.items():
        if metric in metrics:
            weighted_sum += metrics[metric] * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


def determine_annotation_reason(
    metrics: dict[str, float],
    confidence: float,
    config: ExportConfig,
    latency_ms: float | None = None,
) -> str | None:
    """Determine why a run should be flagged for annotation.

    Args:
        metrics: Evaluation metrics.
        confidence: Overall confidence score.
        config: Export configuration.
        latency_ms: Run latency in milliseconds.

    Returns:
        Reason string if should be annotated, None otherwise.
    """
    reasons = []

    if confidence < config.confidence_threshold:
        reasons.append(f"Low confidence ({confidence:.2f} < {config.confidence_threshold})")

    if metrics.get("faithfulness", 1.0) < config.faithfulness_threshold:
        faith_val = metrics.get("faithfulness", 0)
        reasons.append(f"Low faithfulness ({faith_val:.2f} < {config.faithfulness_threshold})")

    if metrics.get("answer_relevancy", 1.0) < config.relevancy_threshold:
        rel_val = metrics.get("answer_relevancy", 0)
        reasons.append(f"Low relevancy ({rel_val:.2f} < {config.relevancy_threshold})")

    if config.include_high_latency and latency_ms and latency_ms > config.latency_threshold_ms:
        reasons.append(f"High latency ({latency_ms:.0f}ms > {config.latency_threshold_ms}ms)")

    if not reasons:
        return None

    return "; ".join(reasons)


def convert_run_to_candidate(
    run: Any,
    reason: str,
    metrics: dict[str, float],
    confidence: float,
) -> AnnotationCandidate:
    """Convert a LangSmith run to an annotation candidate.

    Args:
        run: LangSmith run object.
        reason: Reason for annotation.
        metrics: Evaluation metrics.
        confidence: Confidence score.

    Returns:
        AnnotationCandidate instance.
    """
    # Extract inputs
    inputs = run.inputs or {}
    question = inputs.get("question", inputs.get("query", inputs.get("input", "")))

    # Extract outputs
    outputs = run.outputs or {}
    answer = outputs.get("answer", outputs.get("output", outputs.get("response", "")))
    contexts = outputs.get("contexts", outputs.get("sources", []))
    if isinstance(contexts, list) and contexts and isinstance(contexts[0], dict):
        contexts = [c.get("content", c.get("title", str(c))) for c in contexts]

    ground_truth = outputs.get("ground_truth", inputs.get("ground_truth"))

    # Get timestamp
    timestamp = run.start_time.isoformat() if hasattr(run, "start_time") and run.start_time else ""

    # Build metadata
    metadata = {
        "run_name": getattr(run, "name", ""),
        "run_type": getattr(run, "run_type", ""),
        "latency_ms": (
            (run.end_time - run.start_time).total_seconds() * 1000
            if hasattr(run, "end_time") and run.end_time and run.start_time
            else None
        ),
        "tags": getattr(run, "tags", []),
    }

    return AnnotationCandidate(
        run_id=str(run.id),
        question=str(question),
        answer=str(answer),
        contexts=contexts if isinstance(contexts, list) else [],
        ground_truth=ground_truth,
        metrics=metrics,
        confidence_score=confidence,
        timestamp=timestamp,
        reason=reason,
        metadata=metadata,
    )


def filter_runs_for_annotation(
    runs: list[Any],
    config: ExportConfig,
) -> list[AnnotationCandidate]:
    """Filter runs to find annotation candidates.

    Args:
        runs: List of LangSmith runs.
        config: Export configuration.

    Returns:
        List of AnnotationCandidate instances.
    """
    candidates = []

    for run in runs:
        # Extract metrics
        metrics = extract_metrics_from_run(run)
        confidence = calculate_confidence_score(metrics)

        # Calculate latency
        latency_ms = None
        has_end = hasattr(run, "end_time") and run.end_time
        has_start = hasattr(run, "start_time") and run.start_time
        if has_end and has_start:
            latency_ms = (run.end_time - run.start_time).total_seconds() * 1000

        # Check if should be annotated
        reason = determine_annotation_reason(metrics, confidence, config, latency_ms)

        if reason:
            candidate = convert_run_to_candidate(run, reason, metrics, confidence)
            candidates.append(candidate)

            if len(candidates) >= config.max_runs:
                break

    logger.info("Found %d candidates for annotation", len(candidates))
    return candidates


def export_to_json(
    candidates: list[AnnotationCandidate],
    output_path: Path,
) -> None:
    """Export annotation candidates to JSON file.

    Args:
        candidates: List of annotation candidates.
        output_path: Path to output file.
    """
    data = {
        "exported_at": datetime.now(tz=UTC).isoformat(),
        "total_candidates": len(candidates),
        "candidates": [c.to_dict() for c in candidates],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Exported %d candidates to %s", len(candidates), output_path)


def export_to_langsmith_queue(
    client: Any,
    candidates: list[AnnotationCandidate],
) -> int:
    """Export annotation candidates to LangSmith annotation queue.

    Args:
        client: LangSmith client.
        candidates: List of annotation candidates.

    Returns:
        Number of successfully queued items.
    """
    queued = 0

    for candidate in candidates:
        try:
            # Create annotation task in LangSmith
            # Note: This uses the LangSmith annotation queue API
            client.create_feedback(
                run_id=candidate.run_id,
                key="needs_annotation",
                value=1,
                comment=candidate.reason,
            )
            queued += 1
        except Exception as e:
            logger.warning("Failed to queue run %s: %s", candidate.run_id, e)

    logger.info("Queued %d runs for annotation", queued)
    return queued


def main() -> int:
    """Main entry point for annotation export.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Export low-confidence runs for human annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/annotations/pending.json"),
        help="Output file path for JSON export",
    )
    parser.add_argument(
        "--project",
        "-p",
        default="jama-mcp-graphrag",
        help="LangSmith project name",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.7,
        help="Confidence threshold (runs below this are exported)",
    )
    parser.add_argument(
        "--max-runs",
        "-m",
        type=int,
        default=100,
        help="Maximum number of runs to export",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=7,
        help="Number of days to look back",
    )
    parser.add_argument(
        "--queue",
        action="store_true",
        help="Also add to LangSmith annotation queue",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without writing",
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

    # Build configuration
    config = ExportConfig(
        project=args.project,
        confidence_threshold=args.threshold,
        max_runs=args.max_runs,
        days_back=args.days,
    )

    try:
        # Get LangSmith client
        client = get_langsmith_client()

        # Query runs
        runs = query_evaluation_runs(client, config)

        if not runs:
            logger.info("No runs found matching criteria")
            return 0

        # Filter for annotation candidates
        candidates = filter_runs_for_annotation(runs, config)

        if not candidates:
            logger.info("No runs need annotation (all above threshold)")
            return 0

        # Show summary
        print(f"\nFound {len(candidates)} runs for annotation:")
        print("-" * 60)
        for i, candidate in enumerate(candidates[:PREVIEW_LIMIT], 1):
            print(f"{i}. {candidate.question[:50]}...")
            print(f"   Confidence: {candidate.confidence_score:.2f}")
            print(f"   Reason: {candidate.reason}")
            print()

        if len(candidates) > PREVIEW_LIMIT:
            print(f"... and {len(candidates) - PREVIEW_LIMIT} more\n")

        if args.dry_run:
            print("DRY RUN - No files written")
            return 0

        # Export to JSON
        export_to_json(candidates, args.output)

        # Optionally queue in LangSmith
        if args.queue:
            export_to_langsmith_queue(client, candidates)

        print(f"\nExported {len(candidates)} candidates to {args.output}")
        return 0

    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        return 1
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1
    except Exception:
        logger.exception("Export failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
