"""MLflow tracking integration for GraphRAG evaluation.

Provides MLflow-based experiment tracking as an alternative to LangSmith,
enabling self-hosted observability and experiment comparison.

Features:
- Experiment tracking for RAG evaluations
- Metric logging (faithfulness, relevancy, precision, recall)
- Artifact storage for evaluation results
- Model versioning and comparison

Usage:
    from jama_mcp_server_graphrag.mlflow_tracking import (
        configure_mlflow,
        MLflowTracker,
    )

    # Configure MLflow
    configure_mlflow(tracking_uri="http://localhost:5000")

    # Track evaluation run
    async with MLflowTracker("rag-evaluation") as tracker:
        tracker.log_params({"model": "gpt-4o", "k": 6})
        tracker.log_metrics(metrics.to_dict())
        tracker.log_artifact("results.json")
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

logger = logging.getLogger(__name__)

# MLflow import with graceful fallback
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore[assignment]
    MlflowClient = None  # type: ignore[assignment, misc]


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MLflowConfig:
    """Configuration for MLflow tracking.

    Attributes:
        tracking_uri: MLflow tracking server URI.
        experiment_name: Default experiment name.
        artifact_location: Base path for artifacts.
        registry_uri: Model registry URI (optional).
        tags: Default tags for all runs.
    """

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "jama-graphrag-evaluation"
    artifact_location: str | None = None
    registry_uri: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


def configure_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    artifact_location: str | None = None,
) -> bool:
    """Configure MLflow tracking.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: Experiment name.
        artifact_location: Artifact storage location.

    Returns:
        True if MLflow was configured successfully, False otherwise.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Install with: uv sync --extra mlflow")
        return False

    # Use environment variables or defaults
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "jama-graphrag-evaluation")

    try:
        mlflow.set_tracking_uri(uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            mlflow.create_experiment(
                exp_name,
                artifact_location=artifact_location,
            )

        mlflow.set_experiment(exp_name)
        logger.info("MLflow configured: uri=%s, experiment=%s", uri, exp_name)
    except Exception:
        logger.exception("Failed to configure MLflow")
        return False
    else:
        return True


def get_mlflow_status() -> dict[str, Any]:
    """Get current MLflow configuration status.

    Returns:
        Dictionary with MLflow status information.
    """
    if not MLFLOW_AVAILABLE:
        return {
            "available": False,
            "tracking_uri": None,
            "experiment": None,
            "error": "MLflow not installed",
        }

    try:
        return {
            "available": True,
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment": mlflow.get_experiment(mlflow.active_run().info.experiment_id).name
            if mlflow.active_run()
            else None,
            "active_run": mlflow.active_run().info.run_id if mlflow.active_run() else None,
        }
    except Exception as e:
        return {
            "available": True,
            "tracking_uri": mlflow.get_tracking_uri() if mlflow else None,
            "experiment": None,
            "error": str(e),
        }


# =============================================================================
# TRACKER CLASS
# =============================================================================


class MLflowTracker:
    """Context manager for MLflow run tracking.

    Provides a clean interface for logging parameters, metrics, and artifacts
    within a tracked run.

    Example:
        async with MLflowTracker("evaluation-run") as tracker:
            tracker.log_params({"model": "gpt-4o"})
            tracker.log_metrics({"faithfulness": 0.85})
    """

    def __init__(
        self,
        run_name: str | None = None,
        experiment_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> None:
        """Initialize MLflow tracker.

        Args:
            run_name: Name for this run.
            experiment_name: Experiment to log to.
            tags: Tags for this run.
            nested: Whether this is a nested run.
        """
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.tags = tags or {}
        self.nested = nested
        self._run = None
        self._start_time: float | None = None

    def __enter__(self) -> MLflowTracker:
        """Start MLflow run."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, tracking disabled")
            return self

        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        self._run = mlflow.start_run(
            run_name=self.run_name,
            tags=self.tags,
            nested=self.nested,
        )
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End MLflow run."""
        if self._run is not None and MLFLOW_AVAILABLE:
            # Log duration
            if self._start_time:
                duration = time.time() - self._start_time
                mlflow.log_metric("duration_seconds", duration)

            # Log exception if any
            if exc_type is not None:
                mlflow.set_tag("error", str(exc_val))

            mlflow.end_run()

    async def __aenter__(self) -> MLflowTracker:
        """Async context manager entry."""
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        self.__exit__(exc_type, exc_val, exc_tb)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to the current run.

        Args:
            params: Dictionary of parameters to log.
        """
        if not MLFLOW_AVAILABLE or self._run is None:
            return

        # MLflow requires string values for params
        str_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(str_params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to the current run.

        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number for time-series metrics.
        """
        if not MLFLOW_AVAILABLE or self._run is None:
            return

        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        if not MLFLOW_AVAILABLE or self._run is None:
            return

        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log an artifact file.

        Args:
            local_path: Path to the file to log.
            artifact_path: Destination path in artifact storage.
        """
        if not MLFLOW_AVAILABLE or self._run is None:
            return

        mlflow.log_artifact(str(local_path), artifact_path)

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log.
            artifact_file: Filename for the artifact.
        """
        if not MLFLOW_AVAILABLE or self._run is None:
            return

        mlflow.log_dict(dictionary, artifact_file)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag name.
            value: Tag value.
        """
        if not MLFLOW_AVAILABLE or self._run is None:
            return

        mlflow.set_tag(key, value)

    @property
    def run_id(self) -> str | None:
        """Get the current run ID."""
        if self._run is None:
            return None
        return self._run.info.run_id


# =============================================================================
# EVALUATION LOGGING HELPERS
# =============================================================================


def log_rag_evaluation(  # noqa: PLR0913
    question: str,
    answer: str,
    contexts: list[str],
    metrics: dict[str, float],
    ground_truth: str | None = None,
    latency_ms: float | None = None,
    run_name: str | None = None,
) -> str | None:
    """Log a RAG evaluation result to MLflow.

    Args:
        question: The user's question.
        answer: The generated answer.
        contexts: Retrieved context passages.
        metrics: Evaluation metrics.
        ground_truth: Expected answer (optional).
        latency_ms: Response latency in milliseconds.
        run_name: Name for this evaluation run.

    Returns:
        Run ID if logged successfully, None otherwise.
    """
    if not MLFLOW_AVAILABLE:
        return None

    with MLflowTracker(run_name=run_name or "rag-evaluation") as tracker:
        # Log parameters
        tracker.log_params(
            {
                "question_length": len(question),
                "answer_length": len(answer),
                "num_contexts": len(contexts),
                "has_ground_truth": ground_truth is not None,
            }
        )

        # Log metrics
        tracker.log_metrics(metrics)
        if latency_ms is not None:
            tracker.log_metric("latency_ms", latency_ms)

        # Log artifacts
        evaluation_data = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "metrics": metrics,
            "latency_ms": latency_ms,
        }
        tracker.log_dict(evaluation_data, "evaluation_result.json")

        return tracker.run_id


@contextmanager
def mlflow_evaluation_run(
    experiment_name: str = "jama-graphrag-evaluation",
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Generator[MLflowTracker, None, None]:
    """Context manager for MLflow evaluation runs.

    Args:
        experiment_name: Experiment name.
        run_name: Run name.
        tags: Run tags.

    Yields:
        MLflowTracker instance.

    Example:
        with mlflow_evaluation_run("my-experiment") as tracker:
            tracker.log_metrics({"accuracy": 0.95})
    """
    tracker = MLflowTracker(
        run_name=run_name,
        experiment_name=experiment_name,
        tags=tags,
    )
    with tracker:
        yield tracker


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================


def compare_runs(
    run_ids: list[str],
    metrics: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compare metrics across multiple MLflow runs.

    Args:
        run_ids: List of run IDs to compare.
        metrics: Specific metrics to compare. If None, compares all.

    Returns:
        Dictionary mapping run_id to metrics.
    """
    if not MLFLOW_AVAILABLE:
        return {}

    client = MlflowClient()
    results: dict[str, dict[str, float]] = {}

    for run_id in run_ids:
        try:
            run = client.get_run(run_id)
            run_metrics = run.data.metrics

            if metrics:
                run_metrics = {k: v for k, v in run_metrics.items() if k in metrics}

            results[run_id] = run_metrics
        except Exception:
            logger.exception("Failed to get run %s", run_id)

    return results


def get_best_run(
    experiment_name: str,
    metric: str,
    maximize: bool = True,
) -> dict[str, Any] | None:
    """Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Experiment to search.
        metric: Metric to optimize.
        maximize: Whether to maximize (True) or minimize (False).

    Returns:
        Best run info or None if not found.
    """
    if not MLFLOW_AVAILABLE:
        return None

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        order = "DESC" if maximize else "ASC"
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        if runs.empty:
            return None

        best = runs.iloc[0]
        params = {k.replace("params.", ""): v for k, v in best.items() if k.startswith("params.")}
        return {
            "run_id": best["run_id"],
            "metric_value": best[f"metrics.{metric}"],
            "params": params,
        }

    except Exception:
        logger.exception("Failed to get best run")
        return None


__all__ = [
    "MLFLOW_AVAILABLE",
    "MLflowConfig",
    "MLflowTracker",
    "compare_runs",
    "configure_mlflow",
    "get_best_run",
    "get_mlflow_status",
    "log_rag_evaluation",
    "mlflow_evaluation_run",
]
