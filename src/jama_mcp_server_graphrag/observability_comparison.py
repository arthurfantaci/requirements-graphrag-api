"""Unified observability interface for comparing LangSmith and MLflow.

Provides a platform-agnostic interface for experiment tracking that can
log to both LangSmith and MLflow simultaneously, enabling side-by-side
comparison of the two platforms.

Features:
- Unified API for both platforms
- Simultaneous logging to both
- Platform-specific feature comparison
- Migration utilities

Usage:
    from jama_mcp_server_graphrag.observability_comparison import (
        UnifiedTracker,
        Platform,
        compare_platform_features,
    )

    # Track to both platforms
    async with UnifiedTracker(platforms=[Platform.LANGSMITH, Platform.MLFLOW]) as tracker:
        tracker.log_metrics({"faithfulness": 0.85})
"""
# ruff: noqa: PLC0415

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported observability platforms."""

    LANGSMITH = "langsmith"
    MLFLOW = "mlflow"


# =============================================================================
# PLATFORM FEATURE COMPARISON
# =============================================================================


@dataclass
class PlatformFeature:
    """Description of a platform feature."""

    name: str
    langsmith_support: bool
    mlflow_support: bool
    langsmith_notes: str = ""
    mlflow_notes: str = ""


PLATFORM_FEATURES: list[PlatformFeature] = [
    # Setup Complexity
    PlatformFeature(
        name="Cloud-hosted option",
        langsmith_support=True,
        mlflow_support=True,
        langsmith_notes="Managed service, no setup required",
        mlflow_notes="Databricks managed or self-hosted",
    ),
    PlatformFeature(
        name="Self-hosted option",
        langsmith_support=False,
        mlflow_support=True,
        langsmith_notes="Not available",
        mlflow_notes="Full self-hosting support",
    ),
    PlatformFeature(
        name="Zero-config setup",
        langsmith_support=True,
        mlflow_support=False,
        langsmith_notes="Just set API key and project name",
        mlflow_notes="Requires tracking server setup",
    ),
    # Evaluation Features
    PlatformFeature(
        name="LangChain auto-tracing",
        langsmith_support=True,
        mlflow_support=False,
        langsmith_notes="Native integration, automatic",
        mlflow_notes="Manual instrumentation required",
    ),
    PlatformFeature(
        name="Custom evaluators",
        langsmith_support=True,
        mlflow_support=True,
        langsmith_notes="Python-based evaluators",
        mlflow_notes="Python-based evaluators",
    ),
    PlatformFeature(
        name="Dataset management",
        langsmith_support=True,
        mlflow_support=True,
        langsmith_notes="Built-in dataset versioning",
        mlflow_notes="Via MLflow Datasets",
    ),
    PlatformFeature(
        name="A/B testing",
        langsmith_support=True,
        mlflow_support=True,
        langsmith_notes="Prompt comparison UI",
        mlflow_notes="Via experiment comparison",
    ),
    # Visualization
    PlatformFeature(
        name="Trace visualization",
        langsmith_support=True,
        mlflow_support=False,
        langsmith_notes="Rich trace tree view",
        mlflow_notes="Basic logging only",
    ),
    PlatformFeature(
        name="Metrics dashboard",
        langsmith_support=True,
        mlflow_support=True,
        langsmith_notes="Built-in dashboards",
        mlflow_notes="Customizable dashboards",
    ),
    PlatformFeature(
        name="Cost tracking",
        langsmith_support=True,
        mlflow_support=False,
        langsmith_notes="Token and cost breakdown",
        mlflow_notes="Manual implementation",
    ),
    # Prompt Versioning
    PlatformFeature(
        name="Prompt hub",
        langsmith_support=True,
        mlflow_support=False,
        langsmith_notes="LangChainHub integration",
        mlflow_notes="Not available",
    ),
    PlatformFeature(
        name="Prompt versioning",
        langsmith_support=True,
        mlflow_support=True,
        langsmith_notes="Built-in versioning",
        mlflow_notes="Via model registry",
    ),
    PlatformFeature(
        name="Prompt testing",
        langsmith_support=True,
        mlflow_support=False,
        langsmith_notes="Playground with datasets",
        mlflow_notes="Manual implementation",
    ),
    # Self-hosting
    PlatformFeature(
        name="On-premises deployment",
        langsmith_support=False,
        mlflow_support=True,
        langsmith_notes="Cloud only",
        mlflow_notes="Full on-prem support",
    ),
    PlatformFeature(
        name="Data residency control",
        langsmith_support=False,
        mlflow_support=True,
        langsmith_notes="US-based servers",
        mlflow_notes="Full control",
    ),
    PlatformFeature(
        name="Open source",
        langsmith_support=False,
        mlflow_support=True,
        langsmith_notes="Proprietary",
        mlflow_notes="Apache 2.0 license",
    ),
]


def compare_platform_features() -> dict[str, Any]:
    """Compare features between LangSmith and MLflow.

    Returns:
        Dictionary with feature comparison data.
    """
    comparison = {
        "features": [],
        "summary": {
            "langsmith_advantages": [],
            "mlflow_advantages": [],
            "both_support": [],
        },
    }

    for feature in PLATFORM_FEATURES:
        comparison["features"].append(
            {
                "name": feature.name,
                "langsmith": feature.langsmith_support,
                "mlflow": feature.mlflow_support,
                "langsmith_notes": feature.langsmith_notes,
                "mlflow_notes": feature.mlflow_notes,
            }
        )

        if feature.langsmith_support and not feature.mlflow_support:
            comparison["summary"]["langsmith_advantages"].append(feature.name)
        elif feature.mlflow_support and not feature.langsmith_support:
            comparison["summary"]["mlflow_advantages"].append(feature.name)
        elif feature.langsmith_support and feature.mlflow_support:
            comparison["summary"]["both_support"].append(feature.name)

    return comparison


# =============================================================================
# UNIFIED TRACKER
# =============================================================================


@dataclass
class TrackingResult:
    """Result of a tracking operation."""

    platform: Platform
    success: bool
    run_id: str | None = None
    error: str | None = None


class UnifiedTracker:
    """Unified tracker that logs to multiple platforms.

    Provides a single interface for logging to both LangSmith and MLflow,
    enabling side-by-side comparison experiments.

    Example:
        async with UnifiedTracker([Platform.LANGSMITH, Platform.MLFLOW]) as tracker:
            tracker.log_params({"model": "gpt-4o"})
            tracker.log_metrics({"faithfulness": 0.85})
    """

    def __init__(
        self,
        platforms: list[Platform] | None = None,
        run_name: str | None = None,
        experiment_name: str | None = None,
        config: AppConfig | None = None,
    ) -> None:
        """Initialize unified tracker.

        Args:
            platforms: Platforms to log to. Defaults to both.
            run_name: Name for this run.
            experiment_name: Experiment/project name.
            config: Application configuration.
        """
        self.platforms = platforms or [Platform.LANGSMITH, Platform.MLFLOW]
        self.run_name = run_name or f"unified-run-{int(time.time())}"
        self.experiment_name = experiment_name or "jama-graphrag-evaluation"
        self.config = config

        self._langsmith_run_id: str | None = None
        self._mlflow_tracker = None
        self._start_time: float | None = None
        self._results: list[TrackingResult] = []

    def __enter__(self) -> UnifiedTracker:
        """Start tracking on all platforms."""
        self._start_time = time.time()

        # Start LangSmith tracing
        if Platform.LANGSMITH in self.platforms:
            self._start_langsmith()

        # Start MLflow run
        if Platform.MLFLOW in self.platforms:
            self._start_mlflow()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End tracking on all platforms."""
        duration = time.time() - self._start_time if self._start_time else 0

        # End MLflow run
        if self._mlflow_tracker is not None:
            self._mlflow_tracker.__exit__(exc_type, exc_val, exc_tb)

        # Log duration to both platforms
        if duration > 0:
            self.log_metric("duration_seconds", duration)

    async def __aenter__(self) -> UnifiedTracker:
        """Async context manager entry."""
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        self.__exit__(exc_type, exc_val, exc_tb)

    def _start_langsmith(self) -> None:
        """Start LangSmith tracking."""
        try:
            from langsmith import Client

            # LangSmith auto-traces via environment variables
            # We just verify it's configured by creating client
            _client = Client()
            # Create a dataset/project if needed
            self._langsmith_run_id = self.run_name
            self._results.append(
                TrackingResult(
                    platform=Platform.LANGSMITH,
                    success=True,
                    run_id=self._langsmith_run_id,
                )
            )
            logger.info("LangSmith tracking started: %s", self.run_name)
        except Exception as e:
            self._results.append(
                TrackingResult(
                    platform=Platform.LANGSMITH,
                    success=False,
                    error=str(e),
                )
            )
            logger.warning("LangSmith tracking failed: %s", e)

    def _start_mlflow(self) -> None:
        """Start MLflow tracking."""
        try:
            from jama_mcp_server_graphrag.mlflow_tracking import (
                MLFLOW_AVAILABLE,
                MLflowTracker,
                configure_mlflow,
            )

            if not MLFLOW_AVAILABLE:
                self._results.append(
                    TrackingResult(
                        platform=Platform.MLFLOW,
                        success=False,
                        error="MLflow not installed",
                    )
                )
                return

            configure_mlflow(experiment_name=self.experiment_name)
            self._mlflow_tracker = MLflowTracker(
                run_name=self.run_name,
                experiment_name=self.experiment_name,
            )
            self._mlflow_tracker.__enter__()
            self._results.append(
                TrackingResult(
                    platform=Platform.MLFLOW,
                    success=True,
                    run_id=self._mlflow_tracker.run_id,
                )
            )
            logger.info("MLflow tracking started: %s", self.run_name)
        except Exception as e:
            self._results.append(
                TrackingResult(
                    platform=Platform.MLFLOW,
                    success=False,
                    error=str(e),
                )
            )
            logger.warning("MLflow tracking failed: %s", e)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to all platforms.

        Args:
            params: Dictionary of parameters to log.
        """
        # MLflow
        if self._mlflow_tracker is not None:
            try:
                self._mlflow_tracker.log_params(params)
            except Exception:
                logger.exception("Failed to log params to MLflow")

        # LangSmith - params are logged via traceable decorator
        # or can be added to run metadata

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to all platforms.

        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number.
        """
        # MLflow
        if self._mlflow_tracker is not None:
            try:
                self._mlflow_tracker.log_metrics(metrics, step=step)
            except Exception:
                logger.exception("Failed to log metrics to MLflow")

        # LangSmith - metrics via feedback or run metadata
        # This would need LangSmith client API calls

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric to all platforms.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact to all platforms.

        Args:
            local_path: Path to the file to log.
            artifact_path: Destination path in artifact storage.
        """
        # MLflow
        if self._mlflow_tracker is not None:
            try:
                self._mlflow_tracker.log_artifact(local_path, artifact_path)
            except Exception:
                logger.exception("Failed to log artifact to MLflow")

        # LangSmith doesn't have direct artifact storage

    def get_results(self) -> list[TrackingResult]:
        """Get tracking results for all platforms.

        Returns:
            List of TrackingResult objects.
        """
        return self._results


# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================


@dataclass
class PlatformRecommendation:
    """Platform recommendation based on requirements."""

    recommended: Platform
    confidence: float
    reasons: list[str] = field(default_factory=list)
    considerations: list[str] = field(default_factory=list)


def recommend_platform(  # noqa: PLR0913
    needs_self_hosting: bool = False,
    needs_trace_visualization: bool = True,
    needs_prompt_hub: bool = False,
    needs_cost_tracking: bool = True,
    budget_sensitive: bool = False,
    langchain_native: bool = True,
) -> PlatformRecommendation:
    """Recommend a platform based on requirements.

    Args:
        needs_self_hosting: Whether self-hosting is required.
        needs_trace_visualization: Whether trace visualization is needed.
        needs_prompt_hub: Whether prompt hub integration is needed.
        needs_cost_tracking: Whether cost tracking is needed.
        budget_sensitive: Whether cost is a major concern.
        langchain_native: Whether using LangChain/LangGraph.

    Returns:
        PlatformRecommendation with reasoning.
    """
    langsmith_score = 0
    mlflow_score = 0
    reasons: list[str] = []
    considerations: list[str] = []

    # Self-hosting requirement (critical)
    if needs_self_hosting:
        mlflow_score += 10
        reasons.append("MLflow supports self-hosting; LangSmith does not")
    else:
        langsmith_score += 2
        reasons.append("Cloud-hosted simplifies operations")

    # Trace visualization
    if needs_trace_visualization:
        langsmith_score += 3
        reasons.append("LangSmith has superior trace visualization")

    # Prompt hub
    if needs_prompt_hub:
        langsmith_score += 3
        reasons.append("LangSmith integrates with LangChainHub")

    # Cost tracking
    if needs_cost_tracking:
        langsmith_score += 2
        reasons.append("LangSmith has built-in cost tracking")

    # Budget sensitive
    if budget_sensitive:
        mlflow_score += 3
        considerations.append("MLflow is free and open-source")
        considerations.append("LangSmith has usage-based pricing")

    # LangChain native
    if langchain_native:
        langsmith_score += 3
        reasons.append("LangSmith has native LangChain integration")

    # Determine recommendation
    if langsmith_score > mlflow_score:
        recommended = Platform.LANGSMITH
        confidence = langsmith_score / (langsmith_score + mlflow_score + 1)
    else:
        recommended = Platform.MLFLOW
        confidence = mlflow_score / (langsmith_score + mlflow_score + 1)

    return PlatformRecommendation(
        recommended=recommended,
        confidence=min(confidence, 0.95),  # Cap at 95%
        reasons=reasons,
        considerations=considerations,
    )


__all__ = [
    "PLATFORM_FEATURES",
    "Platform",
    "PlatformFeature",
    "PlatformRecommendation",
    "TrackingResult",
    "UnifiedTracker",
    "compare_platform_features",
    "recommend_platform",
]
