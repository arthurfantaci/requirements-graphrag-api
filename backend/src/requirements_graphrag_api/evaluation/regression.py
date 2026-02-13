"""Regression gate logic for per-vector evaluation thresholds.

Checks experiment results against ``REGRESSION_THRESHOLDS`` from
``constants.py`` and enforces must_pass examples.

Usage:
    from requirements_graphrag_api.evaluation.regression import check_regression

    passed, report = check_regression("explanatory", experiment_results)
    if not passed:
        print(report)
        sys.exit(1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from requirements_graphrag_api.evaluation.constants import REGRESSION_THRESHOLDS

logger = logging.getLogger(__name__)


@dataclass
class RegressionReport:
    """Result of a regression check."""

    vector: str
    passed: bool
    metric_results: dict[str, MetricResult] = field(default_factory=dict)
    must_pass_failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Regression gate: {status} ({self.vector})"]

        for metric, result in sorted(self.metric_results.items()):
            indicator = "OK" if result.passed else "FAIL"
            lines.append(
                f"  [{indicator}] {metric}: {result.actual:.3f} (threshold: {result.threshold:.3f})"
            )

        if self.must_pass_failures:
            lines.append(f"  Must-pass failures: {', '.join(self.must_pass_failures)}")

        return "\n".join(lines)


@dataclass
class MetricResult:
    """Result for a single metric."""

    metric: str
    actual: float
    threshold: float
    passed: bool


def check_regression(
    vector: str,
    metric_scores: dict[str, float],
    *,
    must_pass_failures: list[str] | None = None,
) -> RegressionReport:
    """Check experiment results against regression thresholds.

    Args:
        vector: Vector name (explanatory, structured, conversational, intent).
        metric_scores: Dict mapping metric names to average scores.
        must_pass_failures: Optional list of must_pass example IDs that failed.

    Returns:
        RegressionReport with pass/fail status and details.
    """
    thresholds = REGRESSION_THRESHOLDS.get(vector, {})
    failures = must_pass_failures or []

    results: dict[str, MetricResult] = {}
    all_passed = True

    for metric, threshold in thresholds.items():
        actual = metric_scores.get(metric, 0.0)
        passed = actual >= threshold
        if not passed:
            all_passed = False
            logger.warning(
                "Regression FAIL: %s.%s = %.3f < %.3f",
                vector,
                metric,
                actual,
                threshold,
            )
        results[metric] = MetricResult(
            metric=metric,
            actual=actual,
            threshold=threshold,
            passed=passed,
        )

    if failures:
        all_passed = False
        logger.warning(
            "Must-pass failures in %s: %s",
            vector,
            ", ".join(failures),
        )

    return RegressionReport(
        vector=vector,
        passed=all_passed,
        metric_results=results,
        must_pass_failures=failures,
    )


def check_all_vectors(
    results: dict[str, dict[str, Any]],
) -> dict[str, RegressionReport]:
    """Check regression for all vectors at once.

    Args:
        results: Dict mapping vector names to their metric_scores dicts.
            Each dict may also have a "must_pass_failures" key.

    Returns:
        Dict mapping vector names to RegressionReports.
    """
    reports: dict[str, RegressionReport] = {}
    for vector, data in results.items():
        scores = {k: v for k, v in data.items() if k != "must_pass_failures"}
        failures = data.get("must_pass_failures", [])
        reports[vector] = check_regression(
            vector,
            scores,
            must_pass_failures=failures,
        )
    return reports


__all__ = [
    "MetricResult",
    "RegressionReport",
    "check_all_vectors",
    "check_regression",
]
