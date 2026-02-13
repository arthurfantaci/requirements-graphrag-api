#!/usr/bin/env python
"""Daily quality degradation detection.

Compares feedback scores from the last 7 days against a 30-day baseline
to detect statistically meaningful drops in response quality. Designed
to run on a schedule (cron, GitHub Actions, etc.) and exit non-zero
when degradation is detected.

Usage:
    # Run with defaults (7-day window vs 30-day baseline)
    uv run python scripts/quality_check.py --project graphrag-api-prod

    # Custom windows
    uv run python scripts/quality_check.py --project graphrag-api-prod \
        --recent-days 3 --baseline-days 14

    # Show details without failing
    uv run python scripts/quality_check.py --project graphrag-api-prod --dry-run

Exit codes:
    0 — no degradation detected (or insufficient data)
    1 — degradation detected
    2 — script error

Requires:
    LANGSMITH_API_KEY
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Minimum samples needed before comparing windows
MIN_SAMPLES = 5

# Score drop threshold (absolute) to trigger degradation alert
DEGRADATION_THRESHOLD = 0.10

# Feedback keys to monitor
MONITORED_KEYS: tuple[str, ...] = (
    "user-feedback",
    "hallucination",
    "coherence",
    "answer_relevancy",
    "online_cypher_parse",
    "online_cypher_execution",
)


def _collect_feedback_scores(
    client: Any,
    project: str,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, list[float]]:
    """Collect feedback scores grouped by key for a time window.

    Args:
        client: LangSmith Client.
        project: Project name.
        start_time: Window start.
        end_time: Window end.

    Returns:
        Dict mapping feedback key to list of scores.
    """
    scores: dict[str, list[float]] = {}

    try:
        runs = list(
            client.list_runs(
                project_name=project,
                start_time=start_time,
                end_time=end_time,
                has_feedback=True,
                limit=1000,
            )
        )
    except Exception:
        logger.warning("Failed to list runs for %s", project, exc_info=True)
        return scores

    for run in runs:
        try:
            feedbacks = list(client.list_feedback(run_ids=[run.id]))
        except Exception:
            logger.debug("Failed to list feedback for run %s", run.id, exc_info=True)
            continue

        for fb in feedbacks:
            if fb.key and fb.score is not None:
                scores.setdefault(fb.key, []).append(float(fb.score))

    return scores


def _compute_averages(scores: dict[str, list[float]]) -> dict[str, tuple[float, int]]:
    """Compute average and count per key.

    Returns:
        Dict mapping key to (average, count).
    """
    return {key: (sum(vals) / len(vals), len(vals)) for key, vals in scores.items() if vals}


def check_degradation(
    recent: dict[str, tuple[float, int]],
    baseline: dict[str, tuple[float, int]],
) -> list[dict[str, Any]]:
    """Compare recent averages against baseline, flag degradation.

    Args:
        recent: Recent window averages {key: (avg, count)}.
        baseline: Baseline window averages {key: (avg, count)}.

    Returns:
        List of degradation findings.
    """
    findings = []

    for key in MONITORED_KEYS:
        if key not in recent or key not in baseline:
            continue

        recent_avg, recent_n = recent[key]
        baseline_avg, baseline_n = baseline[key]

        if recent_n < MIN_SAMPLES or baseline_n < MIN_SAMPLES:
            continue

        drop = baseline_avg - recent_avg
        if drop >= DEGRADATION_THRESHOLD:
            findings.append(
                {
                    "key": key,
                    "baseline_avg": round(baseline_avg, 3),
                    "recent_avg": round(recent_avg, 3),
                    "drop": round(drop, 3),
                    "baseline_n": baseline_n,
                    "recent_n": recent_n,
                }
            )

    return findings


def main() -> int:
    """Main entry point."""
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set")
        return 2

    parser = argparse.ArgumentParser(
        description="Daily quality degradation detection",
    )
    parser.add_argument(
        "--project",
        "-p",
        default="graphrag-api-prod",
        help="LangSmith project name (default: graphrag-api-prod)",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=7,
        help="Recent window size in days (default: 7)",
    )
    parser.add_argument(
        "--baseline-days",
        type=int,
        default=30,
        help="Baseline window size in days (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show results without failing on degradation",
    )

    args = parser.parse_args()

    try:
        from langsmith import Client

        client = Client()
    except ImportError:
        logger.error("langsmith package not installed")
        return 2

    now = datetime.now(tz=UTC)
    recent_start = now - timedelta(days=args.recent_days)
    baseline_start = now - timedelta(days=args.baseline_days)

    logger.info("Project: %s", args.project)
    logger.info("Recent window: last %d days", args.recent_days)
    logger.info("Baseline window: last %d days", args.baseline_days)

    # Collect scores
    logger.info("Collecting recent feedback scores...")
    recent_scores = _collect_feedback_scores(client, args.project, recent_start, now)

    logger.info("Collecting baseline feedback scores...")
    baseline_scores = _collect_feedback_scores(client, args.project, baseline_start, now)

    recent_avgs = _compute_averages(recent_scores)
    baseline_avgs = _compute_averages(baseline_scores)

    # Report all monitored keys
    logger.info("─" * 60)
    logger.info("%-30s %8s %8s %8s", "Metric", "Baseline", "Recent", "Delta")
    logger.info("─" * 60)
    for key in MONITORED_KEYS:
        b_avg = f"{baseline_avgs[key][0]:.3f}" if key in baseline_avgs else "N/A"
        r_avg = f"{recent_avgs[key][0]:.3f}" if key in recent_avgs else "N/A"
        if key in baseline_avgs and key in recent_avgs:
            delta = recent_avgs[key][0] - baseline_avgs[key][0]
            d_str = f"{delta:+.3f}"
        else:
            d_str = "—"
        logger.info("%-30s %8s %8s %8s", key, b_avg, r_avg, d_str)
    logger.info("─" * 60)

    # Check for degradation
    findings = check_degradation(recent_avgs, baseline_avgs)

    if not findings:
        logger.info("No degradation detected")
        return 0

    logger.warning("DEGRADATION DETECTED in %d metric(s):", len(findings))
    for f in findings:
        logger.warning(
            "  %s: %.3f → %.3f (drop: %.3f, n=%d/%d)",
            f["key"],
            f["baseline_avg"],
            f["recent_avg"],
            f["drop"],
            f["baseline_n"],
            f["recent_n"],
        )

    if args.dry_run:
        logger.info("DRY RUN — not failing")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
