#!/usr/bin/env python
"""Online evaluation for structured (Text2Cypher) traces.

Batch evaluator that fetches recent structured traces from LangSmith,
validates the generated Cypher (parse, safety, schema), executes valid
queries against Neo4j, and posts per-metric feedback to LangSmith.

Usage:
    # Evaluate last 24h of structured traces
    uv run python scripts/online_eval_cypher.py --project graphrag-api-prod

    # Evaluate last 48h, limit 100, dry run
    uv run python scripts/online_eval_cypher.py --project graphrag-api-prod \
        --hours-back 48 --limit 100 --dry-run

    # Show help
    uv run python scripts/online_eval_cypher.py --help

Exit codes:
    0 — all evaluated traces passed
    1 — one or more traces had failures
    2 — script error (missing env vars, connection failure, etc.)

Requires:
    LANGSMITH_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
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

# Feedback key prefix for online evaluation metrics
FEEDBACK_PREFIX = "online_cypher"


def fetch_structured_traces(
    client: Any,
    project: str,
    limit: int,
    hours_back: int,
) -> list[Any]:
    """Fetch recent structured-intent traces from LangSmith.

    Args:
        client: LangSmith Client instance.
        project: LangSmith project name.
        limit: Maximum number of runs to fetch.
        hours_back: How many hours back to search.

    Returns:
        List of LangSmith Run objects with intent=structured.
    """
    start_time = datetime.now(tz=UTC) - timedelta(hours=hours_back)

    runs = list(
        client.list_runs(
            project_name=project,
            start_time=start_time,
            run_type="chain",
            limit=limit,
            filter='has(metadata, "intent") and eq(metadata["intent"], "structured")',
        )
    )
    logger.info("Fetched %d structured traces (last %dh)", len(runs), hours_back)
    return runs


def evaluate_trace(run: Any, driver: Any) -> dict[str, dict[str, Any]]:
    """Evaluate a single structured trace.

    Runs 4 checks: parse validity, safety, schema adherence, execution.

    Args:
        run: LangSmith Run object.
        driver: Neo4j driver instance.

    Returns:
        Dict mapping metric key to {score, comment}.
    """
    from requirements_graphrag_api.evaluation.constants import validate_cypher_comprehensive
    from requirements_graphrag_api.evaluation.structured_evaluators import (
        _check_parse_validity,
        _check_schema_adherence,
    )

    outputs = run.outputs or {}
    cypher = outputs.get("cypher", "") or outputs.get("output", "")

    results: dict[str, dict[str, Any]] = {}

    # 1. Parse validity
    parse_score, parse_comment = _check_parse_validity(cypher)
    results[f"{FEEDBACK_PREFIX}_parse"] = {"score": parse_score, "comment": parse_comment}

    # 2. Safety (comprehensive Tier 1-3 check)
    safety_error = validate_cypher_comprehensive(cypher)
    if safety_error:
        results[f"{FEEDBACK_PREFIX}_safety"] = {"score": 0, "comment": safety_error}
    else:
        results[f"{FEEDBACK_PREFIX}_safety"] = {"score": 1, "comment": "Safe read-only query"}

    # 3. Schema adherence
    schema_score, schema_comment = _check_schema_adherence(cypher)
    results[f"{FEEDBACK_PREFIX}_schema"] = {"score": schema_score, "comment": schema_comment}

    # 4. Execution (only attempt if parse + safety passed)
    if parse_score == 1 and safety_error is None:
        try:
            records, _, _ = driver.execute_query(cypher)
            row_count = len(records)
            results[f"{FEEDBACK_PREFIX}_execution"] = {
                "score": 1,
                "comment": f"Executed OK, {row_count} rows",
            }
        except Exception as e:
            results[f"{FEEDBACK_PREFIX}_execution"] = {
                "score": 0,
                "comment": f"Execution error: {str(e)[:200]}",
            }
    else:
        skip_reason = parse_comment if parse_score == 0 else safety_error
        results[f"{FEEDBACK_PREFIX}_execution"] = {
            "score": 0,
            "comment": f"Skipped — {skip_reason}",
        }

    return results


def post_feedback(
    client: Any,
    run_id: str,
    results: dict[str, dict[str, Any]],
) -> int:
    """Post evaluation results as feedback to LangSmith.

    Args:
        client: LangSmith Client instance.
        run_id: Run ID to attach feedback to.
        results: Dict mapping metric key to {score, comment}.

    Returns:
        Number of feedback items successfully posted.
    """
    posted = 0
    for key, data in results.items():
        try:
            client.create_feedback(
                run_id=run_id,
                key=key,
                score=data["score"],
                comment=data.get("comment"),
            )
            posted += 1
        except Exception:
            logger.warning("Failed to post feedback '%s' for run %s", key, run_id, exc_info=True)
    return posted


def main() -> int:
    """Main entry point."""
    # Check required env vars
    missing = [v for v in ("LANGSMITH_API_KEY",) if not os.getenv(v)]
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        return 2

    parser = argparse.ArgumentParser(
        description="Online evaluation for structured (Text2Cypher) traces",
    )
    parser.add_argument(
        "--project",
        "-p",
        default="graphrag-api-prod",
        help="LangSmith project name (default: graphrag-api-prod)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=50,
        help="Maximum number of traces to evaluate (default: 50)",
    )
    parser.add_argument(
        "--hours-back",
        type=int,
        default=24,
        help="How many hours back to search (default: 24)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate but don't post feedback to LangSmith",
    )

    args = parser.parse_args()

    try:
        from langsmith import Client

        client = Client()
    except ImportError:
        logger.error("langsmith package not installed")
        return 2

    # Fetch traces
    runs = fetch_structured_traces(client, args.project, args.limit, args.hours_back)
    if not runs:
        logger.info("No structured traces found — nothing to evaluate")
        return 0

    # Set up Neo4j driver
    from requirements_graphrag_api.config import get_config
    from requirements_graphrag_api.neo4j_client import create_driver

    config = get_config()
    driver = create_driver(config)

    total_failures = 0
    total_posted = 0

    try:
        for i, run in enumerate(runs, 1):
            run_id = str(run.id)
            logger.info("[%d/%d] Evaluating run %s", i, len(runs), run_id)

            results = evaluate_trace(run, driver)

            # Count failures (any metric with score < 1)
            failures = sum(1 for d in results.values() if d["score"] < 1)
            total_failures += failures

            # Log results
            for key, data in results.items():
                status = "PASS" if data["score"] >= 1 else "FAIL"
                logger.info("  %s: %s (%.1f) — %s", key, status, data["score"], data["comment"])

            # Post feedback
            if not args.dry_run:
                posted = post_feedback(client, run_id, results)
                total_posted += posted

        # Summary
        logger.info("─" * 60)
        logger.info("Evaluated %d traces", len(runs))
        logger.info("Total metric failures: %d", total_failures)
        if not args.dry_run:
            logger.info("Feedback items posted: %d", total_posted)
        else:
            logger.info("DRY RUN — no feedback posted")
        logger.info("─" * 60)

        return 1 if total_failures > 0 else 0

    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
