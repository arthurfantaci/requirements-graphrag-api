#!/usr/bin/env python
"""Verify online evaluators by sending test queries and checking for feedback.

Sends queries designed to trigger each evaluator, waits for async scoring,
then checks LangSmith for the expected feedback keys.

Usage:
    # Against production
    uv run python scripts/test_online_evaluators.py

    # Against local dev
    uv run python scripts/test_online_evaluators.py --api-url http://localhost:8000

    # Skip the wait (just send queries, check later manually)
    uv run python scripts/test_online_evaluators.py --no-wait

Requires:
    LANGSMITH_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API_URL = os.getenv("TEST_API_URL", "https://graphrag-api.norfolkaibi.com")
API_KEY = os.getenv("GRAPHRAG_API_KEY", "")

# Test queries designed to trigger specific intents.
# We use force_intent to guarantee routing (bypasses classifier sampling variance).
TEST_QUERIES = [
    {
        "label": "explanatory",
        "message": "What is requirements traceability and why is it important?",
        "options": {"force_intent": "explanatory"},
        "expected_evaluators": ["hallucination", "answer_relevancy"],
    },
    {
        "label": "structured",
        "message": "List all webinars about requirements management",
        "options": {"force_intent": "structured"},
        "expected_evaluators": ["online_cypher_parse", "answer_relevancy"],
    },
    {
        "label": "conversational",
        "message": "Can you summarize what we just discussed?",
        "options": {"force_intent": "conversational"},
        "conversation_history": [
            {"role": "user", "content": "What is traceability?"},
            {
                "role": "assistant",
                "content": (
                    "Requirements traceability is the ability to link"
                    " requirements to their origins and track them"
                    " throughout the lifecycle."
                ),
            },
        ],
        "expected_evaluators": ["coherence", "answer_relevancy"],
    },
]


def send_query(
    client: httpx.Client,
    api_url: str,
    query: dict,
) -> dict | None:
    """Send a chat query and parse the SSE response.

    Returns dict with intent, run_id, trace_id, and a snippet of the response.
    """
    payload: dict = {
        "message": query["message"],
    }
    if query.get("options"):
        payload["options"] = query["options"]
    if query.get("conversation_history"):
        payload["conversation_history"] = query["conversation_history"]

    logger.info("Sending [%s]: %s", query["label"], query["message"])

    try:
        with client.stream(
            "POST",
            f"{api_url}/chat",
            json=payload,
            timeout=60.0,
        ) as response:
            response.raise_for_status()

            result = {
                "label": query["label"],
                "intent": None,
                "run_id": None,
                "trace_id": None,
                "response_snippet": None,
                "expected_evaluators": query["expected_evaluators"],
            }

            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if data.get("intent"):
                    result["intent"] = data["intent"]
                if data.get("run_id"):
                    result["run_id"] = data["run_id"]
                if data.get("trace_id"):
                    result["trace_id"] = data["trace_id"]
                if data.get("full_answer"):
                    result["response_snippet"] = data["full_answer"][:100]

            return result

    except httpx.HTTPStatusError as e:
        logger.error("HTTP error for [%s]: %s", query["label"], e)
        return None
    except Exception:
        logger.exception("Failed to send query [%s]", query["label"])
        return None


def check_evaluator_feedback(
    run_id: str,
    expected_keys: list[str],
) -> dict[str, float | None]:
    """Check LangSmith for evaluator feedback on a run.

    Returns dict mapping feedback key to score (or None if not found).
    """
    from langsmith import Client

    client = Client()
    found: dict[str, float | None] = dict.fromkeys(expected_keys)

    try:
        feedbacks = list(client.list_feedback(run_ids=[run_id]))
        for fb in feedbacks:
            if fb.key in expected_keys:
                found[fb.key] = fb.score
    except Exception:
        logger.warning("Failed to fetch feedback for run %s", run_id, exc_info=True)

    return found


def main() -> int:
    """Send test queries and verify online evaluator feedback."""
    parser = argparse.ArgumentParser(description="Test online evaluators")
    parser.add_argument(
        "--api-url",
        default=API_URL,
        help=f"Backend API URL (default: {API_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=API_KEY,
        help="API key for authentication (default: GRAPHRAG_API_KEY env var)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Send queries without waiting for evaluator results",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=90,
        help="Seconds to wait for evaluators to run (default: 90)",
    )
    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("LANGSMITH_API_KEY not set")
        return 2

    # Phase 1: Send test queries
    logger.info("=" * 60)
    logger.info("Phase 1: Sending test queries to %s", args.api_url)
    logger.info("=" * 60)

    results = []
    headers = {}
    api_key = args.api_key
    if api_key:
        headers["X-API-Key"] = api_key
        logger.info("Using API key for authentication")
    else:
        logger.warning("No GRAPHRAG_API_KEY set — requests may fail with 401")

    with httpx.Client(headers=headers) as client:
        for query in TEST_QUERIES:
            result = send_query(client, args.api_url, query)
            if result:
                results.append(result)
                logger.info(
                    "  [%s] intent=%s run_id=%s",
                    result["label"],
                    result["intent"],
                    result["run_id"] or "MISSING",
                )
            else:
                logger.error("  [%s] FAILED", query["label"])

    if not results:
        logger.error("No queries succeeded")
        return 1

    # Phase 2: Check for evaluator feedback
    if args.no_wait:
        logger.info("Skipping evaluator check (--no-wait)")
        logger.info("Run IDs to check manually:")
        for r in results:
            logger.info("  [%s] %s", r["label"], r["run_id"])
        return 0

    logger.info("=" * 60)
    logger.info(
        "Phase 2: Waiting %ds for evaluators to score traces...",
        args.wait_seconds,
    )
    logger.info("=" * 60)
    time.sleep(args.wait_seconds)

    # Check feedback
    all_passed = True
    for result in results:
        if not result["run_id"]:
            logger.warning("  [%s] No run_id — skipping", result["label"])
            continue

        feedback = check_evaluator_feedback(
            result["run_id"],
            result["expected_evaluators"],
        )

        for key, score in feedback.items():
            if score is not None:
                logger.info(
                    "  [%s] %s = %.2f",
                    result["label"],
                    key,
                    score,
                )
            else:
                logger.warning(
                    "  [%s] %s = NOT FOUND (evaluator may not have sampled this trace)",
                    result["label"],
                    key,
                )

        # online_cypher_parse runs at 100%, so it MUST appear
        if "online_cypher_parse" in feedback and feedback["online_cypher_parse"] is None:
            logger.error("  [structured] online_cypher_parse MISSING — this runs at 100%%!")
            all_passed = False

    if all_passed:
        logger.info("=" * 60)
        logger.info("All expected evaluator feedback found (or acceptably sampled out)")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("Some evaluators did not produce expected feedback")
        return 1


if __name__ == "__main__":
    sys.exit(main())
