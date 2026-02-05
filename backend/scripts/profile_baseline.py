#!/usr/bin/env python
"""Performance baseline profiling for Phase A.

Creates a LangSmith dataset 'phase-a-baseline' with 8 test queries and
runs each query 3x against the live API, capturing:
- TTFT (Time to First Token)
- Total latency
- LLM call count (from trace spans)
- Token cost

Tags each run with {"phase": "baseline"} or {"phase": "post-phase-a"}.

Usage:
    # Create dataset + run baseline
    uv run python scripts/profile_baseline.py

    # Run post-optimization comparison
    uv run python scripts/profile_baseline.py --phase post-phase-a

    # Dry run (create dataset only)
    uv run python scripts/profile_baseline.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_NAME = "phase-a-baseline"

BASELINE_QUERIES: list[dict[str, Any]] = [
    {
        "query": "What is requirements traceability?",
        "expected_path": "explanatory",
        "description": "Simple explanatory, no history",
    },
    {
        "query": "How do I implement change management for ISO 26262?",
        "expected_path": "explanatory",
        "description": "Explanatory, triggers research",
    },
    {
        "query": "List all webinars",
        "expected_path": "structured",
        "description": "Structured (keyword match)",
    },
    {
        "query": "Which standards apply to automotive?",
        "expected_path": "structured",
        "description": "Structured (pattern match)",
    },
    {
        "query": "What is the relationship between verification and validation?",
        "expected_path": "explanatory",
        "description": "Explanatory, borderline topic",
    },
    {
        "query": "What is the best way to manage requirements in an agile environment?",
        "expected_path": "explanatory",
        "description": "Explanatory, multi-concept",
    },
    {
        "query": "How does Jama Connect implement it?",
        "expected_path": "explanatory",
        "description": "Follow-up with conversation_history",
        "conversation_history": [
            {"role": "user", "content": "What is requirements traceability?"},
            {
                "role": "assistant",
                "content": (
                    "Requirements traceability is the ability to trace requirements "
                    "throughout the product lifecycle, connecting them to their source, "
                    "implementation, and verification."
                ),
            },
        ],
    },
    {
        "query": "Compare DOORS and Jama Connect for traceability",
        "expected_path": "explanatory",
        "description": "Explanatory, multiple entities",
    },
]


def create_langsmith_dataset() -> str:
    """Create or update the LangSmith baseline dataset."""
    from langsmith import Client

    client = Client()

    # Check if dataset exists
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        logger.info("Dataset '%s' already exists (id=%s)", DATASET_NAME, dataset.id)
        return str(dataset.id)
    except Exception:
        logger.info("Dataset '%s' not found, creating new one", DATASET_NAME)

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Phase A performance baseline - 8 queries covering all paths",
    )
    logger.info("Created dataset '%s' (id=%s)", DATASET_NAME, dataset.id)

    for i, q in enumerate(BASELINE_QUERIES):
        inputs = {"query": q["query"]}
        if q.get("conversation_history"):
            inputs["conversation_history"] = q["conversation_history"]

        client.create_example(
            inputs=inputs,
            outputs={"expected_path": q["expected_path"]},
            dataset_id=dataset.id,
            metadata={
                "description": q["description"],
                "index": i,
            },
        )
        logger.info("  Added example %d: %s", i, q["query"][:50])

    return str(dataset.id)


async def run_query(
    client: httpx.AsyncClient,
    base_url: str,
    query: dict[str, Any],
    run_index: int,
    phase: str,
) -> dict[str, Any]:
    """Run a single query against the API and collect metrics."""
    payload: dict[str, Any] = {
        "message": query["query"],
        "options": {"auto_route": True},
    }
    if query.get("conversation_history"):
        payload["conversation_history"] = query["conversation_history"]

    start = time.perf_counter()
    ttft = None
    total_tokens = 0
    events: list[dict[str, Any]] = []
    full_answer = ""

    try:
        async with client.stream(
            "POST",
            f"{base_url}/chat",
            json=payload,
            timeout=60.0,
        ) as response:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                if line.startswith("event: "):
                    event_type = line[7:].strip()
                    continue

                if line.startswith("data: "):
                    elapsed = time.perf_counter() - start
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if "token" in data and ttft is None:
                        ttft = elapsed
                    if "token" in data:
                        total_tokens += 1
                        full_answer += data.get("token", "")
                    if "full_answer" in data:
                        full_answer = data["full_answer"]

                    events.append({"type": event_type, "elapsed": elapsed, "data": data})

    except httpx.ReadTimeout:
        logger.warning("Timeout for query: %s", query["query"][:40])
        return {
            "query": query["query"],
            "run_index": run_index,
            "error": "timeout",
            "phase": phase,
        }

    total_latency = time.perf_counter() - start

    return {
        "query": query["query"],
        "description": query["description"],
        "run_index": run_index,
        "phase": phase,
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_latency_s": round(total_latency, 3),
        "token_count": total_tokens,
        "answer_length": len(full_answer),
        "event_count": len(events),
    }


async def run_profiling(
    base_url: str,
    phase: str,
    runs_per_query: int = 3,
) -> list[dict[str, Any]]:
    """Run all baseline queries and collect metrics."""
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        for qi, query in enumerate(BASELINE_QUERIES):
            for run_idx in range(runs_per_query):
                logger.info(
                    "[%d/%d] Run %d/3: %s",
                    qi + 1,
                    len(BASELINE_QUERIES),
                    run_idx + 1,
                    query["query"][:50],
                )
                result = await run_query(client, base_url, query, run_idx, phase)
                results.append(result)
                logger.info(
                    "  TTFT=%.2fs  Total=%.2fs  Tokens=%d",
                    result.get("ttft_s") or 0,
                    result.get("total_latency_s", 0),
                    result.get("token_count", 0),
                )

    return results


def print_summary(results: list[dict[str, Any]], phase: str) -> None:
    """Print a summary table of profiling results."""
    print(f"\n{'=' * 80}")
    print(f"Phase: {phase}")
    print(f"{'=' * 80}")
    print(f"{'Query':<55} {'TTFT':>8} {'Total':>8} {'Tokens':>8}")
    print("-" * 80)

    # Group by query
    queries: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        key = r["query"][:52]
        queries.setdefault(key, []).append(r)

    for query_key, runs in queries.items():
        ttfts = [r["ttft_s"] for r in runs if r.get("ttft_s")]
        latencies = [r["total_latency_s"] for r in runs if r.get("total_latency_s")]
        tokens = [r["token_count"] for r in runs if r.get("token_count")]

        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_tokens = sum(tokens) / len(tokens) if tokens else 0

        print(f"{query_key:<55} {avg_ttft:>7.2f}s {avg_latency:>7.2f}s {avg_tokens:>7.0f}")

    print(f"{'=' * 80}")

    # Overall averages
    all_ttfts = [r["ttft_s"] for r in results if r.get("ttft_s")]
    all_latencies = [r["total_latency_s"] for r in results if r.get("total_latency_s")]
    if all_ttfts:
        print(f"Overall avg TTFT: {sum(all_ttfts) / len(all_ttfts):.2f}s")
    if all_latencies:
        print(f"Overall avg Total: {sum(all_latencies) / len(all_latencies):.2f}s")


def main() -> None:
    """Run the baseline profiling CLI."""
    parser = argparse.ArgumentParser(description="Profile baseline performance")
    parser.add_argument(
        "--phase",
        default="baseline",
        choices=["baseline", "post-phase-a"],
        help="Phase tag for LangSmith traces",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="API base URL",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per query")
    parser.add_argument("--dry-run", action="store_true", help="Only create dataset")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    # Create/verify dataset
    logger.info("Creating LangSmith dataset...")
    dataset_id = create_langsmith_dataset()
    logger.info("Dataset ready: %s", dataset_id)

    if args.dry_run:
        logger.info("Dry run complete. Dataset created with %d queries.", len(BASELINE_QUERIES))
        return

    # Run profiling
    logger.info("Running profiling (phase=%s, runs=%d)...", args.phase, args.runs)
    results = asyncio.run(run_profiling(args.base_url, args.phase, args.runs))

    # Print summary
    print_summary(results, args.phase)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / f"profile_results_{args.phase}.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
