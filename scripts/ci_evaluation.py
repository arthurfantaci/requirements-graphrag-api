#!/usr/bin/env python
"""CI-friendly evaluation runner with tiered support.

Provides different evaluation tiers for CI/CD integration:
- Tier 1: Prompt validation only (no API calls)
- Tier 2: Smoke evaluation (10 queries from golden dataset)
- Tier 3: Full benchmark evaluation (250+ queries)
- Tier 4: Deep evaluation with A/B comparison

Usage:
    # Tier 1 - Prompt validation (every PR)
    uv run python scripts/ci_evaluation.py --tier 1

    # Tier 2 - Smoke test (merge to main)
    uv run python scripts/ci_evaluation.py --tier 2

    # Tier 3 - Full benchmark (release)
    uv run python scripts/ci_evaluation.py --tier 3

    # Tier 4 - Deep eval (nightly)
    uv run python scripts/ci_evaluation.py --tier 4

Exit codes:
    0 - All evaluations passed
    1 - Evaluation failed (metrics below threshold)
    2 - Configuration or runtime error
"""
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

# Add src and repo root to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.config import AppConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Validation thresholds
MIN_KNOWN_STANDARDS_COUNT = 10
MIN_DOMAIN_TERMS_COUNT = 15
MIN_GOLDEN_DATASET_COUNT = 25
MIN_MUST_PASS_COUNT = 20
GENERATOR_TEST_COUNT = 5

# Tier identifiers
TIER_SMOKE = 2
TIER_FULL = 3


@dataclass(frozen=True)
class TierConfig:
    """Configuration for an evaluation tier."""

    name: str
    description: str
    max_samples: int | None
    cost_budget: float
    min_avg_score: float
    timeout_seconds: int
    requires_api: bool = True


TIER_CONFIGS: dict[int, TierConfig] = {
    1: TierConfig(
        name="prompt-validation",
        description="Validate prompts and configurations (no API calls)",
        max_samples=0,
        cost_budget=0.0,
        min_avg_score=0.0,
        timeout_seconds=60,
        requires_api=False,
    ),
    2: TierConfig(
        name="smoke-test",
        description="Quick smoke test with 10 golden examples",
        max_samples=10,
        cost_budget=0.50,
        min_avg_score=0.6,
        timeout_seconds=300,
        requires_api=True,
    ),
    3: TierConfig(
        name="full-benchmark",
        description="Full benchmark with 250+ examples",
        max_samples=None,  # All examples
        cost_budget=15.0,
        min_avg_score=0.7,
        timeout_seconds=1200,
        requires_api=True,
    ),
    4: TierConfig(
        name="deep-eval",
        description="Deep evaluation with A/B tests",
        max_samples=None,
        cost_budget=20.0,
        min_avg_score=0.7,
        timeout_seconds=2700,
        requires_api=True,
    ),
}


@dataclass
class EvaluationResult:
    """Result of a CI evaluation run."""

    tier: int
    tier_name: str
    passed: bool
    samples_evaluated: int
    avg_score: float
    min_score_threshold: float
    duration_seconds: float
    cost_estimate: float
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tier": self.tier,
            "tier_name": self.tier_name,
            "passed": self.passed,
            "samples_evaluated": self.samples_evaluated,
            "avg_score": self.avg_score,
            "min_score_threshold": self.min_score_threshold,
            "duration_seconds": self.duration_seconds,
            "cost_estimate": self.cost_estimate,
            "metrics": self.metrics,
            "errors": self.errors,
        }


# =============================================================================
# TIER 1: PROMPT VALIDATION
# =============================================================================


def validate_prompts() -> tuple[bool, list[str]]:
    """Validate that all prompts are properly configured.

    Returns:
        Tuple of (passed, errors).
    """
    errors: list[str] = []

    try:
        # Import evaluation prompts
        from jama_mcp_server_graphrag.evaluation.metrics import (
            ANSWER_RELEVANCY_PROMPT,
            CONTEXT_PRECISION_PROMPT,
            CONTEXT_RECALL_PROMPT,
            FAITHFULNESS_PROMPT,
        )

        # Validate RAG metric prompts have required placeholders
        rag_prompts = {
            "FAITHFULNESS_PROMPT": (FAITHFULNESS_PROMPT, ["context", "question", "answer"]),
            "ANSWER_RELEVANCY_PROMPT": (ANSWER_RELEVANCY_PROMPT, ["question", "answer"]),
            "CONTEXT_PRECISION_PROMPT": (CONTEXT_PRECISION_PROMPT, ["question", "contexts"]),
            "CONTEXT_RECALL_PROMPT": (
                CONTEXT_RECALL_PROMPT,
                ["question", "contexts", "ground_truth"],
            ),
        }

        for name, (prompt, required_vars) in rag_prompts.items():
            for var in required_vars:
                if f"{{{var}}}" not in prompt:
                    errors.append(f"{name} missing placeholder: {{{var}}}")

        # Import domain metric prompts
        from jama_mcp_server_graphrag.evaluation.domain_metrics import (
            CITATION_ACCURACY_PROMPT,
            COMPLETENESS_SCORE_PROMPT,
            DOMAIN_TERMS,
            KNOWN_STANDARDS,
            REGULATORY_ALIGNMENT_PROMPT,
            TECHNICAL_PRECISION_PROMPT,
            TRACEABILITY_COVERAGE_PROMPT,
        )

        domain_prompts = {
            "CITATION_ACCURACY_PROMPT": (
                CITATION_ACCURACY_PROMPT,
                ["expected_standards", "answer"],
            ),
            "TRACEABILITY_COVERAGE_PROMPT": (
                TRACEABILITY_COVERAGE_PROMPT,
                ["question", "answer", "expected_entities"],
            ),
            "TECHNICAL_PRECISION_PROMPT": (
                TECHNICAL_PRECISION_PROMPT,
                ["question", "answer"],
            ),
            "COMPLETENESS_SCORE_PROMPT": (
                COMPLETENESS_SCORE_PROMPT,
                ["question", "answer"],
            ),
            "REGULATORY_ALIGNMENT_PROMPT": (
                REGULATORY_ALIGNMENT_PROMPT,
                ["expected_standards", "question", "answer"],
            ),
        }

        for name, (prompt, required_vars) in domain_prompts.items():
            for var in required_vars:
                if f"{{{var}}}" not in prompt:
                    errors.append(f"{name} missing placeholder: {{{var}}}")

        # Validate known standards and domain terms exist
        if len(KNOWN_STANDARDS) < MIN_KNOWN_STANDARDS_COUNT:
            errors.append(
                f"KNOWN_STANDARDS has only {len(KNOWN_STANDARDS)} items "
                f"(expected {MIN_KNOWN_STANDARDS_COUNT}+)"
            )

        if len(DOMAIN_TERMS) < MIN_DOMAIN_TERMS_COUNT:
            errors.append(
                f"DOMAIN_TERMS has only {len(DOMAIN_TERMS)} items "
                f"(expected {MIN_DOMAIN_TERMS_COUNT}+)"
            )

        logger.info("Validated %d RAG prompts", len(rag_prompts))
        logger.info("Validated %d domain prompts", len(domain_prompts))
        logger.info("KNOWN_STANDARDS: %d items", len(KNOWN_STANDARDS))
        logger.info("DOMAIN_TERMS: %d items", len(DOMAIN_TERMS))

    except ImportError as e:
        errors.append(f"Import error: {e}")

    return len(errors) == 0, errors


def validate_benchmark_dataset() -> tuple[bool, list[str]]:
    """Validate benchmark dataset structure.

    Returns:
        Tuple of (passed, errors).
    """
    errors: list[str] = []

    try:
        from tests.benchmark.generator import generate_evaluation_dataset
        from tests.benchmark.golden_dataset import GOLDEN_DATASET, get_must_pass_examples

        # Validate golden dataset
        if len(GOLDEN_DATASET) < MIN_GOLDEN_DATASET_COUNT:
            errors.append(
                f"Golden dataset has only {len(GOLDEN_DATASET)} examples "
                f"(expected {MIN_GOLDEN_DATASET_COUNT}+)"
            )

        must_pass = get_must_pass_examples()
        if len(must_pass) < MIN_MUST_PASS_COUNT:
            errors.append(f"Must-pass examples: {len(must_pass)} (expected {MIN_MUST_PASS_COUNT}+)")

        # Validate each example has required fields
        for example in GOLDEN_DATASET[:5]:  # Spot check first 5
            if not example.question:
                errors.append(f"Example {example.id} missing question")
            if not example.ground_truth:
                errors.append(f"Example {example.id} missing ground_truth")

        # Validate generator works
        generated = generate_evaluation_dataset(total_examples=GENERATOR_TEST_COUNT)
        if len(generated) < GENERATOR_TEST_COUNT:
            errors.append(
                f"Generator produced only {len(generated)} examples "
                f"(expected {GENERATOR_TEST_COUNT})"
            )

        logger.info("Golden dataset: %d examples", len(GOLDEN_DATASET))
        logger.info("Must-pass examples: %d", len(must_pass))
        logger.info("Generator test: %d examples", len(generated))

    except ImportError as e:
        errors.append(f"Import error: {e}")

    return len(errors) == 0, errors


async def run_tier1() -> EvaluationResult:
    """Run Tier 1: Prompt validation (no API calls)."""
    start_time = time.time()
    errors: list[str] = []

    logger.info("=" * 60)
    logger.info("TIER 1: Prompt Validation")
    logger.info("=" * 60)

    # Validate prompts
    logger.info("Validating evaluation prompts...")
    prompts_ok, prompt_errors = validate_prompts()
    errors.extend(prompt_errors)

    # Validate benchmark dataset
    logger.info("Validating benchmark dataset...")
    dataset_ok, dataset_errors = validate_benchmark_dataset()
    errors.extend(dataset_errors)

    duration = time.time() - start_time
    passed = prompts_ok and dataset_ok

    return EvaluationResult(
        tier=1,
        tier_name="prompt-validation",
        passed=passed,
        samples_evaluated=0,
        avg_score=1.0 if passed else 0.0,
        min_score_threshold=0.0,
        duration_seconds=duration,
        cost_estimate=0.0,
        metrics={"prompts_valid": float(prompts_ok), "dataset_valid": float(dataset_ok)},
        errors=errors,
    )


# =============================================================================
# TIER 2-4: EVALUATION RUNNERS
# =============================================================================


def _calculate_avg_score(aggregate_metrics: dict[str, float]) -> float:
    """Calculate average score from aggregate metrics."""
    avg_score = aggregate_metrics.get("avg_score", 0.0)
    if avg_score == 0.0 and aggregate_metrics:
        scores = [v for k, v in aggregate_metrics.items() if isinstance(v, (int, float))]
        avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score


async def _run_evaluation_tier(
    config: AppConfig,
    tier: int,
    tier_name: str,
) -> EvaluationResult:
    """Run evaluation for a specific tier.

    Args:
        config: Application configuration.
        tier: Tier number (2, 3, or 4).
        tier_name: Human-readable tier name.

    Returns:
        EvaluationResult with pass/fail status.
    """
    start_time = time.time()
    errors: list[str] = []
    tier_config = TIER_CONFIGS[tier]

    logger.info("=" * 60)
    logger.info("TIER %d: %s", tier, tier_name)
    logger.info("=" * 60)

    try:
        from jama_mcp_server_graphrag.core.retrieval import create_vector_retriever
        from jama_mcp_server_graphrag.evaluation import evaluate_rag_pipeline
        from jama_mcp_server_graphrag.neo4j_client import create_driver
        from jama_mcp_server_graphrag.observability import configure_tracing

        configure_tracing(config)

        logger.info("Connecting to Neo4j...")
        driver = create_driver(config)

        try:
            driver.verify_connectivity()
            logger.info("Neo4j connection verified")

            retriever = create_vector_retriever(driver, config)

            logger.info("Running evaluation (max %s samples)...", tier_config.max_samples)
            report = await evaluate_rag_pipeline(
                config,
                retriever,
                driver,
                max_samples=tier_config.max_samples,
            )

            avg_score = _calculate_avg_score(report.aggregate_metrics)
            passed = avg_score >= tier_config.min_avg_score

            # Estimate cost based on tier
            cost_per_sample = {2: 0.05, 3: 0.06, 4: 0.08}.get(tier, 0.05)

            return EvaluationResult(
                tier=tier,
                tier_name=tier_name,
                passed=passed,
                samples_evaluated=report.total_samples,
                avg_score=avg_score,
                min_score_threshold=tier_config.min_avg_score,
                duration_seconds=time.time() - start_time,
                cost_estimate=report.total_samples * cost_per_sample,
                metrics=report.aggregate_metrics,
                errors=errors,
            )

        finally:
            driver.close()

    except Exception as e:
        logger.exception("Tier %d failed with error", tier)
        errors.append(str(e))
        return EvaluationResult(
            tier=tier,
            tier_name=tier_name,
            passed=False,
            samples_evaluated=0,
            avg_score=0.0,
            min_score_threshold=tier_config.min_avg_score,
            duration_seconds=time.time() - start_time,
            cost_estimate=0.0,
            errors=errors,
        )


async def run_tier2(config: AppConfig) -> EvaluationResult:
    """Run Tier 2: Smoke test with 10 golden examples."""
    return await _run_evaluation_tier(config, 2, "Smoke Test")


async def run_tier3(config: AppConfig) -> EvaluationResult:
    """Run Tier 3: Full benchmark evaluation."""
    return await _run_evaluation_tier(config, 3, "Full Benchmark")


async def run_tier4(config: AppConfig) -> EvaluationResult:
    """Run Tier 4: Deep evaluation with extended analysis."""
    return await _run_evaluation_tier(config, 4, "Deep Evaluation")


# =============================================================================
# MAIN
# =============================================================================


async def run_evaluation(tier: int) -> EvaluationResult:
    """Run evaluation for specified tier.

    Args:
        tier: Evaluation tier (1-4).

    Returns:
        EvaluationResult with pass/fail status.
    """
    if tier not in TIER_CONFIGS:
        raise ValueError(f"Invalid tier: {tier}. Must be 1-4.")

    tier_config = TIER_CONFIGS[tier]
    logger.info("Running %s evaluation...", tier_config.name)

    # Tier 1 doesn't need API config
    if tier == 1:
        return await run_tier1()

    # Tiers 2-4 need full config
    from jama_mcp_server_graphrag.config import get_config

    config = get_config()

    if tier == TIER_SMOKE:
        return await run_tier2(config)
    if tier == TIER_FULL:
        return await run_tier3(config)
    return await run_tier4(config)


def main() -> int:
    """Main entry point for CI evaluation.

    Returns:
        Exit code (0=pass, 1=fail, 2=error).
    """
    parser = argparse.ArgumentParser(
        description="CI-friendly evaluation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Evaluation tier (1=prompt validation, 2=smoke, 3=full, 4=deep)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ci_evaluation_results.json",
        help="Output file for JSON results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Run evaluation
        result = asyncio.run(run_evaluation(args.tier))

        # Output results
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULT: {'PASSED' if result.passed else 'FAILED'}")
        print("=" * 60)
        print(f"Tier: {result.tier} ({result.tier_name})")
        print(f"Samples Evaluated: {result.samples_evaluated}")
        print(f"Average Score: {result.avg_score:.4f}")
        print(f"Threshold: {result.min_score_threshold:.4f}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Est. Cost: ${result.cost_estimate:.2f}")

        if result.metrics:
            print("\nMetrics:")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")

        # Save results
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("Results saved to %s", output_path)

    except Exception:
        logger.exception("Evaluation failed with error")
        return 2

    # Return appropriate exit code
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
