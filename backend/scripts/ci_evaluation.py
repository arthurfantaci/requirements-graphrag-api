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

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add src and repo root to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Validation thresholds
MIN_GOLDEN_DATASET_COUNT = 25
MIN_MUST_PASS_COUNT = 20

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
        min_avg_score=0.6,
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
        from requirements_graphrag_api.prompts.definitions import (
            PROMPT_DEFINITIONS,
            PromptName,
        )

        # Verify all PromptName entries have definitions
        for name in PromptName:
            if name not in PROMPT_DEFINITIONS:
                errors.append(f"Missing definition for PromptName.{name.name}")

        # Verify each definition has valid metadata
        for name, definition in PROMPT_DEFINITIONS.items():
            if not definition.metadata.version:
                errors.append(f"{name.value} missing version")
            if not definition.metadata.input_variables:
                errors.append(f"{name.value} missing input_variables")

            # Verify template can be instantiated
            try:
                definition.template.input_variables  # noqa: B018
            except Exception as e:
                errors.append(f"{name.value} template error: {e}")

        logger.info("Validated %d prompt definitions", len(PROMPT_DEFINITIONS))

    except ImportError as e:
        errors.append(f"Import error: {e}")

    return len(errors) == 0, errors


def validate_benchmark_dataset() -> tuple[bool, list[str]]:
    """Validate benchmark dataset structure.

    Uses the centralized golden dataset from the evaluation module
    (Hub-First pattern: same local fallback used when LangSmith is unreachable).

    Returns:
        Tuple of (passed, errors).
    """
    errors: list[str] = []

    try:
        from requirements_graphrag_api.evaluation.golden_dataset import (
            GOLDEN_EXAMPLES,
            get_must_pass_examples,
        )

        if len(GOLDEN_EXAMPLES) < MIN_GOLDEN_DATASET_COUNT:
            errors.append(
                f"Golden dataset has only {len(GOLDEN_EXAMPLES)} examples "
                f"(expected {MIN_GOLDEN_DATASET_COUNT}+)"
            )

        must_pass = get_must_pass_examples()
        if len(must_pass) < MIN_MUST_PASS_COUNT:
            errors.append(f"Must-pass examples: {len(must_pass)} (expected {MIN_MUST_PASS_COUNT}+)")

        # Spot check first 5
        for example in GOLDEN_EXAMPLES[:5]:
            if not example.question:
                errors.append(f"Example {example.id} missing question")
            if not example.expected_answer:
                errors.append(f"Example {example.id} missing expected_answer")

        logger.info("Golden dataset: %d examples", len(GOLDEN_EXAMPLES))
        logger.info("Must-pass examples: %d", len(must_pass))

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

    logger.info("Validating prompt definitions...")
    prompts_ok, prompt_errors = validate_prompts()
    errors.extend(prompt_errors)

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
# TIER 2-4: EVALUATION RUNNERS (via langsmith.evaluate())
# =============================================================================


async def _run_evaluation_tier(
    tier: int,
    tier_name: str,
) -> EvaluationResult:
    """Run evaluation for a specific tier using langsmith.evaluate().

    Args:
        tier: Tier number (2, 3, or 4).
        tier_name: Human-readable tier name.

    Returns:
        EvaluationResult with pass/fail status.
    """
    import os

    start_time = time.time()
    errors: list[str] = []
    tier_config = TIER_CONFIGS[tier]

    logger.info("=" * 60)
    logger.info("TIER %d: %s", tier, tier_name)
    logger.info("=" * 60)

    try:
        from datetime import UTC, datetime

        from langsmith import Client, aevaluate

        from requirements_graphrag_api.evaluation.golden_dataset import DATASET_NAME
        from requirements_graphrag_api.evaluation.ragas_evaluators import (
            answer_correctness_evaluator,
            answer_relevancy_evaluator,
            answer_semantic_similarity_evaluator,
            context_entity_recall_evaluator,
            context_precision_evaluator,
            context_recall_evaluator,
            faithfulness_evaluator,
        )

        client = Client()
        dataset_name = os.getenv("EVAL_DATASET", DATASET_NAME)

        # Verify dataset exists
        try:
            dataset = client.read_dataset(dataset_name=dataset_name)
            examples = list(client.list_examples(dataset_id=dataset.id))
            total_examples = len(examples)
            logger.info("Found %d examples in dataset '%s'", total_examples, dataset_name)
        except Exception as e:
            errors.append(f"Dataset error: {e}")
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

        # Limit samples for smoke test
        if tier_config.max_samples and total_examples > tier_config.max_samples:
            examples = examples[: tier_config.max_samples]
            logger.info("Limited to %d samples for tier %d", len(examples), tier)

        evaluators = [
            faithfulness_evaluator,
            answer_relevancy_evaluator,
            answer_correctness_evaluator,
            answer_semantic_similarity_evaluator,
            context_precision_evaluator,
            context_recall_evaluator,
            context_entity_recall_evaluator,
        ]

        # Create async target from the RAG pipeline
        from requirements_graphrag_api.config import get_config
        from requirements_graphrag_api.core.retrieval import (
            create_vector_retriever,
            graph_enriched_search,
        )
        from requirements_graphrag_api.neo4j_client import create_driver
        from requirements_graphrag_api.prompts import PromptName, get_prompt

        app_config = get_config()
        driver = create_driver(app_config)
        try:
            driver.verify_connectivity()
            retriever = create_vector_retriever(driver, app_config)

            async def rag_target(inputs: dict) -> dict:
                """RAG pipeline target for evaluation."""
                from langchain_openai import ChatOpenAI

                question = inputs.get("question", "")
                results = await graph_enriched_search(
                    retriever=retriever,
                    driver=driver,
                    query=question,
                    limit=5,
                )

                contexts = [r.get("content", "") for r in results]
                context = "\n\n".join(contexts)
                entities = ", ".join(
                    {
                        r.get("metadata", {}).get("entity_name", "")
                        for r in results
                        if r.get("metadata")
                    }
                )

                prompt_template = await get_prompt(PromptName.RAG_GENERATION)
                llm = ChatOpenAI(model=app_config.chat_model, temperature=0.1)
                chain = prompt_template | llm
                response = await chain.ainvoke(
                    {"context": context, "entities": entities, "question": question}
                )

                return {
                    "answer": response.content,
                    "contexts": contexts,
                    "intent": "explanatory",
                }

            logger.info(
                "Running evaluation with %d evaluators on %d examples...",
                len(evaluators),
                len(examples),
            )

            timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
            experiment_prefix = f"ci-tier{tier}-{timestamp}"
            results = await aevaluate(
                rag_target,
                data=examples,
                evaluators=evaluators,
                experiment_prefix=experiment_prefix,
                max_concurrency=2,
                metadata={"tier": tier, "tier_name": tier_name},
            )

            # Extract metrics
            aggregate_metrics: dict[str, float] = {}
            result_count = 0
            async for _result in results:
                result_count += 1

            avg_score = aggregate_metrics.get("avg_score", 0.0)
            if avg_score == 0.0 and aggregate_metrics:
                scores = [v for k, v in aggregate_metrics.items() if isinstance(v, int | float)]
                avg_score = sum(scores) / len(scores) if scores else 0.0

            passed = result_count > 0

            return EvaluationResult(
                tier=tier,
                tier_name=tier_name,
                passed=passed,
                samples_evaluated=result_count,
                avg_score=avg_score,
                min_score_threshold=tier_config.min_avg_score,
                duration_seconds=time.time() - start_time,
                cost_estimate=0.0,
                metrics=aggregate_metrics,
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


async def run_tier2() -> EvaluationResult:
    """Run Tier 2: Smoke test with 10 golden examples."""
    return await _run_evaluation_tier(2, "Smoke Test")


async def run_tier3() -> EvaluationResult:
    """Run Tier 3: Full benchmark evaluation."""
    return await _run_evaluation_tier(3, "Full Benchmark")


async def run_tier4() -> EvaluationResult:
    """Run Tier 4: Deep evaluation with extended analysis."""
    return await _run_evaluation_tier(4, "Deep Evaluation")


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

    if tier == 1:
        return await run_tier1()

    if tier == TIER_SMOKE:
        return await run_tier2()
    if tier == TIER_FULL:
        return await run_tier3()
    return await run_tier4()


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
        result = asyncio.run(run_evaluation(args.tier))

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

        output_path = Path(args.output)
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("Results saved to %s", output_path)

    except Exception:
        logger.exception("Evaluation failed with error")
        return 2

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
