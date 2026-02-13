#!/usr/bin/env python
"""CI-friendly evaluation runner with tiered, per-vector support.

Provides per-vector evaluation tiers for CI/CD integration:
- Tier 1: Prompt + dataset validation (no API calls, $0)
- Tier 2: Smoke test per vector (manual, ~$0.50)
- Tier 3: Full benchmark per vector (release tags, ~$15)
- Tier 4: Deep evaluation per vector (manual, ~$20)

Usage:
    # Tier 1 - Prompt validation (every PR)
    uv run python scripts/ci_evaluation.py --tier 1

    # Tier 2 - Smoke test (manual)
    uv run python scripts/ci_evaluation.py --tier 2

    # Tier 3 - Full benchmark (release)
    uv run python scripts/ci_evaluation.py --tier 3 --regression-gate

    # Tier 4 - Deep eval (manual)
    uv run python scripts/ci_evaluation.py --tier 4 --regression-gate

    # Single vector only
    uv run python scripts/ci_evaluation.py --tier 3 --vector explanatory

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
from pathlib import Path

# Add src and repo root to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

# Add scripts/ dir for sibling imports (run_vector_evaluation)
sys.path.insert(0, str(Path(__file__).parent))

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

# Per-tier vector configs: {vector: max_samples | None (= all examples)}
TIER_VECTOR_CONFIGS: dict[int, dict[str, int | None]] = {
    2: {"intent": None, "explanatory": 10, "structured": 10},
    3: {"intent": None, "explanatory": None, "structured": None, "conversational": None},
    4: {"intent": None, "explanatory": None, "structured": None, "conversational": None},
}


# =============================================================================
# TIER 1: PROMPT + DATASET VALIDATION
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

        for name in PromptName:
            if name not in PROMPT_DEFINITIONS:
                errors.append(f"Missing definition for PromptName.{name.name}")

        for name, definition in PROMPT_DEFINITIONS.items():
            if not definition.metadata.version:
                errors.append(f"{name.value} missing version")
            if not definition.metadata.input_variables:
                errors.append(f"{name.value} missing input_variables")

            try:
                definition.template.input_variables  # noqa: B018
            except Exception as e:
                errors.append(f"{name.value} template error: {e}")

        logger.info("Validated %d prompt definitions", len(PROMPT_DEFINITIONS))

    except ImportError as e:
        errors.append(f"Import error: {e}")

    return len(errors) == 0, errors


def validate_benchmark_dataset() -> tuple[bool, list[str]]:
    """Validate per-vector dataset constants and local fallback data.

    Returns:
        Tuple of (passed, errors).
    """
    errors: list[str] = []

    try:
        from requirements_graphrag_api.evaluation.constants import (
            ALL_VECTOR_DATASETS,
            DATASET_CONVERSATIONAL,
            DATASET_EXPLANATORY,
            DATASET_INTENT,
            DATASET_STRUCTURED,
        )

        for label, value in [
            ("DATASET_EXPLANATORY", DATASET_EXPLANATORY),
            ("DATASET_STRUCTURED", DATASET_STRUCTURED),
            ("DATASET_CONVERSATIONAL", DATASET_CONVERSATIONAL),
            ("DATASET_INTENT", DATASET_INTENT),
        ]:
            if not value:
                errors.append(f"{label} is empty")

        if len(ALL_VECTOR_DATASETS) != 4:
            errors.append(
                f"ALL_VECTOR_DATASETS has {len(ALL_VECTOR_DATASETS)} entries (expected 4)"
            )

        logger.info("Validated %d vector dataset constants", len(ALL_VECTOR_DATASETS))

    except ImportError as e:
        errors.append(f"Import error: {e}")

    # Validate local fallback golden examples
    try:
        from requirements_graphrag_api.evaluation.golden_dataset import (
            GOLDEN_EXAMPLES,
            get_must_pass_examples,
        )

        if len(GOLDEN_EXAMPLES) < 25:
            errors.append(f"Golden dataset has only {len(GOLDEN_EXAMPLES)} examples (expected 25+)")

        must_pass = get_must_pass_examples()
        if len(must_pass) < 20:
            errors.append(f"Must-pass examples: {len(must_pass)} (expected 20+)")

        logger.info(
            "Golden dataset: %d examples (%d must-pass)",
            len(GOLDEN_EXAMPLES),
            len(must_pass),
        )

    except ImportError as e:
        errors.append(f"Import error: {e}")

    return len(errors) == 0, errors


async def run_tier1() -> dict:
    """Run Tier 1: Prompt + dataset validation (no API calls)."""
    start_time = time.time()
    errors: list[str] = []

    logger.info("=" * 60)
    logger.info("TIER 1: Prompt & Dataset Validation")
    logger.info("=" * 60)

    logger.info("Validating prompt definitions...")
    prompts_ok, prompt_errors = validate_prompts()
    errors.extend(prompt_errors)

    logger.info("Validating benchmark datasets...")
    dataset_ok, dataset_errors = validate_benchmark_dataset()
    errors.extend(dataset_errors)

    passed = prompts_ok and dataset_ok

    return {
        "tier": 1,
        "passed": passed,
        "vectors": {},
        "regression": None,
        "duration_seconds": round(time.time() - start_time, 1),
        "errors": errors,
    }


# =============================================================================
# TIER 2-4: PER-VECTOR EVALUATION
# =============================================================================


async def _run_vector_ci(
    vector: str,
    max_samples: int | None,
    prompt_tag: str = "production",
) -> dict:
    """Run evaluation for a single vector.

    Args:
        vector: Vector name (intent, explanatory, structured, conversational).
        max_samples: Max examples to evaluate (None = all).
        prompt_tag: Hub prompt tag.

    Returns:
        Dict with vector, scores, experiment, sample_count.
    """
    from langsmith import Client, aevaluate
    from run_vector_evaluation import (
        _create_conversational_target,
        _create_explanatory_target,
        _create_intent_target,
        _create_structured_target,
        _get_dataset_name,
        _get_evaluators,
    )

    from requirements_graphrag_api.evaluation.constants import experiment_name

    dataset_name = _get_dataset_name(vector)
    evaluators = _get_evaluators(vector)
    exp_name = experiment_name(vector, prompt_tag)

    # Resolve data: slice if max_samples, otherwise pass dataset name
    if max_samples is not None:
        client = Client()
        dataset = client.read_dataset(dataset_name=dataset_name)
        all_examples = list(client.list_examples(dataset_id=dataset.id))
        data = all_examples[:max_samples]
        logger.info(
            "Vector %s: %d/%d examples (capped)",
            vector,
            len(data),
            len(all_examples),
        )
    else:
        data = dataset_name

    # Create target
    target_creators = {
        "explanatory": _create_explanatory_target,
        "structured": _create_structured_target,
        "conversational": _create_conversational_target,
        "intent": _create_intent_target,
    }
    target_fn, driver = await target_creators[vector](prompt_tag)

    try:
        logger.info(
            "Running %s: %d evaluators, dataset=%s, tag=%s",
            vector,
            len(evaluators),
            dataset_name,
            prompt_tag,
        )

        results = await aevaluate(
            target_fn,
            data=data,
            evaluators=evaluators,
            experiment_prefix=exp_name,
            max_concurrency=2,
            metadata={"vector": vector, "prompt_tag": prompt_tag, "ci": True},
        )

        # Collect scores
        metric_scores: dict[str, list[float]] = {}
        result_count = 0
        async for result in results:
            result_count += 1
            eval_results = result.get("evaluation_results", {})
            for er in eval_results.get("results", []):
                if er.score is not None:
                    metric_scores.setdefault(er.key, []).append(er.score)

        avg_scores = {
            metric: round(sum(scores) / len(scores), 4)
            for metric, scores in metric_scores.items()
            if scores
        }

        logger.info(
            "Vector %s complete: %d samples, scores=%s",
            vector,
            result_count,
            avg_scores,
        )

        return {
            "vector": vector,
            "scores": avg_scores,
            "experiment": exp_name,
            "sample_count": result_count,
        }

    finally:
        if driver is not None:
            driver.close()


async def _run_tiers_2_4(
    tier: int,
    vector_filter: str | None = None,
    regression_gate: bool = False,
    prompt_tag: str = "production",
) -> dict:
    """Run per-vector evaluation for tiers 2-4.

    Args:
        tier: Tier number (2, 3, or 4).
        vector_filter: Optional single vector to run.
        regression_gate: Whether to check regression thresholds.
        prompt_tag: Hub prompt tag.

    Returns:
        Result dict with per-vector scores and regression status.
    """
    start_time = time.time()
    errors: list[str] = []

    logger.info("=" * 60)
    logger.info("TIER %d: Per-Vector Evaluation", tier)
    logger.info("=" * 60)

    tier_vectors = TIER_VECTOR_CONFIGS[tier]

    # Filter to single vector if specified
    if vector_filter:
        if vector_filter not in tier_vectors:
            return {
                "tier": tier,
                "passed": False,
                "vectors": {},
                "regression": None,
                "duration_seconds": round(time.time() - start_time, 1),
                "errors": [f"Vector '{vector_filter}' not in tier {tier} config"],
            }
        tier_vectors = {vector_filter: tier_vectors[vector_filter]}

    # Run each vector sequentially
    vector_results: dict[str, dict] = {}
    for vector, max_samples in tier_vectors.items():
        try:
            result = await _run_vector_ci(vector, max_samples, prompt_tag)
            vector_results[vector] = result
        except Exception as e:
            logger.exception("Vector %s failed", vector)
            errors.append(f"{vector}: {e}")

    # Regression gate
    regression_output = None
    passed = len(vector_results) > 0 and len(errors) == 0

    if regression_gate and vector_results:
        from requirements_graphrag_api.evaluation.regression import check_all_vectors

        gate_input = {v: r["scores"] for v, r in vector_results.items()}
        reports = check_all_vectors(gate_input)

        gate_passed = all(r.passed for r in reports.values())
        regression_output = {
            "passed": gate_passed,
            "details": {v: r.summary() for v, r in reports.items()},
        }

        for _v, report in reports.items():
            logger.info(report.summary())

        passed = passed and gate_passed

    return {
        "tier": tier,
        "passed": passed,
        "vectors": {
            v: {
                "scores": r["scores"],
                "experiment": r["experiment"],
                "sample_count": r["sample_count"],
            }
            for v, r in vector_results.items()
        },
        "regression": regression_output,
        "duration_seconds": round(time.time() - start_time, 1),
        "errors": errors,
    }


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Main entry point for CI evaluation.

    Returns:
        Exit code (0=pass, 1=fail, 2=error).
    """
    parser = argparse.ArgumentParser(
        description="CI-friendly per-vector evaluation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        help="Evaluation tier (1=validation, 2=smoke, 3=full, 4=deep)",
    )
    parser.add_argument(
        "--vector",
        choices=["intent", "explanatory", "structured", "conversational"],
        help="Run only this vector (default: all vectors for the tier)",
    )
    parser.add_argument(
        "--regression-gate",
        action="store_true",
        help="Check results against regression thresholds (exit 1 on failure)",
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
        if args.tier == 1:
            result = asyncio.run(run_tier1())
        else:
            result = asyncio.run(
                _run_tiers_2_4(
                    tier=args.tier,
                    vector_filter=args.vector,
                    regression_gate=args.regression_gate,
                )
            )

        # Print summary
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULT: {'PASSED' if result['passed'] else 'FAILED'}")
        print("=" * 60)
        print(f"Tier: {result['tier']}")
        print(f"Duration: {result['duration_seconds']}s")

        if result["vectors"]:
            print("\nPer-Vector Results:")
            for vector, vr in result["vectors"].items():
                print(f"\n  {vector} ({vr['sample_count']} samples):")
                for metric, score in sorted(vr["scores"].items()):
                    print(f"    {metric}: {score:.4f}")

        if result.get("regression"):
            reg = result["regression"]
            status = "PASSED" if reg["passed"] else "FAILED"
            print(f"\nRegression Gate: {status}")

        if result.get("errors"):
            print("\nErrors:")
            for error in result["errors"]:
                print(f"  - {error}")

        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2))
        logger.info("Results saved to %s", output_path)

    except Exception:
        logger.exception("Evaluation failed with error")
        return 2

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
