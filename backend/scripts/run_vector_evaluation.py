#!/usr/bin/env python
"""Unified per-vector evaluation runner.

Runs evaluations for a specific vector (explanatory, structured,
conversational, intent, or all) against the corresponding LangSmith
dataset with vector-appropriate evaluators.

Usage:
    # Run explanatory vector evaluation
    uv run python scripts/run_vector_evaluation.py --vector explanatory

    # Run all vectors
    uv run python scripts/run_vector_evaluation.py --vector all

    # Use staging-tagged prompts for A/B testing
    uv run python scripts/run_vector_evaluation.py --vector explanatory \
        --prompt-tag staging

    # Dry run
    uv run python scripts/run_vector_evaluation.py --vector all --dry-run

Requires:
- LANGSMITH_API_KEY, OPENAI_API_KEY
- NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
- Per-vector datasets created (run migrate_golden_datasets.py first)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
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

VECTORS = ("explanatory", "structured", "conversational", "intent")


# =============================================================================
# EVALUATOR SETS PER VECTOR
# =============================================================================


def _get_evaluators(vector: str) -> list:
    """Return the evaluator list for a given vector."""
    if vector == "explanatory":
        from requirements_graphrag_api.evaluation.ragas_evaluators import (
            answer_correctness_evaluator,
            answer_relevancy_evaluator,
            answer_semantic_similarity_evaluator,
            context_entity_recall_evaluator,
            context_precision_evaluator,
            context_recall_evaluator,
            faithfulness_evaluator,
        )

        return [
            faithfulness_evaluator,
            answer_relevancy_evaluator,
            context_precision_evaluator,
            context_recall_evaluator,
            answer_correctness_evaluator,
            answer_semantic_similarity_evaluator,
            context_entity_recall_evaluator,
        ]

    if vector == "structured":
        from requirements_graphrag_api.evaluation.structured_evaluators import (
            cypher_execution_success,
            cypher_parse_validity,
            cypher_safety,
            cypher_schema_adherence,
            result_correctness,
            result_shape_accuracy,
        )

        return [
            cypher_parse_validity,
            cypher_schema_adherence,
            cypher_execution_success,
            result_shape_accuracy,
            cypher_safety,
            result_correctness,
        ]

    if vector == "conversational":
        from requirements_graphrag_api.evaluation.conversational_evaluators import (
            conversation_combined,
        )

        return [conversation_combined]

    if vector == "intent":
        from requirements_graphrag_api.prompts.evaluation import (
            create_intent_accuracy_evaluator,
        )

        # Wrap the sync evaluator for LangSmith aevaluate()
        intent_fn = create_intent_accuracy_evaluator()

        async def intent_evaluator(run: Any, example: Any = None) -> dict[str, Any]:
            outputs = run.outputs or {}
            expected = (example.outputs or {}) if example else {}
            scores = intent_fn(
                {
                    "output": outputs.get("output", ""),
                    "expected": expected,
                }
            )
            return {
                "key": "intent_accuracy",
                "score": scores.get("intent_accuracy", 0.0),
            }

        return [intent_evaluator]

    msg = f"Unknown vector: {vector}"
    raise ValueError(msg)


# =============================================================================
# TARGET FUNCTIONS PER VECTOR
# =============================================================================


async def _create_explanatory_target(
    prompt_tag: str = "production",
) -> Any:
    """Create RAG pipeline target for explanatory vector."""
    from langchain_openai import ChatOpenAI

    from requirements_graphrag_api.config import get_config
    from requirements_graphrag_api.core.retrieval import (
        create_vector_retriever,
        graph_enriched_search,
    )
    from requirements_graphrag_api.neo4j_client import create_driver
    from requirements_graphrag_api.prompts import PromptName, get_prompt

    config = get_config()
    driver = create_driver(config)
    retriever = create_vector_retriever(driver, config)
    prompt = await get_prompt(PromptName.SYNTHESIS, tag=prompt_tag)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs.get("question", "")
        results = await graph_enriched_search(
            retriever,
            driver,
            question,
            limit=6,
        )
        contexts = [r["content"] for r in results]
        context = "\n\n".join(contexts)

        chain = prompt | llm
        response = await chain.ainvoke(
            {
                "context": context,
                "previous_context": "",
                "question": question,
            }
        )

        raw = response.content
        if raw.startswith("```"):
            raw_lines = raw.split("\n")
            raw = "\n".join(ln for ln in raw_lines if not ln.startswith("```")).strip()
        try:
            parsed = json.loads(raw)
            answer = parsed.get("answer", raw)
        except (json.JSONDecodeError, KeyError):
            answer = raw

        return {
            "answer": answer,
            "contexts": contexts,
            "intent": "explanatory",
        }

    return target, driver


async def _create_structured_target(
    prompt_tag: str = "production",
) -> Any:
    """Create Text2Cypher pipeline target for structured vector."""
    from langchain_openai import ChatOpenAI

    from requirements_graphrag_api.config import get_config
    from requirements_graphrag_api.neo4j_client import create_driver
    from requirements_graphrag_api.prompts import PromptName, get_prompt

    config = get_config()
    driver = create_driver(config)
    prompt = await get_prompt(PromptName.TEXT2CYPHER, tag=prompt_tag)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs.get("question", "")

        chain = prompt | llm
        response = await chain.ainvoke({"question": question})

        cypher = response.content.strip()
        if cypher.startswith("```"):
            lines = cypher.split("\n")
            cypher = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()

        # Execute the Cypher query
        error = ""
        results = []
        row_count = 0
        try:
            records, _, _ = await asyncio.to_thread(
                driver.execute_query,
                cypher,
            )
            results = [dict(r) for r in records]
            row_count = len(results)
        except Exception as e:
            error = str(e)

        return {
            "cypher": cypher,
            "output": cypher,
            "results": results,
            "row_count": row_count,
            "error": error,
        }

    return target, driver


async def _create_conversational_target(
    prompt_tag: str = "production",
) -> Any:
    """Create conversational pipeline target."""
    from langchain_openai import ChatOpenAI

    from requirements_graphrag_api.config import get_config
    from requirements_graphrag_api.core.retrieval import (
        create_vector_retriever,
        graph_enriched_search,
    )
    from requirements_graphrag_api.neo4j_client import create_driver
    from requirements_graphrag_api.prompts import PromptName, get_prompt

    config = get_config()
    driver = create_driver(config)
    retriever = create_vector_retriever(driver, config)
    prompt = await get_prompt(PromptName.SYNTHESIS, tag=prompt_tag)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs.get("question", "")
        history = inputs.get("history", "")

        results = await graph_enriched_search(
            retriever,
            driver,
            question,
            limit=6,
        )
        contexts = [r["content"] for r in results]
        context = "\n\n".join(contexts)

        chain = prompt | llm
        response = await chain.ainvoke(
            {
                "context": context,
                "previous_context": history,
                "question": question,
            }
        )

        raw = response.content
        if raw.startswith("```"):
            raw_lines = raw.split("\n")
            raw = "\n".join(ln for ln in raw_lines if not ln.startswith("```")).strip()
        try:
            parsed = json.loads(raw)
            answer = parsed.get("answer", raw)
        except (json.JSONDecodeError, KeyError):
            answer = raw

        return {"answer": answer, "output": answer}

    return target, driver


async def _create_intent_target(
    prompt_tag: str = "production",
) -> Any:
    """Create intent classification target."""
    from langchain_openai import ChatOpenAI

    from requirements_graphrag_api.prompts import PromptName, get_prompt

    prompt = await get_prompt(PromptName.INTENT_CLASSIFIER, tag=prompt_tag)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs.get("question", "")
        history = inputs.get("history", "")

        chain = prompt | llm
        response = await chain.ainvoke(
            {
                "question": question,
                "conversation_history": history,
            }
        )
        return {"output": response.content}

    # No driver needed for intent classification
    return target, None


# =============================================================================
# MAIN RUNNER
# =============================================================================


def _get_dataset_name(vector: str) -> str:
    """Get the LangSmith dataset name for a vector."""
    from requirements_graphrag_api.evaluation.constants import (
        DATASET_CONVERSATIONAL,
        DATASET_EXPLANATORY,
        DATASET_INTENT,
        DATASET_STRUCTURED,
    )

    return {
        "explanatory": DATASET_EXPLANATORY,
        "structured": DATASET_STRUCTURED,
        "conversational": DATASET_CONVERSATIONAL,
        "intent": DATASET_INTENT,
    }[vector]


async def _run_single_vector(
    vector: str,
    prompt_tag: str = "production",
    max_concurrency: int = 2,
) -> dict[str, Any]:
    """Run LangSmith experiment for a single vector.

    Returns:
        Dict with metric scores and experiment metadata.
    """
    from langsmith import aevaluate

    from requirements_graphrag_api.evaluation.constants import experiment_name

    dataset_name = _get_dataset_name(vector)
    exp_name = experiment_name(vector, prompt_tag)
    evaluators = _get_evaluators(vector)

    logger.info("─" * 60)
    logger.info("Vector: %s", vector)
    logger.info("Dataset: %s", dataset_name)
    logger.info("Experiment: %s", exp_name)
    logger.info("Evaluators: %d", len(evaluators))
    logger.info("Prompt tag: %s", prompt_tag)
    logger.info("─" * 60)

    # Create target function
    target_creators = {
        "explanatory": _create_explanatory_target,
        "structured": _create_structured_target,
        "conversational": _create_conversational_target,
        "intent": _create_intent_target,
    }
    target_fn, driver = await target_creators[vector](prompt_tag)

    try:
        results = await aevaluate(
            target_fn,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=exp_name,
            max_concurrency=max_concurrency,
            metadata={
                "vector": vector,
                "prompt_tag": prompt_tag,
            },
        )

        # Collect scores
        metric_scores: dict[str, list[float]] = {}
        async for result in results:
            eval_results = result.get("evaluation_results", {})
            for er in eval_results.get("results", []):
                key = er.key
                if er.score is not None:
                    metric_scores.setdefault(key, []).append(er.score)

        # Compute averages
        avg_scores = {
            metric: sum(scores) / len(scores) for metric, scores in metric_scores.items() if scores
        }

        logger.info("Results for %s:", vector)
        for metric, avg in sorted(avg_scores.items()):
            logger.info("  %s: %.3f", metric, avg)

        return {
            "vector": vector,
            "experiment": exp_name,
            "dataset": dataset_name,
            "scores": avg_scores,
        }

    finally:
        if driver is not None:
            driver.close()


def run_vectors(
    vectors: list[str],
    prompt_tag: str = "production",
    max_concurrency: int = 2,
    *,
    dry_run: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run evaluation for one or more vectors.

    Args:
        vectors: List of vectors to run.
        prompt_tag: Hub prompt tag ("production" or "staging").
        max_concurrency: Max concurrent evaluations per vector.
        dry_run: If True, only show what would happen.

    Returns:
        Dict mapping vector name to results.
    """
    if dry_run:
        for v in vectors:
            evaluators = _get_evaluators(v)
            dataset = _get_dataset_name(v)
            logger.info(
                "[DRY RUN] %s: %d evaluators, dataset=%s, tag=%s",
                v,
                len(evaluators),
                dataset,
                prompt_tag,
            )
        return {}

    async def _run_all() -> dict[str, dict[str, Any]]:
        all_results = {}
        for v in vectors:
            result = await _run_single_vector(v, prompt_tag, max_concurrency)
            all_results[v] = result
        return all_results

    return asyncio.run(_run_all())


def main() -> int:
    """Main entry point."""
    for var in ("LANGSMITH_API_KEY", "OPENAI_API_KEY"):
        if not os.getenv(var):
            logger.error("%s not set", var)
            return 1

    parser = argparse.ArgumentParser(
        description="Unified per-vector LangSmith evaluation runner",
    )
    parser.add_argument(
        "--vector",
        "-v",
        choices=[*VECTORS, "all"],
        required=True,
        help="Evaluation vector to run",
    )
    parser.add_argument(
        "--prompt-tag",
        "-t",
        choices=["production", "staging"],
        default="production",
        help="Hub prompt tag (default: production)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Max concurrent evaluations (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without running",
    )
    parser.add_argument(
        "--regression-gate",
        action="store_true",
        help="Check results against regression thresholds (exit 1 on failure)",
    )

    args = parser.parse_args()

    vectors = list(VECTORS) if args.vector == "all" else [args.vector]

    results = run_vectors(
        vectors,
        prompt_tag=args.prompt_tag,
        max_concurrency=args.max_concurrency,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return 0

    # Regression gate check
    if args.regression_gate and results:
        from requirements_graphrag_api.evaluation.regression import (
            check_all_vectors,
        )

        gate_input = {v: r["scores"] for v, r in results.items() if "scores" in r}
        reports = check_all_vectors(gate_input)

        any_failed = False
        for _vector, report in reports.items():
            logger.info(report.summary())
            if not report.passed:
                any_failed = True

        if any_failed:
            logger.error("Regression gate FAILED")
            return 1
        logger.info("Regression gate PASSED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
