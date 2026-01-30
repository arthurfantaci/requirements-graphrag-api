"""RAGAS evaluators compatible with LangSmith evaluate().

This module provides evaluator functions that wrap the RAGAS-style prompts
from metrics.py into a format compatible with langsmith.evaluate().

LangSmith Concepts:
- Evaluators take a Run and Example and return EvaluationResult
- Scores should be 0.0-1.0 scale
- Feedback keys identify metrics in the UI

Usage:
    from langsmith import evaluate
    from requirements_graphrag_api.evaluation.ragas_evaluators import (
        faithfulness_evaluator,
        answer_relevancy_evaluator,
    )

    results = evaluate(
        target=my_rag_chain,
        data="my-dataset",
        evaluators=[faithfulness_evaluator, answer_relevancy_evaluator],
    )
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI

from requirements_graphrag_api.evaluation.metrics import (
    ANSWER_RELEVANCY_PROMPT,
    CONTEXT_PRECISION_PROMPT,
    CONTEXT_RECALL_PROMPT,
    FAITHFULNESS_PROMPT,
)

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

logger = logging.getLogger(__name__)

# Default model for LLM-as-judge evaluations
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


def _parse_llm_score(response: str) -> tuple[float, str]:
    """Parse score and reasoning from LLM evaluation response.

    Args:
        response: LLM response containing JSON with score and reasoning.

    Returns:
        Tuple of (score, reasoning).
    """
    try:
        # Handle markdown code blocks
        if "```json" in response:
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)

        parsed = json.loads(response.strip())
        score = float(parsed.get("score", 0.0))
        reasoning = str(parsed.get("reasoning", ""))
        return min(1.0, max(0.0, score)), reasoning
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("Failed to parse LLM score: %s - %s", e, response[:100])
        return 0.0, f"Parse error: {e}"


def _get_judge_llm(model: str = DEFAULT_JUDGE_MODEL) -> ChatOpenAI:
    """Get LLM instance for judging.

    Args:
        model: Model name to use.

    Returns:
        ChatOpenAI instance.
    """
    return ChatOpenAI(model=model, temperature=0)


async def faithfulness_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Evaluate answer faithfulness to context.

    Checks whether the answer is grounded in the retrieved context.
    An answer is faithful if every claim can be verified from the context.

    Args:
        run: The run containing inputs and outputs.
        example: Optional example with expected outputs.

    Returns:
        Dict with 'key', 'score', and 'comment' for LangSmith.
    """
    # Extract data from run
    inputs = run.inputs or {}
    outputs = run.outputs or {}

    question = inputs.get("question", "")
    context = inputs.get("context", "") or outputs.get("context", "")
    answer = outputs.get("answer", "") or outputs.get("output", "")

    if not all([question, context, answer]):
        return {
            "key": "faithfulness",
            "score": 0.0,
            "comment": "Missing required fields: question, context, or answer",
        }

    # Format and run evaluation prompt
    prompt = FAITHFULNESS_PROMPT.format(
        question=question,
        context=context,
        answer=answer,
    )

    llm = _get_judge_llm()
    response = await llm.ainvoke(prompt)
    score, reasoning = _parse_llm_score(response.content)

    return {
        "key": "faithfulness",
        "score": score,
        "comment": reasoning,
    }


async def answer_relevancy_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Evaluate answer relevancy to question.

    Checks whether the answer directly addresses the question asked.

    Args:
        run: The run containing inputs and outputs.
        example: Optional example with expected outputs.

    Returns:
        Dict with 'key', 'score', and 'comment' for LangSmith.
    """
    inputs = run.inputs or {}
    outputs = run.outputs or {}

    question = inputs.get("question", "")
    answer = outputs.get("answer", "") or outputs.get("output", "")

    if not all([question, answer]):
        return {
            "key": "answer_relevancy",
            "score": 0.0,
            "comment": "Missing required fields: question or answer",
        }

    prompt = ANSWER_RELEVANCY_PROMPT.format(
        question=question,
        answer=answer,
    )

    llm = _get_judge_llm()
    response = await llm.ainvoke(prompt)
    score, reasoning = _parse_llm_score(response.content)

    return {
        "key": "answer_relevancy",
        "score": score,
        "comment": reasoning,
    }


async def context_precision_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Evaluate context precision.

    Checks what proportion of retrieved contexts are relevant to the question.

    Args:
        run: The run containing inputs and outputs.
        example: Optional example with expected outputs.

    Returns:
        Dict with 'key', 'score', and 'comment' for LangSmith.
    """
    inputs = run.inputs or {}
    outputs = run.outputs or {}

    question = inputs.get("question", "")
    # Contexts can be in inputs or outputs depending on pipeline structure
    contexts = (
        inputs.get("contexts", []) or outputs.get("contexts", []) or outputs.get("context", "")
    )

    # Convert to string representation if list
    if isinstance(contexts, list):
        contexts_str = "\n\n---\n\n".join(str(c) for c in contexts)
    else:
        contexts_str = str(contexts)

    if not all([question, contexts_str]):
        return {
            "key": "context_precision",
            "score": 0.0,
            "comment": "Missing required fields: question or contexts",
        }

    prompt = CONTEXT_PRECISION_PROMPT.format(
        question=question,
        contexts=contexts_str,
    )

    llm = _get_judge_llm()
    response = await llm.ainvoke(prompt)
    score, reasoning = _parse_llm_score(response.content)

    return {
        "key": "context_precision",
        "score": score,
        "comment": reasoning,
    }


async def context_recall_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Evaluate context recall against ground truth.

    Checks whether retrieved contexts contain the information needed
    to generate the ground truth answer.

    Args:
        run: The run containing inputs and outputs.
        example: Example with ground truth answer.

    Returns:
        Dict with 'key', 'score', and 'comment' for LangSmith.
    """
    inputs = run.inputs or {}
    outputs = run.outputs or {}

    question = inputs.get("question", "")
    contexts = (
        inputs.get("contexts", []) or outputs.get("contexts", []) or outputs.get("context", "")
    )

    # Get ground truth from example
    ground_truth = ""
    if example and example.outputs:
        ground_truth = (
            example.outputs.get("expected_answer", "")
            or example.outputs.get("ground_truth", "")
            or example.outputs.get("answer", "")
        )

    # Convert to string representation if list
    if isinstance(contexts, list):
        contexts_str = "\n\n---\n\n".join(str(c) for c in contexts)
    else:
        contexts_str = str(contexts)

    if not all([question, contexts_str, ground_truth]):
        return {
            "key": "context_recall",
            "score": 0.0,
            "comment": "Missing required fields: question, contexts, or ground_truth",
        }

    prompt = CONTEXT_RECALL_PROMPT.format(
        question=question,
        contexts=contexts_str,
        ground_truth=ground_truth,
    )

    llm = _get_judge_llm()
    response = await llm.ainvoke(prompt)
    score, reasoning = _parse_llm_score(response.content)

    return {
        "key": "context_recall",
        "score": score,
        "comment": reasoning,
    }


# Synchronous wrappers for use with langsmith.evaluate()
# which expects sync functions by default


def faithfulness_evaluator_sync(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for faithfulness_evaluator."""
    import asyncio

    return asyncio.run(faithfulness_evaluator(run, example))


def answer_relevancy_evaluator_sync(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for answer_relevancy_evaluator."""
    import asyncio

    return asyncio.run(answer_relevancy_evaluator(run, example))


def context_precision_evaluator_sync(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for context_precision_evaluator."""
    import asyncio

    return asyncio.run(context_precision_evaluator(run, example))


def context_recall_evaluator_sync(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for context_recall_evaluator."""
    import asyncio

    return asyncio.run(context_recall_evaluator(run, example))


__all__ = [
    "answer_relevancy_evaluator",
    "answer_relevancy_evaluator_sync",
    "context_precision_evaluator",
    "context_precision_evaluator_sync",
    "context_recall_evaluator",
    "context_recall_evaluator_sync",
    "faithfulness_evaluator",
    "faithfulness_evaluator_sync",
]
