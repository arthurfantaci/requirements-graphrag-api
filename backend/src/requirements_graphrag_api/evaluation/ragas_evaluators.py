"""RAGAS evaluators compatible with LangSmith evaluate().

This module provides async evaluator functions using Hub-versioned prompts
from definitions.py, compatible with langsmith.evaluate() (SDK 0.7+).

LangSmith Concepts:
- Evaluators take a Run and Example and return EvaluationResult
- Scores should be 0.0-1.0 scale
- Feedback keys identify metrics in the UI
- langsmith.evaluate() runs evaluators in ThreadPoolExecutor — callers must
  wrap these async evaluators in sync functions using a shared event loop

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
import math
import re
from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from requirements_graphrag_api.prompts import PromptName, get_prompt

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


def _parse_llm_json(response: str) -> dict[str, Any]:
    """Parse structured JSON from LLM response.

    Handles markdown code blocks and returns parsed dict.
    Returns empty dict on failure.

    Args:
        response: LLM response containing JSON.

    Returns:
        Parsed dict.
    """
    try:
        if "```json" in response:
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)

        return json.loads(response.strip())
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("Failed to parse LLM JSON: %s - %s", e, response[:100])
        return {}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity in [0.0, 1.0].
    """
    dot_product = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot_product / (norm_a * norm_b)))


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

    Uses claim-level verification: extracts claims from the answer and
    checks each against the context.

    Args:
        run: The run containing inputs and outputs.
        example: Optional example with expected outputs.

    Returns:
        Dict with 'key', 'score', and 'comment' for LangSmith.
    """
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

    template = await get_prompt(PromptName.EVAL_FAITHFULNESS)
    messages = template.format_messages(question=question, context=context, answer=answer)

    llm = _get_judge_llm()
    response = await llm.ainvoke(messages)
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

    template = await get_prompt(PromptName.EVAL_ANSWER_RELEVANCY)
    messages = template.format_messages(question=question, answer=answer)

    llm = _get_judge_llm()
    response = await llm.ainvoke(messages)
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
    contexts = (
        inputs.get("contexts", []) or outputs.get("contexts", []) or outputs.get("context", "")
    )

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

    template = await get_prompt(PromptName.EVAL_CONTEXT_PRECISION)
    messages = template.format_messages(question=question, contexts=contexts_str)

    llm = _get_judge_llm()
    response = await llm.ainvoke(messages)
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

    Uses atomic fact decomposition: decomposes ground truth into atomic facts
    and checks if each is supported by the retrieved contexts.

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

    ground_truth = ""
    if example and example.outputs:
        ground_truth = (
            example.outputs.get("expected_answer", "")
            or example.outputs.get("ground_truth", "")
            or example.outputs.get("answer", "")
        )

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

    template = await get_prompt(PromptName.EVAL_CONTEXT_RECALL)
    messages = template.format_messages(
        question=question,
        contexts=contexts_str,
        ground_truth=ground_truth,
    )

    llm = _get_judge_llm()
    response = await llm.ainvoke(messages)
    score, reasoning = _parse_llm_score(response.content)

    return {
        "key": "context_recall",
        "score": score,
        "comment": reasoning,
    }


async def answer_correctness_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Evaluate factual correctness of answer against ground truth.

    Decomposes both answer and ground truth into atomic claims,
    classifies each as TP/FP/FN, then computes F1 in Python.

    Args:
        run: The run containing inputs and outputs.
        example: Example with ground truth answer.

    Returns:
        Dict with 'key', 'score', and 'comment' for LangSmith.
    """
    inputs = run.inputs or {}
    outputs = run.outputs or {}

    question = inputs.get("question", "")
    answer = outputs.get("answer", "") or outputs.get("output", "")

    ground_truth = ""
    if example and example.outputs:
        ground_truth = (
            example.outputs.get("expected_answer", "")
            or example.outputs.get("ground_truth", "")
            or example.outputs.get("answer", "")
        )

    if not all([question, answer, ground_truth]):
        return {
            "key": "answer_correctness",
            "score": 0.0,
            "comment": "Missing required fields: question, answer, or ground_truth",
        }

    template = await get_prompt(PromptName.EVAL_ANSWER_CORRECTNESS)
    messages = template.format_messages(question=question, answer=answer, ground_truth=ground_truth)

    llm = _get_judge_llm()
    response = await llm.ainvoke(messages)
    parsed = _parse_llm_json(response.content)

    # Compute F1 from LLM-provided claim classifications
    tp = int(parsed.get("tp", 0))
    fp = int(parsed.get("fp", 0))
    fn = int(parsed.get("fn", 0))
    reasoning = str(parsed.get("reasoning", ""))

    denominator = 2 * tp + fp + fn
    f1 = (2 * tp / denominator) if denominator > 0 else 0.0

    return {
        "key": "answer_correctness",
        "score": min(1.0, max(0.0, f1)),
        "comment": reasoning,
    }


async def answer_semantic_similarity_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Evaluate semantic similarity between answer and ground truth.

    Uses embedding cosine similarity — no LLM judge call needed.

    Args:
        run: The run containing inputs and outputs.
        example: Example with ground truth answer.

    Returns:
        Dict with 'key', 'score', and 'comment' for LangSmith.
    """
    outputs = run.outputs or {}

    answer = outputs.get("answer", "") or outputs.get("output", "")

    ground_truth = ""
    if example and example.outputs:
        ground_truth = (
            example.outputs.get("expected_answer", "")
            or example.outputs.get("ground_truth", "")
            or example.outputs.get("answer", "")
        )

    if not all([answer, ground_truth]):
        return {
            "key": "answer_semantic_similarity",
            "score": 0.0,
            "comment": "Missing required fields: answer or ground_truth",
        }

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    answer_vec, gt_vec = await embeddings.aembed_documents([answer, ground_truth])

    score = _cosine_similarity(answer_vec, gt_vec)

    return {
        "key": "answer_semantic_similarity",
        "score": score,
        "comment": "",
    }


async def context_entity_recall_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Evaluate entity recall between context and ground truth.

    Extracts entities from both context and ground truth using an LLM,
    then computes recall as set intersection over ground truth entities.

    Args:
        run: The run containing inputs and outputs.
        example: Example with ground truth answer.

    Returns:
        Dict with 'key', 'score', and 'comment' for LangSmith.
    """
    inputs = run.inputs or {}
    outputs = run.outputs or {}

    contexts = (
        inputs.get("contexts", []) or outputs.get("contexts", []) or outputs.get("context", "")
    )

    ground_truth = ""
    if example and example.outputs:
        ground_truth = (
            example.outputs.get("expected_answer", "")
            or example.outputs.get("ground_truth", "")
            or example.outputs.get("answer", "")
        )

    if isinstance(contexts, list):
        contexts_str = "\n\n".join(str(c) for c in contexts)
    else:
        contexts_str = str(contexts)

    if not all([contexts_str, ground_truth]):
        return {
            "key": "context_entity_recall",
            "score": 0.0,
            "comment": "Missing required fields: contexts or ground_truth",
        }

    template = await get_prompt(PromptName.EVAL_CONTEXT_ENTITY_RECALL)
    llm = _get_judge_llm()

    # Call 1: extract entities from context
    ctx_messages = template.format_messages(text=contexts_str)
    ctx_response = await llm.ainvoke(ctx_messages)
    ctx_parsed = _parse_llm_json(ctx_response.content)
    ctx_entities = {e.strip().lower() for e in ctx_parsed.get("entities", [])}

    # Call 2: extract entities from ground truth
    gt_messages = template.format_messages(text=ground_truth)
    gt_response = await llm.ainvoke(gt_messages)
    gt_parsed = _parse_llm_json(gt_response.content)
    gt_entities = {e.strip().lower() for e in gt_parsed.get("entities", [])}

    # Compute recall: proportion of ground truth entities found in context
    if not gt_entities:
        return {
            "key": "context_entity_recall",
            "score": 1.0,
            "comment": "No entities in ground truth",
        }

    overlap = ctx_entities & gt_entities
    recall = len(overlap) / len(gt_entities)

    comment = (
        f"Found {len(overlap)}/{len(gt_entities)} ground truth entities in context. "
        f"Context entities: {sorted(ctx_entities)}, GT entities: {sorted(gt_entities)}"
    )

    return {
        "key": "context_entity_recall",
        "score": min(1.0, max(0.0, recall)),
        "comment": comment,
    }


__all__ = [
    "answer_correctness_evaluator",
    "answer_relevancy_evaluator",
    "answer_semantic_similarity_evaluator",
    "context_entity_recall_evaluator",
    "context_precision_evaluator",
    "context_recall_evaluator",
    "faithfulness_evaluator",
]
