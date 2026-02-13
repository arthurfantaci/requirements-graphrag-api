"""Conversational vector evaluators for conversation recall evaluation.

Provides 4 evaluators compatible with ``langsmith.evaluate()``:
- All are LLM-as-judge using prompts from definitions.py
- ``conversation_combined`` is a cost-optimized batched evaluator that
  produces 3 scores in a single LLM call

Usage:
    evaluators=[conversation_coherence, context_retention, conversation_hallucination]
    # OR the batched version:
    evaluators=[conversation_combined]
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI

from requirements_graphrag_api.evaluation.constants import JUDGE_MODEL
from requirements_graphrag_api.prompts import PromptName, get_prompt

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

logger = logging.getLogger(__name__)


def _extract_outputs(run: Run, example: Example | None) -> dict[str, str]:
    """Extract common fields from run/example for conversational eval."""
    inputs = run.inputs or {}
    outputs = run.outputs or {}
    ref = (example.outputs or {}) if example else {}

    return {
        "history": inputs.get("history", "") or inputs.get("conversation_history", ""),
        "question": inputs.get("question", ""),
        "answer": outputs.get("answer", "") or outputs.get("output", ""),
        "expected_references": str(ref.get("expected_references", "[]")),
    }


async def conversation_coherence(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """LLM-as-judge: evaluates coherence between response and conversation history.

    Score 0.0-1.0. First-turn (no history) defaults to 1.0 unless off-topic.
    """
    fields = _extract_outputs(run, example)
    if not fields["question"] or not fields["answer"]:
        return {"key": "conv_coherence", "score": 0.0, "comment": "Missing question or answer"}

    template = await get_prompt(PromptName.EVAL_CONV_COHERENCE)
    messages = template.format_messages(
        history=fields["history"],
        question=fields["question"],
        answer=fields["answer"],
    )

    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    response = await llm.ainvoke(messages)

    try:
        parsed = json.loads(response.content)
        score = float(parsed.get("score", 0.0))
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        from requirements_graphrag_api.evaluation.ragas_evaluators import _parse_llm_score

        score, reasoning = _parse_llm_score(response.content)

    return {"key": "conv_coherence", "score": score, "comment": reasoning}


async def context_retention(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """LLM-as-judge: evaluates whether the response retains conversation history.

    Score 0.0-1.0 based on expected reference coverage.
    """
    fields = _extract_outputs(run, example)
    if not fields["question"] or not fields["answer"]:
        return {
            "key": "conv_context_retention",
            "score": 0.0,
            "comment": "Missing question or answer",
        }

    template = await get_prompt(PromptName.EVAL_CONV_CONTEXT_RETENTION)
    messages = template.format_messages(
        history=fields["history"],
        question=fields["question"],
        answer=fields["answer"],
        expected_references=fields["expected_references"],
    )

    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    response = await llm.ainvoke(messages)

    try:
        parsed = json.loads(response.content)
        score = float(parsed.get("score", 0.0))
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        from requirements_graphrag_api.evaluation.ragas_evaluators import _parse_llm_score

        score, reasoning = _parse_llm_score(response.content)

    return {"key": "conv_context_retention", "score": score, "comment": reasoning}


async def conversation_hallucination(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """LLM-as-judge: binary check for fabricated conversation content.

    Score: 1 = clean, 0 = hallucination detected.
    """
    fields = _extract_outputs(run, example)
    if not fields["question"] or not fields["answer"]:
        return {"key": "conv_hallucination", "score": 0, "comment": "Missing question or answer"}

    template = await get_prompt(PromptName.EVAL_CONV_HALLUCINATION)
    messages = template.format_messages(
        history=fields["history"],
        question=fields["question"],
        answer=fields["answer"],
    )

    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    response = await llm.ainvoke(messages)

    try:
        parsed = json.loads(response.content)
        score = int(parsed.get("score", 0))
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        from requirements_graphrag_api.evaluation.ragas_evaluators import _parse_llm_score

        score_f, reasoning = _parse_llm_score(response.content)
        score = 1 if score_f >= 0.5 else 0

    return {"key": "conv_hallucination", "score": score, "comment": reasoning}


async def conversation_combined(
    run: Run,
    example: Example | None = None,
) -> list[dict[str, Any]]:
    """Batched LLM-as-judge: 3 scores in 1 call (cost-optimized).

    Returns a list of 3 evaluation results:
    - conv_coherence (0.0-1.0)
    - conv_context_retention (0.0-1.0)
    - conv_hallucination (0 or 1)
    """
    fields = _extract_outputs(run, example)
    if not fields["question"] or not fields["answer"]:
        return [
            {"key": "conv_coherence", "score": 0.0, "comment": "Missing question or answer"},
            {
                "key": "conv_context_retention",
                "score": 0.0,
                "comment": "Missing question or answer",
            },
            {"key": "conv_hallucination", "score": 0, "comment": "Missing question or answer"},
        ]

    template = await get_prompt(PromptName.EVAL_CONV_COMBINED)
    messages = template.format_messages(
        history=fields["history"],
        question=fields["question"],
        answer=fields["answer"],
        expected_references=fields["expected_references"],
    )

    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    response = await llm.ainvoke(messages)

    try:
        parsed = json.loads(response.content)
        coherence = parsed.get("coherence", {})
        retention = parsed.get("context_retention", {})
        hallucination = parsed.get("hallucination", {})

        return [
            {
                "key": "conv_coherence",
                "score": float(coherence.get("score", 0.0)),
                "comment": coherence.get("reasoning", ""),
            },
            {
                "key": "conv_context_retention",
                "score": float(retention.get("score", 0.0)),
                "comment": retention.get("reasoning", ""),
            },
            {
                "key": "conv_hallucination",
                "score": int(hallucination.get("score", 0)),
                "comment": hallucination.get("reasoning", ""),
            },
        ]
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.warning("Failed to parse combined judge response: %s", e)
        return [
            {"key": "conv_coherence", "score": 0.0, "comment": f"Parse error: {e}"},
            {"key": "conv_context_retention", "score": 0.0, "comment": f"Parse error: {e}"},
            {"key": "conv_hallucination", "score": 0, "comment": f"Parse error: {e}"},
        ]


__all__ = [
    "context_retention",
    "conversation_coherence",
    "conversation_combined",
    "conversation_hallucination",
]
