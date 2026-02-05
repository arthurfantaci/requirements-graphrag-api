"""Agentic RAG evaluators for LangSmith.

This module provides evaluators specific to the agentic RAG system:
- Tool Selection Accuracy: Did the agent choose appropriate tools?
- Iteration Efficiency: How many iterations were needed vs optimal?
- Critic Calibration: How accurate was the self-critique?
- Answer Quality: Overall answer quality (combines existing metrics)

These evaluators work with the agentic orchestrator outputs to assess
agent behavior beyond just answer quality.

Usage:
    from langsmith import evaluate
    from requirements_graphrag_api.evaluation.agentic_evaluators import (
        tool_selection_evaluator,
        iteration_efficiency_evaluator,
        critic_calibration_evaluator,
    )

    results = evaluate(
        target=agentic_rag_chain,
        data="graphrag-agentic-eval",
        evaluators=[tool_selection_evaluator, iteration_efficiency_evaluator],
    )
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

logger = logging.getLogger(__name__)

# Default model for LLM-as-judge evaluations
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

# Maximum iterations before penalizing efficiency
MAX_EFFICIENT_ITERATIONS = 3


# =============================================================================
# AGENTIC EVALUATION PROMPTS
# =============================================================================

TOOL_SELECTION_PROMPT = """You are evaluating whether an AI agent selected \
appropriate tools for a question.

Given:
- Question: {question}
- Expected tools: {expected_tools}
- Tools actually used: {actual_tools}
- Tool usage order: {tool_order}

Evaluate whether the agent selected the right tools and used them in a sensible order.

Scoring criteria:
- 1.0: All expected tools used, no unnecessary tools, logical order
- 0.75: All expected tools used with minor inefficiencies
- 0.5: Some expected tools missing OR unnecessary tools used
- 0.25: Most expected tools missing
- 0.0: Wrong tools entirely

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""

ITERATION_EFFICIENCY_PROMPT = """You are evaluating the iteration efficiency of an AI agent.

Given:
- Question: {question}
- Question complexity: {complexity}
- Number of iterations used: {iterations_used}
- Expected iterations for this complexity: {expected_iterations}

Evaluate whether the agent used an appropriate number of iterations.

Scoring criteria:
- 1.0: Optimal number of iterations (within expected range)
- 0.75: 1 iteration more than expected
- 0.5: 2 iterations more than expected or 1 less than needed
- 0.25: Significantly over-iterated (wasteful)
- 0.0: Under-iterated (incomplete answer) or massively over-iterated

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""

CRITIC_CALIBRATION_PROMPT = """You are evaluating the calibration of an AI agent's self-critique.

Given:
- Question: {question}
- Agent's confidence score: {confidence}
- Agent's completeness assessment: {completeness}
- Actual answer quality (expert rating): {actual_quality}
- Missing aspects identified by agent: {missing_aspects}
- Actual missing aspects (expert): {actual_missing}

Evaluate whether the agent's self-assessment matched reality.

A well-calibrated critic:
- Has confidence matching actual quality
- Correctly identifies completeness level
- Finds the actual missing aspects

Scoring criteria:
- 1.0: Perfect calibration (confidence within 0.1 of actual quality, all aspects identified)
- 0.75: Good calibration (confidence within 0.2, most aspects identified)
- 0.5: Moderate calibration (confidence off by 0.3, some aspects missed)
- 0.25: Poor calibration (overconfident or underconfident by >0.4)
- 0.0: Completely miscalibrated

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""

MULTI_HOP_REASONING_PROMPT = """You are evaluating an AI agent's multi-hop reasoning ability.

Given:
- Multi-hop question: {question}
- Required reasoning steps: {required_steps}
- Agent's reasoning chain: {reasoning_chain}
- Final answer: {answer}
- Ground truth answer: {ground_truth}

Evaluate whether the agent correctly connected multiple pieces of information.

Multi-hop reasoning requires:
1. Identifying all relevant entities/concepts
2. Finding relationships between them
3. Synthesizing information across sources
4. Arriving at a logically sound conclusion

Scoring criteria:
- 1.0: All reasoning steps present, correct connections, accurate answer
- 0.75: Most steps present, minor logic gaps, mostly correct answer
- 0.5: Some steps missing, partial reasoning, partially correct answer
- 0.25: Major reasoning gaps, incorrect connections
- 0.0: No multi-hop reasoning evident, wrong answer

Respond with only a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""


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
    """Get LLM instance for judging."""
    return ChatOpenAI(model=model, temperature=0)


# =============================================================================
# EVALUATOR FUNCTIONS
# =============================================================================


async def tool_selection_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Evaluate agent's tool selection accuracy.

    Checks whether the agent selected appropriate tools for the query type.

    Args:
        run: The run containing inputs and outputs with tool usage.
        example: Example with expected tools.

    Returns:
        Dict with 'key', 'score', and 'comment' for LangSmith.
    """
    inputs = run.inputs or {}
    outputs = run.outputs or {}

    question = inputs.get("question", "") or inputs.get("query", "")

    # Extract tool usage from outputs or metadata
    actual_tools = outputs.get("tools_used", [])
    tool_order = outputs.get("tool_order", [])

    # If no explicit tool tracking, try to infer from run name or metadata
    if not actual_tools and run.name:
        # Check if run name indicates tool usage
        if "graph_search" in run.name.lower():
            actual_tools.append("graph_search")
        if "text2cypher" in run.name.lower():
            actual_tools.append("text2cypher")

    # Get expected tools from example
    expected_tools = []
    if example and example.outputs:
        expected_tools = example.outputs.get("expected_tools", [])

    if not expected_tools:
        return {
            "key": "tool_selection",
            "score": 1.0,  # No expected tools specified, skip evaluation
            "comment": "No expected tools specified in example",
        }

    prompt = TOOL_SELECTION_PROMPT.format(
        question=question,
        expected_tools=", ".join(expected_tools),
        actual_tools=", ".join(actual_tools) if actual_tools else "None recorded",
        tool_order=", ".join(tool_order) if tool_order else "Unknown",
    )

    llm = _get_judge_llm()
    response = await llm.ainvoke(prompt)
    score, reasoning = _parse_llm_score(response.content)

    return {
        "key": "tool_selection",
        "score": score,
        "comment": reasoning,
    }


async def iteration_efficiency_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Evaluate agent's iteration efficiency.

    Checks whether the agent completed the task in an optimal number of iterations.

    Args:
        run: The run containing iteration count in outputs.
        example: Example with expected iterations for complexity level.

    Returns:
        Dict with 'key', 'score', and 'comment' for LangSmith.
    """
    inputs = run.inputs or {}
    outputs = run.outputs or {}

    question = inputs.get("question", "") or inputs.get("query", "")
    iterations_used = outputs.get("iteration_count", 1)

    # Get expected iterations from example based on complexity
    complexity = "medium"
    expected_iterations = 2

    if example and example.inputs:
        complexity = example.inputs.get("complexity", "medium")

    if example and example.outputs:
        expected_iterations = example.outputs.get("expected_iterations", 2)

    # Default expected iterations by complexity
    complexity_defaults = {
        "simple": 1,
        "medium": 2,
        "complex": 3,
        "multi_hop": 3,
    }
    if not expected_iterations:
        expected_iterations = complexity_defaults.get(complexity, 2)

    prompt = ITERATION_EFFICIENCY_PROMPT.format(
        question=question,
        complexity=complexity,
        iterations_used=iterations_used,
        expected_iterations=expected_iterations,
    )

    llm = _get_judge_llm()
    response = await llm.ainvoke(prompt)
    score, reasoning = _parse_llm_score(response.content)

    return {
        "key": "iteration_efficiency",
        "score": score,
        "comment": reasoning,
    }


async def critic_calibration_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Evaluate the calibration of the agent's self-critique.

    Checks whether the critic's confidence and completeness assessments
    match the actual answer quality.

    Args:
        run: The run containing critique in outputs.
        example: Example with expert quality assessment.

    Returns:
        Dict with 'key', 'score', and 'comment' for LangSmith.
    """
    inputs = run.inputs or {}
    outputs = run.outputs or {}

    question = inputs.get("question", "") or inputs.get("query", "")

    # Extract critique from outputs
    critique = outputs.get("critique", {})
    if isinstance(critique, dict):
        confidence = critique.get("confidence", 0.5)
        completeness = critique.get("completeness", "partial")
        missing_aspects = critique.get("missing_aspects", [])
    else:
        confidence = 0.5
        completeness = "unknown"
        missing_aspects = []

    # Get expert assessment from example
    actual_quality = 0.5
    actual_missing = []

    if example and example.outputs:
        actual_quality = example.outputs.get("expert_quality", 0.5)
        actual_missing = example.outputs.get("expert_missing_aspects", [])

    prompt = CRITIC_CALIBRATION_PROMPT.format(
        question=question,
        confidence=confidence,
        completeness=completeness,
        actual_quality=actual_quality,
        missing_aspects=", ".join(missing_aspects) if missing_aspects else "None",
        actual_missing=", ".join(actual_missing) if actual_missing else "None",
    )

    llm = _get_judge_llm()
    response = await llm.ainvoke(prompt)
    score, reasoning = _parse_llm_score(response.content)

    return {
        "key": "critic_calibration",
        "score": score,
        "comment": reasoning,
    }


async def multi_hop_reasoning_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Evaluate the agent's multi-hop reasoning ability.

    Checks whether the agent correctly connected multiple pieces of information
    to answer complex questions.

    Args:
        run: The run containing answer and reasoning chain.
        example: Example with required reasoning steps and ground truth.

    Returns:
        Dict with 'key', 'score', and 'comment' for LangSmith.
    """
    inputs = run.inputs or {}
    outputs = run.outputs or {}

    question = inputs.get("question", "") or inputs.get("query", "")
    answer = outputs.get("answer", "") or outputs.get("final_answer", "")
    reasoning_chain = outputs.get("reasoning_chain", [])

    # Get ground truth and required steps from example
    ground_truth = ""
    required_steps = []

    if example and example.outputs:
        ground_truth = example.outputs.get("expected_answer", "")
        required_steps = example.outputs.get("required_reasoning_steps", [])

    if not ground_truth:
        return {
            "key": "multi_hop_reasoning",
            "score": 1.0,
            "comment": "No ground truth provided for multi-hop evaluation",
        }

    steps_str = "\n".join(f"- {s}" for s in required_steps) if required_steps else "Not specified"
    chain_str = "\n".join(f"- {s}" for s in reasoning_chain) if reasoning_chain else "Not recorded"

    prompt = MULTI_HOP_REASONING_PROMPT.format(
        question=question,
        required_steps=steps_str,
        reasoning_chain=chain_str,
        answer=answer,
        ground_truth=ground_truth,
    )

    llm = _get_judge_llm()
    response = await llm.ainvoke(prompt)
    score, reasoning = _parse_llm_score(response.content)

    return {
        "key": "multi_hop_reasoning",
        "score": score,
        "comment": reasoning,
    }


# =============================================================================
# SYNCHRONOUS WRAPPERS
# =============================================================================


def tool_selection_evaluator_sync(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for tool_selection_evaluator."""
    import asyncio

    return asyncio.run(tool_selection_evaluator(run, example))


def iteration_efficiency_evaluator_sync(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for iteration_efficiency_evaluator."""
    import asyncio

    return asyncio.run(iteration_efficiency_evaluator(run, example))


def critic_calibration_evaluator_sync(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for critic_calibration_evaluator."""
    import asyncio

    return asyncio.run(critic_calibration_evaluator(run, example))


def multi_hop_reasoning_evaluator_sync(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for multi_hop_reasoning_evaluator."""
    import asyncio

    return asyncio.run(multi_hop_reasoning_evaluator(run, example))


__all__ = [
    "critic_calibration_evaluator",
    "critic_calibration_evaluator_sync",
    "iteration_efficiency_evaluator",
    "iteration_efficiency_evaluator_sync",
    "multi_hop_reasoning_evaluator",
    "multi_hop_reasoning_evaluator_sync",
    "tool_selection_evaluator",
    "tool_selection_evaluator_sync",
]
