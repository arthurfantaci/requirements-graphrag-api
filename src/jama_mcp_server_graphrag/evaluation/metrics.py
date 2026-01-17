"""RAG evaluation metrics using RAGAS and LLM-based assessment.

Implements core RAG evaluation metrics:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are retrieved contexts relevant?
- Context Recall: Are all necessary contexts retrieved?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from jama_mcp_server_graphrag.observability import traceable

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class RAGMetrics:
    """Container for RAG evaluation metrics.

    Attributes:
        faithfulness: Score (0-1) measuring answer groundedness in context.
        answer_relevancy: Score (0-1) measuring answer relevance to question.
        context_precision: Score (0-1) measuring context relevance.
        context_recall: Score (0-1) measuring context completeness.
    """

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def average(self) -> float:
        """Calculate average score across all metrics."""
        return (
            self.faithfulness + self.answer_relevancy + self.context_precision + self.context_recall
        ) / 4

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary format."""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "average": self.average,
        }


FAITHFULNESS_PROMPT = """You are evaluating the faithfulness of an AI-generated answer.

Faithfulness measures whether the answer is fully grounded in the provided context.
A faithful answer only contains information that can be directly inferred from the context.

Context:
{context}

Question: {question}

Answer: {answer}

Rate the faithfulness on a scale of 0 to 1:
- 1.0: Every claim in the answer is supported by the context
- 0.5: Some claims are supported, some are not
- 0.0: The answer contains claims not supported by context

Respond with ONLY a decimal number between 0 and 1."""

ANSWER_RELEVANCY_PROMPT = """You are evaluating the relevancy of an AI-generated answer.

Answer relevancy measures how well the answer addresses the question asked.
A relevant answer directly addresses what was asked without unnecessary information.

Question: {question}

Answer: {answer}

Rate the answer relevancy on a scale of 0 to 1:
- 1.0: The answer directly and completely addresses the question
- 0.5: The answer partially addresses the question
- 0.0: The answer does not address the question at all

Respond with ONLY a decimal number between 0 and 1."""

CONTEXT_PRECISION_PROMPT = """You are evaluating the precision of retrieved contexts.

Context precision measures whether the retrieved contexts are relevant to the question.
High precision means all retrieved contexts are useful for answering the question.

Question: {question}

Retrieved Contexts:
{contexts}

Rate the context precision on a scale of 0 to 1:
- 1.0: All contexts are highly relevant to the question
- 0.5: Some contexts are relevant, others are not
- 0.0: None of the contexts are relevant

Respond with ONLY a decimal number between 0 and 1."""

CONTEXT_RECALL_PROMPT = """You are evaluating the recall of retrieved contexts.

Context recall measures whether the retrieved contexts contain all information
needed to answer the question according to the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Retrieved Contexts:
{contexts}

Rate the context recall on a scale of 0 to 1:
- 1.0: Contexts contain all information needed for the ground truth answer
- 0.5: Contexts contain some but not all needed information
- 0.0: Contexts are missing key information from ground truth

Respond with ONLY a decimal number between 0 and 1."""


def _parse_score(response: str) -> float:
    """Parse a score from LLM response.

    Args:
        response: LLM response containing a score.

    Returns:
        Parsed score clamped between 0 and 1.
    """
    try:
        # Extract first number from response
        cleaned = response.strip()
        score = float(cleaned)
        return max(0.0, min(1.0, score))
    except ValueError:
        logger.warning("Failed to parse score from response: %s", response[:50])
        return 0.5  # Default to middle score on parse failure


@traceable(name="compute_faithfulness", run_type="chain")
async def compute_faithfulness(
    config: AppConfig,
    question: str,
    answer: str,
    contexts: list[str],
) -> float:
    """Compute faithfulness score for an answer.

    Measures whether the answer is grounded in the provided contexts.

    Args:
        config: Application configuration.
        question: The user's question.
        answer: The generated answer.
        contexts: Retrieved context passages.

    Returns:
        Faithfulness score between 0 and 1.
    """
    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    context_text = "\n---\n".join(contexts) if contexts else "No context provided."

    messages = [
        SystemMessage(content="You are an evaluation assistant."),
        HumanMessage(
            content=FAITHFULNESS_PROMPT.format(
                context=context_text,
                question=question,
                answer=answer,
            )
        ),
    ]

    chain = llm | StrOutputParser()
    response = await chain.ainvoke(messages)
    return _parse_score(response)


@traceable(name="compute_answer_relevancy", run_type="chain")
async def compute_answer_relevancy(
    config: AppConfig,
    question: str,
    answer: str,
) -> float:
    """Compute answer relevancy score.

    Measures how well the answer addresses the question.

    Args:
        config: Application configuration.
        question: The user's question.
        answer: The generated answer.

    Returns:
        Answer relevancy score between 0 and 1.
    """
    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    messages = [
        SystemMessage(content="You are an evaluation assistant."),
        HumanMessage(
            content=ANSWER_RELEVANCY_PROMPT.format(
                question=question,
                answer=answer,
            )
        ),
    ]

    chain = llm | StrOutputParser()
    response = await chain.ainvoke(messages)
    return _parse_score(response)


@traceable(name="compute_context_precision", run_type="chain")
async def compute_context_precision(
    config: AppConfig,
    question: str,
    contexts: list[str],
) -> float:
    """Compute context precision score.

    Measures whether retrieved contexts are relevant to the question.

    Args:
        config: Application configuration.
        question: The user's question.
        contexts: Retrieved context passages.

    Returns:
        Context precision score between 0 and 1.
    """
    if not contexts:
        return 0.0

    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    contexts_text = "\n---\n".join(f"[Context {i + 1}]: {ctx}" for i, ctx in enumerate(contexts))

    messages = [
        SystemMessage(content="You are an evaluation assistant."),
        HumanMessage(
            content=CONTEXT_PRECISION_PROMPT.format(
                question=question,
                contexts=contexts_text,
            )
        ),
    ]

    chain = llm | StrOutputParser()
    response = await chain.ainvoke(messages)
    return _parse_score(response)


@traceable(name="compute_context_recall", run_type="chain")
async def compute_context_recall(
    config: AppConfig,
    question: str,
    contexts: list[str],
    ground_truth: str,
) -> float:
    """Compute context recall score.

    Measures whether contexts contain all information needed
    to produce the ground truth answer.

    Args:
        config: Application configuration.
        question: The user's question.
        contexts: Retrieved context passages.
        ground_truth: The expected/reference answer.

    Returns:
        Context recall score between 0 and 1.
    """
    if not contexts:
        return 0.0

    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    contexts_text = "\n---\n".join(f"[Context {i + 1}]: {ctx}" for i, ctx in enumerate(contexts))

    messages = [
        SystemMessage(content="You are an evaluation assistant."),
        HumanMessage(
            content=CONTEXT_RECALL_PROMPT.format(
                question=question,
                contexts=contexts_text,
                ground_truth=ground_truth,
            )
        ),
    ]

    chain = llm | StrOutputParser()
    response = await chain.ainvoke(messages)
    return _parse_score(response)


@traceable(name="compute_all_metrics", run_type="chain")
async def compute_all_metrics(
    config: AppConfig,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> RAGMetrics:
    """Compute all RAG evaluation metrics.

    Args:
        config: Application configuration.
        question: The user's question.
        answer: The generated answer.
        contexts: Retrieved context passages.
        ground_truth: The expected/reference answer.

    Returns:
        RAGMetrics containing all evaluation scores.
    """
    # Compute metrics (can be parallelized in future)
    faithfulness = await compute_faithfulness(config, question, answer, contexts)
    relevancy = await compute_answer_relevancy(config, question, answer)
    precision = await compute_context_precision(config, question, contexts)
    recall = await compute_context_recall(config, question, contexts, ground_truth)

    return RAGMetrics(
        faithfulness=faithfulness,
        answer_relevancy=relevancy,
        context_precision=precision,
        context_recall=recall,
    )
