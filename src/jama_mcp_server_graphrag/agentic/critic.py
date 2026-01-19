"""Answer critic for validating retrieval quality.

Evaluates whether retrieved context is sufficient to answer
the user's question and suggests follow-up queries if needed.

This module uses the centralized prompt catalog for prompt management,
enabling version control, A/B testing, and monitoring via LangSmith Hub.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from jama_mcp_server_graphrag.observability import traceable
from jama_mcp_server_graphrag.prompts import PromptName, get_prompt_sync

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """Result of critiquing the answer quality."""

    answerable: bool
    confidence: float
    completeness: Literal["complete", "partial", "insufficient"]
    missing_aspects: list[str] = field(default_factory=list)
    followup_query: str | None = None
    reasoning: str = ""
    raw_response: str = ""


@traceable(name="critique_answer", run_type="chain")
async def critique_answer(
    config: AppConfig,
    question: str,
    context: str,
) -> CritiqueResult:
    """Critique whether the retrieved context can answer the question.

    This is used to:
    1. Validate retrieval quality before generating an answer
    2. Identify missing information for follow-up queries
    3. Provide confidence scores for user-facing responses

    The prompt is fetched from the centralized catalog, enabling:
    - Version control via LangSmith Hub
    - A/B testing between prompt variants
    - Performance monitoring and evaluation

    Args:
        config: Application configuration.
        question: Original user question.
        context: Retrieved context from the knowledge graph.

    Returns:
        CritiqueResult with evaluation metrics.
    """
    logger.info("Critiquing answer for: '%s'", question[:50])

    # Get prompt from catalog (uses cache if available)
    prompt_template = get_prompt_sync(PromptName.CRITIC)

    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    # Use the prompt template from the catalog
    chain = prompt_template | llm | StrOutputParser()

    response = await chain.ainvoke({"context": context, "question": question})

    # Parse JSON response
    try:
        # Clean up response if needed
        response_clean = response.strip()
        if response_clean.startswith("```"):
            lines = response_clean.split("\n")
            response_clean = "\n".join(line for line in lines if not line.startswith("```")).strip()

        data = json.loads(response_clean)

        result = CritiqueResult(
            answerable=data.get("answerable", False),
            confidence=float(data.get("confidence", 0.0)),
            completeness=data.get("completeness", "insufficient"),
            missing_aspects=data.get("missing_aspects", []),
            followup_query=data.get("followup_query"),
            reasoning=data.get("reasoning", ""),
            raw_response=response,
        )
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse critique response: %s", e)
        # Default to conservative estimate
        result = CritiqueResult(
            answerable=True,  # Allow generation but with low confidence
            confidence=0.5,
            completeness="partial",
            reasoning=f"Failed to parse critique: {e}",
            raw_response=response,
        )

    logger.info(
        "Critique result: answerable=%s, confidence=%.2f, completeness=%s",
        result.answerable,
        result.confidence,
        result.completeness,
    )
    return result


__all__ = ["CritiqueResult", "critique_answer"]
