"""Answer critic for validating retrieval quality.

Evaluates whether retrieved context is sufficient to answer
the user's question and suggests follow-up queries if needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)

ANSWER_CRITIC_PROMPT: Final[str] = """
You are an answer quality critic for a Requirements Management knowledge graph.

Given the user's original question and the retrieved context, evaluate:

1. ANSWERABLE: Can the question be answered from the retrieved context? (yes/no)
2. CONFIDENCE: How confident are you? (0.0 - 1.0)
3. COMPLETENESS: Is the information complete or partial?
4. FOLLOWUP: If not fully answerable, what follow-up query would help?

Retrieved Context:
{context}

Original Question: {question}

Return a JSON object:
{{
    "answerable": true/false,
    "confidence": 0.0-1.0,
    "completeness": "complete" | "partial" | "insufficient",
    "missing_aspects": ["list of missing information if any"],
    "followup_query": "suggested query if needed, or null",
    "reasoning": "brief explanation"
}}
"""


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

    Args:
        config: Application configuration.
        question: Original user question.
        context: Retrieved context from the knowledge graph.

    Returns:
        CritiqueResult with evaluation metrics.
    """
    logger.info("Critiquing answer for: '%s'", question[:50])

    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    system_message = SystemMessage(
        content=ANSWER_CRITIC_PROMPT.format(context=context, question=question)
    )

    human_message = HumanMessage(
        content="Evaluate the context and return the critique as JSON."
    )

    chain = llm | StrOutputParser()
    response = await chain.ainvoke([system_message, human_message])

    # Parse JSON response
    try:
        # Clean up response if needed
        response_clean = response.strip()
        if response_clean.startswith("```"):
            lines = response_clean.split("\n")
            response_clean = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

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
