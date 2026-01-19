"""Step-back prompting for query refinement.

Transforms specific questions into broader queries for better retrieval,
then uses the broader context to answer the specific question.

This module uses the centralized prompt catalog for prompt management,
enabling version control, A/B testing, and monitoring via LangSmith Hub.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from jama_mcp_server_graphrag.observability import traceable
from jama_mcp_server_graphrag.prompts import PromptName, get_prompt_sync

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)


@traceable(name="generate_stepback_query", run_type="chain")
async def generate_stepback_query(
    config: AppConfig,
    question: str,
) -> str:
    """Generate a step-back (broader) version of a specific question.

    Step-back prompting improves retrieval by:
    1. Taking a specific, detailed question
    2. Generating a broader, more general version
    3. Retrieving context for the broader question
    4. Using that context to answer the specific question

    The prompt is fetched from the centralized catalog, enabling:
    - Version control via LangSmith Hub
    - A/B testing between prompt variants
    - Performance monitoring and evaluation

    Args:
        config: Application configuration.
        question: Specific question to transform.

    Returns:
        Broader step-back question.
    """
    logger.info("Generating step-back query for: '%s'", question[:50])

    # Get prompt from catalog (uses cache if available)
    prompt_template = get_prompt_sync(PromptName.STEPBACK)

    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    # Use the prompt template from the catalog
    chain = prompt_template | llm | StrOutputParser()

    stepback_query = await chain.ainvoke({"question": question})

    stepback_query = stepback_query.strip()
    logger.info("Step-back query: '%s'", stepback_query[:50])

    return stepback_query


__all__ = ["generate_stepback_query"]
