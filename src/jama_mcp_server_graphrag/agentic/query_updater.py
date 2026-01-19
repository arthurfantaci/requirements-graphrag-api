"""Query updater for multi-part questions.

Updates remaining questions with context from previously answered parts.

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


@traceable(name="update_query", run_type="chain")
async def update_query(
    config: AppConfig,
    question: str,
    previous_answers: list[dict[str, str]],
) -> str:
    """Update a query with context from previous answers.

    For multi-part questions, this function refines subsequent queries
    by incorporating information from earlier answers, making retrieval
    more accurate and focused.

    The prompt is fetched from the centralized catalog, enabling:
    - Version control via LangSmith Hub
    - A/B testing between prompt variants
    - Performance monitoring and evaluation

    Args:
        config: Application configuration.
        question: Original question to update.
        previous_answers: List of previous Q&A pairs, each with
            'question' and 'answer' keys.

    Returns:
        Updated question with incorporated context.
    """
    if not previous_answers:
        logger.info("No previous answers, returning original question")
        return question

    logger.info("Updating query with %d previous answers", len(previous_answers))

    # Format previous answers
    answers_text = "\n".join(f"Q: {qa['question']}\nA: {qa['answer']}" for qa in previous_answers)

    # Get prompt from catalog (uses cache if available)
    prompt_template = get_prompt_sync(PromptName.QUERY_UPDATER)

    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    # Use the prompt template from the catalog
    chain = prompt_template | llm | StrOutputParser()

    updated_query = await chain.ainvoke({
        "previous_answers": answers_text,
        "question": question,
    })

    updated_query = updated_query.strip()
    logger.info("Updated query: '%s'", updated_query[:50])

    return updated_query


__all__ = ["update_query"]
