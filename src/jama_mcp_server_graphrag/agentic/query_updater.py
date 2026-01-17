"""Query updater for multi-part questions.

Updates remaining questions with context from previously answered parts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)

QUERY_UPDATE_PROMPT: Final[str] = """
You are an expert at updating questions to make them more atomic,
specific, and easier to answer.

You do this by filling in missing information in the question with
the extra information provided from previous answers.

Rules:
1. Only edit the question if needed
2. If the original question is already complete, keep it unchanged
3. Do not ask for more information than the original question
4. Only rephrase to make the question more complete with known context

Previous Answers:
{previous_answers}

Original Question: {question}

Return the updated question (just the question, no explanations):
"""


async def update_query(
    config: AppConfig,
    question: str,
    previous_answers: list[dict[str, str]],
) -> str:
    """Update a query with context from previous answers.

    For multi-part questions, this function refines subsequent queries
    by incorporating information from earlier answers, making retrieval
    more accurate and focused.

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

    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    system_message = SystemMessage(
        content=QUERY_UPDATE_PROMPT.format(previous_answers=answers_text, question=question)
    )

    human_message = HumanMessage(
        content="Return the updated question based on the context provided."
    )

    chain = llm | StrOutputParser()
    updated_query = await chain.ainvoke([system_message, human_message])

    updated_query = updated_query.strip()
    logger.info("Updated query: '%s'", updated_query[:50])

    return updated_query
