"""Step-back prompting for query refinement.

Transforms specific questions into broader queries for better retrieval,
then uses the broader context to answer the specific question.
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

STEPBACK_SYSTEM_PROMPT: Final[str] = """
You are an expert at requirements management. Your task is to step back
and paraphrase a question to a more generic step-back question, which
is easier to answer.

The step-back question should:
1. Remove overly specific details while preserving the core topic
2. Broaden the scope to capture more relevant context
3. Use terminology that matches the knowledge base

Examples:
Input: "What ISO standard applies to automotive functional safety for ASIL-D?"
Output: "What are the automotive industry safety standards?"

Input: "How do I implement bidirectional traceability in Jama Connect for FDA compliance?"
Output: "What are the best practices for requirements traceability?"

Input: "What specific challenges does a systems engineer face during the verification phase?"
Output: "What are the challenges in requirements verification?"

Input: "Which V-Model phase produces the software requirements specification?"
Output: "What artifacts are produced during requirements development?"

Return ONLY the step-back question, no explanations.
"""


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

    Args:
        config: Application configuration.
        question: Specific question to transform.

    Returns:
        Broader step-back question.
    """
    logger.info("Generating step-back query for: '%s'", question[:50])

    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    system_message = SystemMessage(content=STEPBACK_SYSTEM_PROMPT)
    human_message = HumanMessage(content=question)

    chain = llm | StrOutputParser()
    stepback_query = await chain.ainvoke([system_message, human_message])

    stepback_query = stepback_query.strip()
    logger.info("Step-back query: '%s'", stepback_query[:50])

    return stepback_query
