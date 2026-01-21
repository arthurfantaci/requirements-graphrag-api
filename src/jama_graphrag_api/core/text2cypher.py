"""Text to Cypher query generation.

Converts natural language questions into Cypher queries using LLM
with few-shot examples and schema context.

This module uses the centralized prompt catalog for prompt management,
enabling version control, A/B testing, and monitoring via LangSmith Hub.

Updated Data Model (2026-01):
- Chunks linked via FROM_ARTICLE to Articles (reversed from HAS_CHUNK)
- MENTIONED_IN direction: Entity -> Chunk
- Definition nodes replace GlossaryTerm
- New media types: Image, Video, Webinar
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from jama_graphrag_api.observability import traceable
from jama_graphrag_api.prompts import PromptName, get_prompt_sync
from jama_graphrag_api.prompts.definitions import TEXT2CYPHER_EXAMPLES

if TYPE_CHECKING:
    from neo4j import Driver

    from jama_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)

# Constants
LOG_TRUNCATE_LENGTH: Final[int] = 100


@traceable(name="generate_cypher", run_type="llm")
async def generate_cypher(
    config: AppConfig,
    driver: Driver,
    question: str,
) -> str:
    """Generate a Cypher query from a natural language question.

    The prompt is fetched from the centralized catalog, enabling:
    - Version control via LangSmith Hub
    - A/B testing between prompt variants
    - Performance monitoring and evaluation

    Args:
        config: Application configuration.
        driver: Neo4j driver (for schema).
        question: Natural language question.

    Returns:
        Generated Cypher query string.
    """
    logger.info("Generating Cypher for: '%s'", question)

    # Get schema from database
    schema_info = ""
    try:
        with driver.session() as session:
            # Get node labels and counts
            labels_result = session.run(
                """
                CALL db.labels() YIELD label
                CALL {
                    WITH label
                    MATCH (n)
                    WHERE label IN labels(n)
                    RETURN count(n) AS count
                }
                RETURN label, count
                ORDER BY count DESC
                """
            )
            labels = [(r["label"], r["count"]) for r in labels_result]
            schema_info = "Node counts: " + ", ".join(f"{lbl}({cnt})" for lbl, cnt in labels[:15])
    except Exception as e:
        logger.warning("Failed to get schema: %s", e)
        schema_info = "Schema information unavailable"

    # Get prompt from catalog (uses cache if available)
    prompt_template = get_prompt_sync(PromptName.TEXT2CYPHER)

    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    # Use the prompt template from the catalog
    chain = prompt_template | llm | StrOutputParser()

    cypher = await chain.ainvoke(
        {
            "schema": schema_info,
            "examples": TEXT2CYPHER_EXAMPLES,
            "question": question,
        }
    )

    # Clean up the response
    cypher = cypher.strip()
    if cypher.startswith("```"):
        # Remove markdown code blocks
        lines = cypher.split("\n")
        cypher = "\n".join(line for line in lines if not line.startswith("```")).strip()

    if len(cypher) > LOG_TRUNCATE_LENGTH:
        truncated = cypher[:LOG_TRUNCATE_LENGTH] + "..."
    else:
        truncated = cypher
    logger.info("Generated Cypher: %s", truncated)
    return cypher


@traceable(name="text2cypher_query", run_type="chain")
async def text2cypher_query(
    config: AppConfig,
    driver: Driver,
    question: str,
    *,
    execute: bool = True,
) -> dict[str, Any]:
    """Generate and optionally execute a Cypher query from natural language.

    Args:
        config: Application configuration.
        driver: Neo4j driver.
        question: Natural language question.
        execute: Whether to execute the query (default True).

    Returns:
        Dictionary with generated query and optional results.
    """
    logger.info("Text2Cypher: question='%s', execute=%s", question, execute)

    # Generate the Cypher query
    cypher = await generate_cypher(config, driver, question)

    response: dict[str, Any] = {
        "question": question,
        "cypher": cypher,
    }

    if execute:
        # Validate query is read-only (before trying to execute)
        cypher_upper = cypher.upper()
        forbidden = ["DELETE", "MERGE", "CREATE", "SET", "REMOVE", "DROP"]
        forbidden_found = next((keyword for keyword in forbidden if keyword in cypher_upper), None)

        if forbidden_found:
            logger.warning("Query contains forbidden keyword: %s", forbidden_found)
            response["error"] = f"Query contains forbidden keyword: {forbidden_found}"
            response["results"] = []
            response["row_count"] = 0
        else:
            try:
                # Execute the query using driver session
                with driver.session() as session:
                    result = session.run(cypher)
                    results = [dict(record) for record in result]
                response["results"] = results
                response["row_count"] = len(results)
                logger.info("Query executed successfully, %d rows returned", len(results))
            except Exception as e:
                logger.warning("Query execution failed: %s", e)
                response["error"] = str(e)
                response["results"] = []
                response["row_count"] = 0

    return response


__all__ = ["generate_cypher", "text2cypher_query"]
