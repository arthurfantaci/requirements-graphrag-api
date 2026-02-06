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

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Final

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langsmith import get_current_run_tree
from neo4j import Query

from requirements_graphrag_api.observability import traceable_safe
from requirements_graphrag_api.prompts import PromptName, get_prompt_sync
from requirements_graphrag_api.prompts.definitions import TEXT2CYPHER_EXAMPLES

if TYPE_CHECKING:
    from neo4j import Driver

    from requirements_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)

# Constants
LOG_TRUNCATE_LENGTH: Final[int] = 100

# Valid Cypher query starters (read-only operations)
CYPHER_STARTERS: Final[tuple[str, ...]] = (
    "MATCH",
    "OPTIONAL",
    "RETURN",
    "WITH",
    "UNWIND",
    "CALL",
    "EXPLAIN",
    "PROFILE",
    "USE",
)


@traceable_safe(name="generate_cypher", run_type="llm")
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


def _validate_cypher(cypher: str) -> str | None:
    """Validate generated Cypher is safe to execute. Returns error message or None."""
    first_word = cypher.strip().split()[0].upper() if cypher.strip() else ""
    if first_word not in CYPHER_STARTERS:
        return (
            "This question cannot be answered with a database query. "
            "Try rephrasing as a requirements management question."
        )

    cypher_upper = cypher.upper()
    forbidden = ["DELETE", "MERGE", "CREATE", "SET", "REMOVE", "DROP"]
    forbidden_found = next((k for k in forbidden if k in cypher_upper), None)
    if forbidden_found:
        return f"Query contains forbidden keyword: {forbidden_found}"

    return None


def _execute_cypher(
    driver: Driver,
    cypher: str,
    *,
    timeout: float,
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute Cypher query with timeout via Query object. Returns (results, error_or_none)."""
    try:
        with driver.session() as session:
            result = session.run(Query(cypher, timeout=timeout))
            return [dict(record) for record in result], None
    except Exception as e:
        return [], str(e)


@traceable_safe(name="text2cypher_query", run_type="chain")
async def text2cypher_query(
    config: AppConfig,
    driver: Driver,
    question: str,
    *,
    execute: bool = True,
    langsmith_extra: dict[str, Any] | None = None,
    llm_timeout: float = 20.0,
    neo4j_timeout: float = 15.0,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Generate and optionally execute a Cypher query from natural language.

    Args:
        config: Application configuration.
        driver: Neo4j driver.
        question: Natural language question.
        execute: Whether to execute the query (default True).
        langsmith_extra: Optional LangSmith metadata for thread grouping.
        llm_timeout: Timeout in seconds for the LLM call (default 20s).
        neo4j_timeout: Timeout in seconds for Neo4j query execution (default 15s).
        max_retries: Number of retries on LLM timeout (default 1).

    Returns:
        Dictionary with generated query and optional results.
    """
    _ = langsmith_extra
    logger.info("Text2Cypher: question='%s', execute=%s", question, execute)

    for attempt in range(max_retries + 1):
        try:
            async with asyncio.timeout(llm_timeout):
                cypher = await generate_cypher(config, driver, question)
        except TimeoutError:
            if attempt < max_retries:
                logger.warning("Text2Cypher timeout (attempt %d/%d)", attempt + 1, max_retries + 1)
                continue
            logger.error("Text2Cypher timeout exhausted after %d attempts", max_retries + 1)
            error_response: dict[str, Any] = {
                "question": question,
                "cypher": "",
                "results": [],
                "row_count": 0,
                "error": (
                    f"Query timed out after {max_retries + 1} attempts. Try a simpler question."
                ),
            }
            try:
                run_tree = get_current_run_tree()
                if run_tree:
                    error_response["run_id"] = str(run_tree.id)
            except Exception:
                logger.debug("Could not get run_id for timeout error")
            return error_response

        # LLM succeeded — no retry for validation/execution errors
        response: dict[str, Any] = {"question": question, "cypher": cypher}

        if execute:
            validation_error = _validate_cypher(cypher)
            if validation_error:
                logger.warning("Cypher validation failed: %s", validation_error)
                response["error"] = validation_error
                response["results"] = []
                response["row_count"] = 0
                break

            results, exec_error = _execute_cypher(driver, cypher, timeout=neo4j_timeout)
            if exec_error:
                logger.warning("Query execution failed: %s", exec_error)
                response["error"] = exec_error
                response["results"] = []
                response["row_count"] = 0
            else:
                response["results"] = results
                response["row_count"] = len(results)
                if len(results) == 0:
                    response["message"] = (
                        "No results found for this query. Try rephrasing "
                        "or asking an explanatory question instead."
                    )
                logger.info("Query executed successfully, %d rows returned", len(results))

        break  # Success or non-retryable error

    # Capture run_id (success path)
    try:
        run_tree = get_current_run_tree()
        if run_tree:
            response["run_id"] = str(run_tree.id)
    except Exception:
        logger.debug("Could not get run_id - tracing may be disabled")

    return response


__all__ = ["_execute_cypher", "_validate_cypher", "generate_cypher", "text2cypher_query"]
