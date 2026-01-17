"""Neo4j client wrapper implementing driver best practices.

This module encapsulates Neo4j driver best practices from:
https://neo4j.com/blog/developer/neo4j-driver-best-practices/

Key Best Practices Implemented:
1. Single driver instance (expensive to create, reuse across requests)
2. Connectivity verification at startup (fail fast on bad config)
3. Explicit transaction functions (proper cluster routing)
4. Query parameters (security and performance)
5. Result processing within transaction scope
6. Connection pool sizing for serverless environments
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from jama_mcp_server_graphrag.exceptions import Neo4jConnectionError

if TYPE_CHECKING:
    from neo4j import Driver

    from jama_mcp_server_graphrag.config import AppConfig

logger: logging.Logger = logging.getLogger(__name__)


def create_driver(config: AppConfig) -> Driver:
    """Create a Neo4j driver instance with best practices.

    Best Practices Applied:
    - Single driver instance (created once, reused)
    - Connectivity verified immediately
    - Connection pool sized for serverless

    Args:
        config: Application configuration with Neo4j settings.

    Returns:
        Configured and verified Neo4j Driver instance.

    Raises:
        Neo4jConnectionError: If connection or authentication fails.
    """
    logger.info(
        "Creating Neo4j driver for %s (pool_size=%d)",
        config.neo4j_uri.split("@")[-1],  # Hide credentials in logs
        config.neo4j_max_connection_pool_size,
    )

    try:
        driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_username, config.neo4j_password),
            # Serverless optimization: smaller pool = faster cold starts
            max_connection_pool_size=config.neo4j_max_connection_pool_size,
            connection_acquisition_timeout=config.neo4j_connection_acquisition_timeout,
        )

        # Best Practice: Verify connectivity immediately
        # Catches bad URI, credentials, or network issues at startup
        driver.verify_connectivity()
    except AuthError as e:
        logger.error("Neo4j authentication failed - check credentials")
        raise Neo4jConnectionError(f"Authentication failed: {e}") from e
    except ServiceUnavailable as e:
        logger.error("Neo4j service unavailable - check URI and network")
        raise Neo4jConnectionError(f"Service unavailable: {e}") from e
    except Exception as e:
        logger.exception("Failed to create Neo4j driver")
        raise Neo4jConnectionError(f"Driver creation failed: {e}") from e
    else:
        logger.info("Neo4j connectivity verified successfully")
        return driver


def execute_read_query(
    driver: Driver,
    query: str,
    parameters: dict[str, Any] | None = None,
    database: str = "neo4j",
) -> list[dict[str, Any]]:
    """Execute a read query using explicit transaction function.

    Best Practices Applied:
    - Uses execute_read() for proper cluster routing (reads go to any member)
    - Query parameters for security and query caching
    - Results processed within transaction scope

    Args:
        driver: Neo4j driver instance.
        query: Cypher query string with $parameter placeholders.
        parameters: Query parameters (use these, NEVER string concatenation).
        database: Target database name.

    Returns:
        List of result records as dictionaries.

    Example:
        >>> results = execute_read_query(
        ...     driver,
        ...     "MATCH (e:Entity) WHERE e.name = $name RETURN e.definition",
        ...     {"name": "requirements traceability"}
        ... )
    """

    def _execute(tx: Any, query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        result = tx.run(query, params or {})
        # Best Practice: Process results within transaction scope
        return [record.data() for record in result]

    with driver.session(database=database) as session:
        return session.execute_read(_execute, query, parameters)


def execute_write_query(
    driver: Driver,
    query: str,
    parameters: dict[str, Any] | None = None,
    database: str = "neo4j",
) -> list[dict[str, Any]]:
    """Execute a write query using explicit transaction function.

    Best Practices Applied:
    - Uses execute_write() for proper cluster routing (writes go to leader)
    - Query parameters for security
    - Results processed within transaction scope

    Args:
        driver: Neo4j driver instance.
        query: Cypher query string with $parameter placeholders.
        parameters: Query parameters.
        database: Target database name.

    Returns:
        List of result records as dictionaries.
    """

    def _execute(tx: Any, query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        result = tx.run(query, params or {})
        return [record.data() for record in result]

    with driver.session(database=database) as session:
        return session.execute_write(_execute, query, parameters)


def execute_read_with_bookmark(
    driver: Driver,
    query: str,
    parameters: dict[str, Any] | None = None,
    bookmarks: list[Any] | None = None,
    database: str = "neo4j",
) -> tuple[list[dict[str, Any]], Any]:
    """Execute a read query with bookmark for causal consistency.

    Use this when you need to guarantee reading your own writes
    across different sessions or processes.

    Args:
        driver: Neo4j driver instance.
        query: Cypher query string.
        parameters: Query parameters.
        bookmarks: Bookmarks from previous writes to ensure consistency.
        database: Target database name.

    Returns:
        Tuple of (results, new_bookmark) for chaining.
    """

    def _execute(tx: Any, query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        result = tx.run(query, params or {})
        return [record.data() for record in result]

    with driver.session(database=database, bookmarks=bookmarks) as session:
        results = session.execute_read(_execute, query, parameters)
        return results, session.last_bookmarks()
