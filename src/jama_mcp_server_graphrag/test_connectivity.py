"""Neo4j connectivity test script.

Run this script to verify your Neo4j connection is working correctly.

Usage:
    uv run python -m jama_mcp_server_graphrag.test_connectivity
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from jama_mcp_server_graphrag.config import get_config
from jama_mcp_server_graphrag.neo4j_client import create_driver, execute_read_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60


def main() -> int:
    """Test Neo4j connectivity and basic queries."""
    # Load environment variables
    load_dotenv()

    logger.info(SEPARATOR)
    logger.info("Neo4j Connectivity Test")
    logger.info(SEPARATOR)

    # Step 1: Load configuration
    logger.info("\n[1/4] Loading configuration...")
    try:
        config = get_config()
        logger.info("  URI: %s", config.neo4j_uri.split("@")[-1])
        logger.info("  Database: %s", config.neo4j_database)
        logger.info("  Pool Size: %d", config.neo4j_max_connection_pool_size)
    except Exception as e:
        logger.error("  FAILED: %s", e)
        return 1

    # Step 2: Create driver and verify connectivity
    logger.info("\n[2/4] Creating driver and verifying connectivity...")
    try:
        driver = create_driver(config)
        logger.info("  SUCCESS: Driver created and connectivity verified")
    except Exception as e:
        logger.error("  FAILED: %s", e)
        return 1

    # Step 3: Test basic query - count nodes
    logger.info("\n[3/4] Testing basic query (node counts)...")
    try:
        # Count nodes by label
        query = """
        MATCH (n)
        WITH labels(n) as labels
        UNWIND labels as label
        RETURN label, count(*) as count
        ORDER BY count DESC
        LIMIT 10
        """
        results = execute_read_query(driver, query, database=config.neo4j_database)

        if results:
            logger.info("  Node counts by label:")
            for row in results:
                logger.info("    %s: %d", row["label"], row["count"])
        else:
            logger.warning("  No nodes found in database")
    except Exception as e:
        logger.error("  FAILED: %s", e)
        driver.close()
        return 1

    # Step 4: Test vector index exists
    logger.info("\n[4/4] Checking vector index...")
    try:
        query = """
        SHOW INDEXES
        WHERE type = 'VECTOR'
        """
        results = execute_read_query(driver, query, database=config.neo4j_database)

        if results:
            logger.info("  Vector indexes found:")
            for idx in results:
                logger.info("    - %s (state: %s)", idx.get("name"), idx.get("state"))
        else:
            logger.warning("  No vector indexes found - you may need to create one")
    except Exception as e:
        logger.warning("  Could not check indexes: %s", e)

    # Cleanup
    driver.close()
    logger.info("\n%s", SEPARATOR)
    logger.info("All connectivity tests PASSED")
    logger.info(SEPARATOR)

    return 0


if __name__ == "__main__":
    sys.exit(main())
