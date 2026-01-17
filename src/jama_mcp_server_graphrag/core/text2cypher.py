"""Text to Cypher query generation.

Converts natural language questions into Cypher queries using LLM
with few-shot examples and schema context.

Updated Data Model (2026-01):
- Chunks linked via FROM_ARTICLE to Articles (reversed from HAS_CHUNK)
- MENTIONED_IN direction: Entity -> Chunk
- Definition nodes replace GlossaryTerm
- New media types: Image, Video, Webinar
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from jama_mcp_server_graphrag.observability import traceable

if TYPE_CHECKING:
    from neo4j import Driver

    from jama_mcp_server_graphrag.config import AppConfig

logger = logging.getLogger(__name__)

# Constants
LOG_TRUNCATE_LENGTH: Final[int] = 100

# Few-shot examples for Text2Cypher - Updated for new schema
FEW_SHOT_EXAMPLES: Final[str] = """
Example 1:
Question: How many chapters are there?
Cypher: MATCH (c:Chapter) RETURN count(c) AS chapter_count

Example 2:
Question: Which chapter has the most articles?
Cypher: MATCH (c:Chapter)-[:CONTAINS]->(a:Article)
RETURN c.title AS chapter, count(a) AS article_count
ORDER BY article_count DESC
LIMIT 1

Example 3:
Question: List all tools mentioned in the guide
Cypher: MATCH (t:Tool)
RETURN t.name AS tool_name, t.display_name AS display_name, t.vendor AS vendor
ORDER BY t.name

Example 4:
Question: What entities are related to requirements traceability?
Cypher: MATCH (e:Concept)-[r]-(related)
WHERE toLower(e.name) CONTAINS 'traceability'
  AND NOT related:Chunk AND NOT related:Article
RETURN e.display_name AS entity, type(r) AS relationship, related.display_name AS related_entity
LIMIT 10

Example 5:
Question: How many definition terms are defined?
Cypher: MATCH (d:Definition) RETURN count(d) AS term_count

Example 6:
Question: What standards apply to automotive?
Cypher: MATCH (s:Standard)-[:APPLIES_TO]->(i:Industry)
WHERE toLower(i.name) CONTAINS 'automotive'
   OR toLower(s.display_name) CONTAINS 'automotive'
RETURN s.name AS standard, s.display_name AS display_name, s.organization AS organization

Example 7:
Question: Which articles mention ISO 26262?
Cypher: MATCH (e:Standard)-[:MENTIONED_IN]->(c:Chunk)-[:FROM_ARTICLE]->(a:Article)
WHERE toLower(e.name) CONTAINS 'iso 26262'
RETURN DISTINCT a.article_title AS article, a.url AS url

Example 8:
Question: What are the top 5 most mentioned entities?
Cypher: MATCH (entity)-[:MENTIONED_IN]->(c:Chunk)
WITH labels(entity)[0] AS entity_type, entity.display_name AS entity_name, count(c) AS mention_count
RETURN entity_type, entity_name, mention_count
ORDER BY mention_count DESC
LIMIT 5

Example 9:
Question: What challenges does requirements management address?
Cypher: MATCH (c:Concept)-[:ADDRESSES]->(ch:Challenge)
WHERE toLower(c.name) CONTAINS 'requirement'
RETURN c.display_name AS concept, ch.display_name AS challenge
LIMIT 10

Example 10:
Question: What images are in article about traceability?
Cypher: MATCH (a:Article)-[:HAS_IMAGE]->(img:Image)
WHERE toLower(a.article_title) CONTAINS 'traceability'
RETURN a.article_title AS article, img.alt_text AS image_description, img.url AS image_url
"""

SYSTEM_PROMPT: Final[str] = """You are a Cypher query expert for a Neo4j knowledge graph
about requirements management from the Jama Software guide.

Your task is to convert natural language questions into valid Cypher queries.

Schema Information:
{schema}

## Node Labels

### Content Hierarchy
- Chapter (15) - chapter_number, title, overview_url, article_count
- Article (103) - article_id, article_title, url, chapter_number, chapter_title
- Chunk (2159) - text, embedding, index

### Domain Entities
- Concept (1523) - name, display_name, definition
- Challenge (839) - name, display_name
- Artifact (601) - name, display_name, artifact_type
- Bestpractice (330) - name, display_name
- Processstage (285) - name, display_name, sequence
- Role (181) - name, display_name, responsibilities
- Tool (159) - name, display_name, vendor, category
- Standard (123) - name, display_name, organization, domain
- Methodology (30) - name, display_name
- Industry (18) - name, display_name, regulated

### Reference Data
- Definition (134) - term, definition, url, term_id

### Media
- Image (163) - url, alt_text, context, source_article_id
- Video (1) - title, url, platform, video_id, embed_url
- Webinar (38) - title, url, description, thumbnail_url

## Key Relationships
- (Chunk)-[:FROM_ARTICLE]->(Article) - Chunk belongs to article
- (Chunk)-[:NEXT_CHUNK]->(Chunk) - Sequential chunk ordering
- (Entity)-[:MENTIONED_IN]->(Chunk) - Entity mentioned in chunk (Entity points TO Chunk)
- (Article)-[:HAS_IMAGE]->(Image) - Article contains image
- (Article)-[:HAS_VIDEO]->(Video) - Article contains video
- (Article)-[:HAS_WEBINAR]->(Webinar) - Article references webinar
- (Article)-[:REFERENCES]->(Article) - Cross-references between articles
- (Chapter)-[:CONTAINS]->(Article) - Chapter contains articles
- (Concept)-[:ADDRESSES]->(Challenge) - Concept addresses challenge
- (Concept)-[:REQUIRES]->(Concept|Artifact) - Dependencies
- (Concept)-[:COMPONENT_OF]->(Concept) - Part-of relationships
- (Concept)-[:RELATED_TO]->(Concept) - General relationships
- (Concept)-[:ALTERNATIVE_TO]->(Concept) - Alternative approaches
- (Concept)-[:PREREQUISITE_FOR]->(Concept) - Prerequisites
- (Standard)-[:DEFINES]->(Concept|Artifact) - Standard defines entities
- (Standard)-[:APPLIES_TO]->(Industry) - Standard applies to industry
- (Bestpractice)-[:APPLIES_TO]->(Processstage) - Best practice for stage
- (Role)-[:PRODUCES]->(Artifact) - Role produces artifacts
- (Role)-[:USED_BY]->(Artifact|Tool) - Role uses tools/artifacts
- (Tool)-[:ADDRESSES]->(Challenge) - Tool addresses challenges

Few-shot Examples:
{examples}

IMPORTANT RULES:
1. Return ONLY the Cypher query, no explanations
2. Use parameterized queries when possible (e.g., $name, $limit)
3. Always limit results (default LIMIT 10) to prevent large result sets
4. Use toLower() for case-insensitive string matching
5. Return meaningful aliases (AS column_name)
6. Never use DELETE, MERGE, CREATE, or SET - only read queries
7. Remember: MENTIONED_IN direction is Entity -> Chunk, not Chunk -> Entity
8. Use FROM_ARTICLE to get from Chunk to Article, not HAS_CHUNK
9. Prefer display_name for human-readable output
"""


@traceable(name="generate_cypher", run_type="llm")
async def generate_cypher(
    config: AppConfig,
    driver: Driver,
    question: str,
) -> str:
    """Generate a Cypher query from a natural language question.

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

    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=config.openai_api_key,
    )

    system_message = SystemMessage(
        content=SYSTEM_PROMPT.format(
            schema=schema_info,
            examples=FEW_SHOT_EXAMPLES,
        )
    )

    human_message = HumanMessage(content=f"Generate a Cypher query for this question: {question}")

    chain = llm | StrOutputParser()
    cypher = await chain.ainvoke([system_message, human_message])

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
