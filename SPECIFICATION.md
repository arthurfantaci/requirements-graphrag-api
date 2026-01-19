# Jama MCP Server GraphRAG Development Plan

## CLAUDE.md - Instructions for Claude Code

This document provides comprehensive guidance for Claude Code to develop a production-ready GraphRAG backend that serves as both an MCP Server for Claude Desktop and a REST API backend for a React chatbot application, using Neo4j, LangChain, LangGraph, FastMCP, and FastAPI.

---

## Project Overview

### Purpose

Build a professional-grade GraphRAG backend for the **Jama Software "Essential Guide to Requirements Management and Traceability"** knowledge base. The application serves two purposes:

1. **MCP Server**: Exposes GraphRAG tools to Claude Desktop via Model Context Protocol
2. **REST API Backend**: Powers a React frontend chatbot for querying requirements management content

Both interfaces share the same core GraphRAG logic, Neo4j connection, and retrieval workflows.

### Architecture Overview

```
┌─────────────────┐     ┌─────────────────────────────────────┐     ┌───────────────┐
│  Claude Desktop │────▶│                                     │     │               │
│  (MCP Client)   │     │   jama-mcp-server-graphrag          │     │  Neo4j AuraDB │
└─────────────────┘     │                                     │────▶│  (jama-guide  │
                        │  ┌───────────┐  ┌────────────────┐  │     │   -to-rm)     │
┌─────────────────┐     │  │ MCP Tools │  │ Core GraphRAG  │  │     │               │
│  React Frontend │────▶│  ├───────────┤  │  - Retrieval   │  │     └───────────────┘
│  (Chatbot UI)   │     │  │ REST API  │──│  - Text2Cypher │  │
└─────────────────┘     │  └───────────┘  │  - Workflows   │  │
                        │                 └────────────────┘  │
                        └─────────────────────────────────────┘
```

### Capabilities

The server will:

1. Connect to an existing Neo4j graph database with vector indexes
2. Provide semantic search via vector similarity
3. Enable **hybrid search** combining vector and full-text retrieval
4. Enable graph-enriched retrieval combining vector search with Cypher traversals
5. Support natural language to Cypher query generation with **few-shot examples**
6. Implement **agentic RAG patterns** with retriever routing and answer validation
7. Expose schema exploration and metadata tools
8. Provide **evaluation tools** using RAGAS metrics
9. Implement production patterns for testability, observability, and maintainability
10. Support **Docker containerization** and **CI/CD deployment to Vercel**
11. **Expose REST API endpoints** for React frontend integration

### Target Audience

This project is a portfolio demonstration of skills expected of a professional Agentic AI developer delivering full-stack, production-ready applications.

### Reference Implementation

This development plan incorporates best practices from:
- **"Essential GraphRAG" by Tomaž Bratanič and Oskar Hane (Manning, 2025)**
- Neo4j GraphRAG patterns and LangChain integration
- FastMCP MCP server framework

---

## Knowledge Graph Data Model

### Domain: Requirements Management

The Neo4j database contains a **Requirements Management Knowledge Graph** sourced from Jama Software's requirements management guide. It covers requirements engineering best practices, industry standards, tools, methodologies, and domain-specific applications across multiple regulated industries.

### Content Structure

```
Chapter (15 nodes)
  └── Article (103 nodes)
        ├── article_id, article_title, url, chapter_number, chapter_title
        ├── Image (163 nodes) via HAS_IMAGE
        ├── Video (1 node) via HAS_VIDEO
        └── Webinar (38 nodes) via HAS_WEBINAR

Chunk (2,159 nodes)
  ├── text, embedding (1536 dim), index
  ├── FROM_ARTICLE → Article
  ├── NEXT_CHUNK → Chunk (sequential ordering)
  └── Entity nodes → MENTIONED_IN → Chunk
```

### Chapters (Topics Covered)

| Chapter | Topic | Articles |
|---------|-------|----------|
| 1 | Requirements Management | 9 |
| 2 | Writing Requirements | 13 |
| 3 | Requirements Gathering and Management Processes | 10 |
| 4 | Requirements Traceability | 18 |
| 5 | Requirements Management Tools and Software | 8 |
| 6 | Requirements Validation and Verification | 3 |
| 7 | Meeting Regulatory Compliance and Industry Standards | 7 |
| 8 | Systems Engineering | 7 |
| 9 | Automotive Development | 3 |
| 10 | Medical Device & Life Sciences Development | 10 |
| 11 | Aerospace & Defense Development | 3 |
| 12 | Architecture, Engineering, and Construction (AEC) | 2 |
| 13 | Industrial Manufacturing, Automation & Robotics | 5 |
| 14 | Semiconductor Development | 3 |
| 15 | AI in Product Development | 2 |

### Entity Types (4,089 total entities)

Entities use multi-labeling (e.g., a node can be both `Entity` and `Concept`):

| Entity Type | Count | Description |
|-------------|-------|-------------|
| **Concept** | 1,523 | Abstract concepts (requirements traceability, verification, etc.) |
| **Challenge** | 839 | Problems and challenges in requirements management |
| **Artifact** | 601 | Deliverables, documents, outputs |
| **Bestpractice** | 330 | Best practices and recommendations |
| **Processstage** | 285 | Process steps and phases |
| **Role** | 181 | Job roles (Product Manager, Systems Engineer, etc.) |
| **Tool** | 159 | Software tools (Jama Connect, Excel, DOORS, etc.) |
| **Standard** | 123 | Industry standards (ISO 26262, FDA, DO-178C, INCOSE, etc.) |
| **Methodology** | 30 | Methods (Agile, V-Model, Waterfall, etc.) |
| **Industry** | 18 | Industry sectors (Automotive, Medical, Aerospace, etc.) |

### Definition Nodes (134 terms)

Glossary definitions for domain-specific terminology:
- Properties: `term`, `definition`, `url`, `term_id`
- Examples: Requirement, Traceability, Baseline, Backlog, Acceptance Criteria, Atomic Requirements, Application Lifecycle Management (ALM), Build Verification Test, etc.

### Media Nodes (New)

| Node Type | Count | Properties |
|-----------|-------|------------|
| **Image** | 163 | url, alt_text, context, source_article_id |
| **Video** | 1 | title, url, platform, video_id, embed_url |
| **Webinar** | 38 | title, url, description, thumbnail_url |

### Neo4j Indexes

| Index Name | Type | Target | Properties |
|------------|------|--------|------------|
| `chunk_embeddings` | VECTOR | Chunk | embedding (1536 dim, COSINE) |
| `entity_fulltext` | FULLTEXT | Entity | name, definition |
| `definition_fulltext` | FULLTEXT | Definition | term, definition |

### Key Relationships

| Relationship | Count | Pattern | Properties |
|--------------|-------|---------|------------|
| `MENTIONED_IN` | 8,524 | Entity → Chunk | - |
| `FROM_ARTICLE` | 2,159 | Chunk → Article | - |
| `NEXT_CHUNK` | 2,056 | Chunk → Chunk | - |
| `RELATED_TO` | 807 | Entity → Entity | relationship_nature (optional) |
| `ADDRESSES` | 703 | Concept → Challenge | effectiveness (optional) |
| `REQUIRES` | 434 | Entity → Entity/Artifact | - |
| `COMPONENT_OF` | 283 | Entity → Entity | - |
| `HAS_IMAGE` | 163 | Article → Image | - |
| `USED_BY` | 132 | Role → Artifact/Tool | - |
| `APPLIES_TO` | 139 | Standard/Bestpractice → Industry/Processstage | - |
| `CONTAINS` | 103 | Chapter → Article | - |
| `PRODUCES` | 97 | Processstage/Role → Artifact | - |
| `DEFINES` | 77 | Standard → Concept/Artifact | - |
| `HAS_WEBINAR` | 38 | Article → Webinar | - |
| `REFERENCES` | 29 | Article → Article | - |
| `HAS_VIDEO` | 1 | Article → Video | - |

### Important Entity Properties

```python
# Entity node properties
entity_properties = {
    "name": str,           # Entity name (e.g., "requirements traceability")
    "entity_type": list,   # Can be multiple (e.g., ["concept", "best_practice"])
    "definition": str,     # Definition of the entity
    "benefit": str | list, # Benefits of the entity
    "impact": str,         # Impact description
    "confidence": str,     # Extraction confidence
    "source_text": str,    # Original text
    "source_article_id": str | list,  # Source article(s)
    "organization": str,   # For Standards (e.g., "ISO", "FDA")
    "vendor": str,         # For Tools (e.g., "Jama Software")
}

# Relationship properties (on semantic relationships)
relationship_properties = {
    "confidence": float,   # 0.0 - 1.0 confidence score
    "evidence": str,       # Supporting text evidence
    "context": str,        # Additional context
}
```

### Sample Data Patterns

**Entity with Rich Context:**
```json
{
  "name": "requirements traceability",
  "entity_type": ["concept", "best_practice", "challenge"],
  "definition": "an essential part of requirements management",
  "benefit": ["ensures development efficiency", "provides proof of requirement coverage"],
  "related_to": [
    {"name": "Simplified Regulatory Compliance", "evidence": "Easily demonstrate..."},
    {"name": "Improved Impact Analysis", "evidence": "Quickly understand..."}
  ],
  "component_of": ["requirements management", "requirements management process"],
  "mentioned_in_chunks": 61
}
```

**Industry Standards:**
- ISO 26262 (Automotive functional safety)
- FDA regulations (Medical devices)
- DO-178C (Aerospace software)
- INCOSE Guide for Writing Requirements
- MIL-STD-961E (Defense)

**Tools Referenced:**
- Jama Connect (Primary - vendor's product)
- Excel, Word (Traditional approaches)
- DOORS (IBM)
- Requirements management tools (generic)

---

## Domain-Specific Cypher Queries

### Hybrid Search Query (Essential GraphRAG Ch. 2)

Combines vector similarity search with full-text search for improved retrieval accuracy:

```cypher
// Hybrid search combining vector and keyword matching
// From Essential GraphRAG Chapter 2
CALL {
    // Vector similarity search with score normalization
    CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
    YIELD node, score
    WITH collect({node: node, score: score}) AS nodes, max(score) AS maxScore
    UNWIND nodes AS n
    RETURN n.node AS node, (n.score / maxScore) AS score, 'vector' AS source
    
    UNION
    
    // Full-text search with score normalization
    CALL db.index.fulltext.queryNodes('chunk_fulltext', $query, {limit: $k})
    YIELD node, score
    WITH collect({node: node, score: score}) AS nodes, max(score) AS maxScore
    UNWIND nodes AS n
    RETURN n.node AS node, (n.score / maxScore) AS score, 'fulltext' AS source
}
// Deduplicate and combine scores
WITH node, max(score) AS score, collect(source) AS sources
ORDER BY score DESC
LIMIT $k
RETURN node, score, sources
```

### Graph-Enriched Retrieval Query

Use this pattern with `neo4j-graphrag VectorRetriever` to enrich vector search results with graph context:

```cypher
// Graph-enriched retrieval pattern (Updated 2026-01)
// Note: MENTIONED_IN direction is Entity → Chunk (not Chunk → Entity)
// Note: FROM_ARTICLE direction is Chunk → Article (not Article → Chunk)

// After vector search returns chunk IDs, enrich with context:
MATCH (chunk:Chunk)-[:FROM_ARTICLE]->(article:Article)
WHERE elementId(chunk) IN $chunk_ids

// Get entities mentioned in these chunks (Entity points TO Chunk)
OPTIONAL MATCH (entity)-[:MENTIONED_IN]->(chunk)
WHERE NOT entity:Chunk AND NOT entity:Article

// Get related entities (one hop)
OPTIONAL MATCH (entity)-[rel:RELATED_TO|COMPONENT_OF|ADDRESSES]->(related)
WHERE NOT related:Chunk AND NOT related:Article

RETURN
    chunk.text AS text,
    chunk.index AS chunk_index,
    {
        article_title: article.article_title,
        article_url: article.url,
        chapter_title: article.chapter_title,
        entities: collect(DISTINCT {
            name: entity.name,
            type: labels(entity)[0],
            display_name: entity.display_name,
            definition: entity.definition
        })[0..10],
        related_concepts: collect(DISTINCT {
            name: related.name,
            relationship: type(rel),
            display_name: related.display_name
        })[0..5]
    } AS metadata
```

### Parent Document Retrieval Query (Essential GraphRAG Ch. 3)

Matches on smaller chunks but returns full parent context for richer LLM responses:

```cypher
// Parent document retrieval pattern (Updated 2026-01)
// Note: FROM_ARTICLE direction is Chunk → Article
CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
YIELD node AS chunk, score

// Get parent article with full context
MATCH (chunk)-[:FROM_ARTICLE]->(article:Article)
MATCH (chapter:Chapter)-[:CONTAINS]->(article)

// Collect all chunks from matched articles for full context
WITH DISTINCT article, chapter, max(score) AS best_score
ORDER BY best_score DESC
LIMIT 3

// Get all chunks for this article using reversed pattern
MATCH (all_chunks:Chunk)-[:FROM_ARTICLE]->(article)

RETURN
    article.article_title AS title,
    article.url AS url,
    chapter.title AS chapter,
    best_score AS relevance,
    collect(all_chunks.text ORDER BY all_chunks.index)[0..5] AS chunk_texts
```

### Entity-Centric Search Query

Find entities and their relationships for concept exploration:

```cypher
// Search entities by name using fulltext index (Updated 2026-01)
// Note: MENTIONED_IN direction is Entity → Chunk
CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
YIELD node AS entity, score
WHERE score > 0.5

// Get entity relationships
OPTIONAL MATCH (entity)-[r:RELATED_TO|COMPONENT_OF|ADDRESSES|REQUIRES]->(related)
WHERE NOT related:Chunk AND NOT related:Article

// Get chunks where this entity is mentioned (Entity points TO Chunk)
OPTIONAL MATCH (entity)-[:MENTIONED_IN]->(chunk:Chunk)

RETURN
    entity.name AS name,
    labels(entity) AS types,
    entity.display_name AS display_name,
    entity.definition AS definition,
    score,
    collect(DISTINCT {
        related_name: related.name,
        relationship: type(r),
        display_name: related.display_name
    })[0..5] AS relationships,
    count(DISTINCT chunk) AS mention_count
ORDER BY score DESC
LIMIT 10
```

### Standards and Compliance Query

Query for regulatory standards and their applications:

```cypher
// Find standards applicable to an industry
MATCH (s:Standard)
WHERE s.name CONTAINS $industry OR s.organization CONTAINS $industry
OPTIONAL MATCH (s)-[:APPLIES_TO]->(target)
OPTIONAL MATCH (s)-[:DEFINES]->(defined)
OPTIONAL MATCH (s)<-[:MENTIONS]-(article:Article)

RETURN 
    s.name AS standard,
    s.organization AS organization,
    s.definition AS definition,
    collect(DISTINCT target.name) AS applies_to,
    collect(DISTINCT defined.name) AS defines,
    collect(DISTINCT article.title)[0..3] AS mentioned_in_articles
```

### Challenge-Solution Query

Find challenges and the best practices/tools that address them:

```cypher
// Find challenges and their solutions
MATCH (challenge:Challenge)
WHERE challenge.name CONTAINS $topic OR challenge.definition CONTAINS $topic

OPTIONAL MATCH (solution)-[:ADDRESSES]->(challenge)
WHERE solution:Bestpractice OR solution:Tool OR solution:Methodology

RETURN 
    challenge.name AS challenge,
    challenge.definition AS description,
    challenge.impact AS impact,
    collect(DISTINCT {
        solution_name: solution.name,
        solution_type: labels(solution),
        benefit: solution.benefit
    }) AS solutions
LIMIT 10
```

### Community-Based Retrieval Query (Essential GraphRAG Ch. 7)

Leverages the Chapter structure as natural communities:

```cypher
// Community-aware retrieval using Chapter hierarchy (Updated 2026-01)
// Note: Entities point TO chunks via MENTIONED_IN
// Note: Chunks point TO articles via FROM_ARTICLE
MATCH (ch:Chapter)-[:CONTAINS]->(a:Article)
MATCH (e)-[:MENTIONED_IN]->(c:Chunk)-[:FROM_ARTICLE]->(a)
WHERE e.name CONTAINS $topic OR e.definition CONTAINS $topic
WITH ch, count(DISTINCT e) AS entity_relevance, collect(DISTINCT e.name)[0..10] AS key_entities

ORDER BY entity_relevance DESC
LIMIT 1

// Get comprehensive context from this community (chapter)
MATCH (ch)-[:CONTAINS]->(a:Article)
OPTIONAL MATCH (c:Chunk)-[:FROM_ARTICLE]->(a)

RETURN
    ch.title AS community,
    ch.chapter_number AS community_id,
    entity_relevance AS relevance_score,
    key_entities,
    collect(DISTINCT {
        title: a.article_title,
        url: a.url
    }) AS articles,
    count(DISTINCT c) AS total_chunks
```

### Process Stage Workflow Query

Understand process stages and their prerequisites:

```cypher
// Get process stages with their dependencies
MATCH (stage:Processstage)
WHERE stage.name CONTAINS $process_name

OPTIONAL MATCH (prereq)-[:PREREQUISITE_FOR]->(stage)
OPTIONAL MATCH (stage)-[:PRODUCES]->(output)
OPTIONAL MATCH (stage)-[:REQUIRES]->(requirement)

RETURN 
    stage.name AS stage,
    stage.definition AS description,
    collect(DISTINCT prereq.name) AS prerequisites,
    collect(DISTINCT output.name) AS produces,
    collect(DISTINCT requirement.name) AS requires
```

---

## Neo4j Driver Best Practices

Based on [Neo4j Driver Best Practices](https://neo4j.com/blog/developer/neo4j-driver-best-practices/), the following patterns are critical for production deployments, especially in serverless environments like Vercel.

### Connection URI Scheme

**Always use `neo4j+s://` for Neo4j Aura and production clusters:**

| Scheme | Use Case | Certificate Validation |
|--------|----------|----------------------|
| `neo4j+s://` | **Recommended** - Aura, production clusters | ✅ Yes |
| `neo4j+ssc://` | Self-signed certificates | ⚠️ Self-signed only |
| `neo4j://` | Local development only | ❌ No |
| `bolt://` | Single instance, no routing | ❌ No |

The `neo4j+s://` scheme:
- Works with single instances AND clusters
- Enables certificate validation for secure connections
- Supports automatic routing to cluster members

**Warning:** Using `bolt://` with clusters will route ALL queries to the leader, underutilizing 2/3 of cluster resources.

### Driver Instance Management (Critical for Serverless)

**Create ONE driver instance and reuse it** - this is especially critical for Vercel/serverless:

```python
# ❌ BAD - Creates new driver on every request (cold start penalty)
def handle_request():
    driver = GraphDatabase.driver(uri, auth=(user, password))
    # ... use driver
    driver.close()

# ✅ GOOD - Reuse driver instance across requests
_driver: Driver | None = None

def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(uri, auth=(user, password))
        _driver.verify_connectivity()  # Fail fast on bad config
    return _driver
```

**Why this matters:**
- Driver objects contain connection pools (expensive to create)
- Creating a driver can take several seconds
- In serverless, this adds to cold start latency
- Sessions are cheap - create/close as many as needed

### Verify Connectivity Early

**Always verify connectivity immediately after creating a driver:**

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))
driver.verify_connectivity()  # Fails immediately if credentials wrong
```

This catches configuration errors (wrong URI, credentials) at startup rather than on first query.

### Use Explicit Transaction Functions

**Prefer `session.execute_read()` and `session.execute_write()` over `session.run()`:**

```python
# ❌ BAD - Auto-commit transaction, no cluster routing
def get_entities_bad(session, name):
    result = session.run("MATCH (e:Entity {name: $name}) RETURN e", name=name)
    return [record["e"] for record in result]

# ✅ GOOD - Explicit read transaction, proper cluster routing
def get_entities_good(session, name):
    def _query(tx, name):
        result = tx.run("MATCH (e:Entity {name: $name}) RETURN e", name=name)
        return [record["e"] for record in result]
    return session.execute_read(_query, name)

# ✅ GOOD - Explicit write transaction
def create_entity(session, name, definition):
    def _create(tx, name, definition):
        tx.run(
            "CREATE (e:Entity {name: $name, definition: $definition})",
            name=name, definition=definition
        )
    session.execute_write(_create, name, definition)
```

**Benefits:**
- `execute_read()` routes to any cluster member (load distribution)
- `execute_write()` routes to the leader
- Proper transaction boundaries for multi-query atomicity

### Always Use Query Parameters

**Never use string concatenation for query values:**

```python
# ❌ BAD - SQL/Cypher injection vulnerability, no query caching
def search_bad(tx, user_input):
    tx.run(f"MATCH (e:Entity) WHERE e.name = '{user_input}' RETURN e")

# ✅ GOOD - Safe, enables query plan caching
def search_good(tx, user_input):
    tx.run("MATCH (e:Entity) WHERE e.name = $name RETURN e", name=user_input)
```

**Benefits:**
- Prevents [Cypher injection attacks](https://neo4j.com/developer/kb/protecting-against-cypher-injection/)
- Enables query plan caching (faster execution)
- Fewer unique query strings = better database performance

### Process Results Within Transaction Scope

**Database cursors are not available outside transaction functions:**

```python
# ❌ BAD - Result cursor invalid outside transaction
def get_friends_bad(tx, name):
    result = tx.run("MATCH (p:Person)-[:KNOWS]->(f) WHERE p.name = $name RETURN f.name", name=name)
    return result  # Cursor won't work outside transaction!

# ✅ GOOD - Process and extract data within transaction
def get_friends_good(tx, name):
    result = tx.run("MATCH (p:Person)-[:KNOWS]->(f) WHERE p.name = $name RETURN f.name AS friend", name=name)
    return [record["friend"] for record in result]  # List of values
```

### Session Reuse for Causal Consistency

**Reuse sessions when later operations must see earlier writes:**

```python
# Reading your own writes - reuse the same session
with driver.session() as session:
    # Write operation
    session.execute_write(create_entity, "new_concept", "A new concept")
    
    # Read operation - guaranteed to see the write above
    # because same session chains bookmarks automatically
    entities = session.execute_read(get_all_entities)
```

**For cross-process consistency, pass bookmarks explicitly:**

```python
# Process A: Write and export bookmark
with driver.session() as session:
    session.execute_write(create_entity, "shared_concept", "...")
    bookmark = session.last_bookmark()
    # Send bookmark to Process B

# Process B: Read with bookmark guarantee
with driver.session(bookmarks=[bookmark_from_process_a]) as session:
    # Guaranteed to see Process A's writes
    entities = session.execute_read(get_all_entities)
```

### Serverless-Specific Optimizations (Vercel/Lambda)

For serverless deployments, consider these additional patterns:

```python
# Reduce connection pool for faster cold starts
driver = GraphDatabase.driver(
    uri,
    auth=(user, password),
    max_connection_pool_size=5,  # Smaller pool for serverless
    connection_acquisition_timeout=30,  # Longer timeout for cold starts
)
```

**Serverless considerations:**
- Keep connection pools small (5-10 connections)
- Use connection warmup in initialization
- Consider connection pooling services for high-traffic scenarios
- Monitor cold start times and optimize accordingly

---

## Text2Cypher Best Practices (Essential GraphRAG Ch. 4)

### Few-Shot Examples

Domain-specific examples to improve Cypher generation accuracy:

```python
TEXT2CYPHER_FEW_SHOT_EXAMPLES: Final[list[dict[str, str]]] = [
    {
        "question": "What standards apply to medical devices?",
        "cypher": """
MATCH (s:Standard)-[:APPLIES_TO]->(e:Entity)
WHERE any(t IN e.entity_type WHERE toLower(t) CONTAINS 'medical')
   OR e.name CONTAINS 'medical' OR e.name CONTAINS 'FDA'
RETURN s.name AS standard, s.organization AS organization, 
       collect(DISTINCT e.name)[0..5] AS applies_to
"""
    },
    {
        "question": "What challenges does requirements traceability address?",
        "cypher": """
MATCH (c:Concept)-[:ADDRESSES]->(ch:Challenge)
WHERE toLower(c.name) CONTAINS 'traceability'
RETURN c.name AS concept, ch.name AS challenge, ch.definition AS description
"""
    },
    {
        "question": "What tools are used for requirements management?",
        "cypher": """
MATCH (t:Tool)
WHERE any(et IN t.entity_type WHERE toLower(et) = 'tool')
RETURN t.name AS tool, t.vendor AS vendor, t.benefit AS benefits
ORDER BY t.name
LIMIT 20
"""
    },
    {
        "question": "Which best practices help with regulatory compliance?",
        "cypher": """
MATCH (bp:Bestpractice)-[:ADDRESSES]->(ch:Challenge)
WHERE toLower(ch.name) CONTAINS 'compliance' 
   OR toLower(ch.name) CONTAINS 'regulatory'
RETURN bp.name AS best_practice, bp.benefit AS benefit, 
       collect(DISTINCT ch.name) AS addresses_challenges
"""
    },
    {
        "question": "What are the prerequisites for requirements validation?",
        "cypher": """
MATCH (prereq)-[:PREREQUISITE_FOR]->(stage:Processstage)
WHERE toLower(stage.name) CONTAINS 'validation'
RETURN stage.name AS stage, collect(DISTINCT prereq.name) AS prerequisites
"""
    },
    {
        "question": "How many entities of each type are in the graph?",
        "cypher": """
MATCH (e:Entity)
UNWIND e.entity_type AS type
RETURN type, count(*) AS count
ORDER BY count DESC
"""
    },
    {
        "question": "What methodologies are mentioned in the automotive chapter?",
        "cypher": """
MATCH (ch:Chapter)-[:CONTAINS]->(a:Article)-[:MENTIONS]->(m:Methodology)
WHERE toLower(ch.title) CONTAINS 'automotive'
RETURN DISTINCT m.name AS methodology, m.definition AS description
"""
    },
]
```

### Terminology Mappings

Semantic mapping between user language and graph schema:

```python
TERMINOLOGY_MAPPINGS: Final[str] = """
Terminology Mapping for Requirements Management Knowledge Graph:

Entity Type Mappings:
- "best practices", "recommendations", "guidelines" → Bestpractice nodes
- "problems", "issues", "challenges", "pain points" → Challenge nodes
- "tools", "software", "applications", "platforms" → Tool nodes
- "standards", "regulations", "compliance", "certifications" → Standard nodes
- "roles", "stakeholders", "team members", "responsibilities" → Role nodes
- "processes", "stages", "phases", "steps" → Processstage nodes
- "concepts", "principles", "ideas" → Concept nodes
- "documents", "deliverables", "artifacts", "outputs" → Artifact nodes
- "methods", "approaches", "frameworks" → Methodology nodes

Industry Mappings:
- "automotive", "vehicle", "car" → automotive industry, ISO 26262
- "medical", "healthcare", "FDA", "life sciences" → medical device industry
- "aerospace", "aviation", "aircraft", "DO-178" → aerospace industry
- "defense", "military", "MIL-STD" → defense industry
- "semiconductor", "chip", "electronics" → semiconductor industry

Standard Organization Mappings:
- "ISO" → International Organization for Standardization
- "FDA" → Food and Drug Administration (US medical devices)
- "INCOSE" → International Council on Systems Engineering
- "IEEE" → Institute of Electrical and Electronics Engineers
- "SAE" → Society of Automotive Engineers

Tool Mappings:
- "Jama", "Jama Connect" → Primary requirements management tool
- "DOORS" → IBM Rational DOORS
- "Excel", "spreadsheet" → Traditional manual approaches
"""
```

### Text2Cypher Prompt Template

```python
TEXT2CYPHER_PROMPT_TEMPLATE: Final[str] = """
You are a Cypher query expert for a Requirements Management knowledge graph.

INSTRUCTIONS:
Generate a Cypher statement to query the graph database and answer the user's question.
Use ONLY the provided schema elements. Do not invent properties or relationships.

GRAPH SCHEMA:
{schema}

TERMINOLOGY MAPPING:
{terminology_mappings}

FEW-SHOT EXAMPLES:
{examples}

FORMAT INSTRUCTIONS:
- Return ONLY the Cypher query, no explanations
- Do NOT use markdown code blocks
- Use parameterized queries where appropriate
- Limit results to reasonable numbers (10-20) unless asked for more
- Handle case-insensitivity with toLower() for text matching

USER QUESTION: {question}

CYPHER QUERY:
"""
```

---

## Agentic RAG Architecture (Essential GraphRAG Ch. 5)

### Retriever Router

The retriever router selects the optimal tool for each query type:

```python
"""Retriever router for agentic RAG.

Analyzes user queries and routes them to the most appropriate
retrieval tool based on query characteristics.
"""

from __future__ import annotations

from typing import Final

# Tool descriptions for routing decisions
RETRIEVER_TOOLS: Final[dict[str, str]] = {
    "graphrag_vector_search": """
        Basic semantic search using vector embeddings.
        Use for: General content lookup, finding relevant passages.
        Example: "What does the guide say about traceability?"
    """,
    "graphrag_hybrid_search": """
        Combined vector + keyword search for improved accuracy.
        Use for: Queries with specific technical terms or acronyms.
        Example: "ISO 26262 ASIL requirements"
    """,
    "graphrag_retrieve": """
        Graph-enriched retrieval with entity relationships.
        Use for: Understanding how concepts relate to each other.
        Example: "How does V-Model relate to requirements validation?"
    """,
    "graphrag_explore_entity": """
        Deep dive into specific entities and their relationships.
        Use for: Learning about a specific concept, standard, or tool.
        Example: "Tell me about ISO 26262" or "What is requirements traceability?"
    """,
    "graphrag_standards": """
        Standards and compliance lookup.
        Use for: Regulatory requirements, industry standards, certifications.
        Example: "What standards apply to medical devices?"
    """,
    "graphrag_definitions": """
        Term definitions from the knowledge base.
        Use for: Understanding specific terminology.
        Example: "Define baseline" or "What is an atomic requirement?"
    """,
    "graphrag_text2cypher": """
        Natural language to Cypher query generation.
        Use for: Complex queries, aggregations, specific patterns.
        Example: "Which chapter has the most entities?" or "List all tools by vendor"
    """,
    "graphrag_chat": """
        Full RAG conversational Q&A.
        Use for: Complex questions requiring synthesis from multiple sources.
        Example: "How should I approach requirements management for a medical device?"
    """,
}

ROUTER_SYSTEM_PROMPT: Final[str] = """
You are a retrieval router for a Requirements Management knowledge graph.

Your task is to analyze the user's question and select the best retrieval tool(s).

Available tools:
{tools}

Selection Guidelines:
1. For simple lookups or general questions → graphrag_vector_search or graphrag_hybrid_search
2. For questions about how concepts relate → graphrag_retrieve
3. For deep dives into specific entities → graphrag_explore_entity
4. For regulatory/compliance questions → graphrag_standards
5. For terminology definitions → graphrag_definitions
6. For aggregations or complex patterns → graphrag_text2cypher
7. For multi-faceted questions requiring synthesis → graphrag_chat

You may select multiple tools if the question has multiple parts.

Return a JSON object with:
{{
    "selected_tools": ["tool_name1", "tool_name2"],
    "reasoning": "Brief explanation of why these tools were selected",
    "tool_params": {{
        "tool_name1": {{"query": "refined query for this tool"}},
        "tool_name2": {{"query": "refined query for this tool"}}
    }}
}}

User Question: {question}
"""
```

### Step-Back Prompting (Essential GraphRAG Ch. 3)

Query rewriting to improve retrieval accuracy:

```python
"""Step-back prompting for query refinement.

Transforms specific questions into broader queries for better retrieval,
then uses the broader context to answer the specific question.
"""

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
```

### Answer Critic (Essential GraphRAG Ch. 5)

Validates that retrieved context answers the question:

```python
"""Answer critic for validating retrieval quality.

Evaluates whether retrieved context is sufficient to answer
the user's question and suggests follow-up queries if needed.
"""

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
```

### Continuous Query Updating

For multi-part questions, update subsequent queries with answers:

```python
"""Query updater for multi-part questions.

Updates remaining questions with context from previously answered parts.
"""

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
```

---

## Evaluation Framework (Essential GraphRAG Ch. 8)

### RAGAS Metrics Implementation

```python
"""RAG evaluation using RAGAS metrics.

Implements Context Recall, Faithfulness, and Answer Correctness
metrics for evaluating GraphRAG performance.
"""

from __future__ import annotations

from typing import Final

# Context Recall Evaluation
CONTEXT_RECALL_PROMPT: Final[str] = """
Goal: Given a context and an answer, analyze each sentence in the answer 
and classify whether the sentence can be attributed to the given context.

Use only 'Yes' (1) or 'No' (0) as a binary classification.

Context:
{context}

Answer:
{answer}

Ground Truth:
{ground_truth}

For each sentence in the ground truth, determine if it can be found in the context.

Return JSON:
{{
    "sentences": [
        {{"sentence": "...", "in_context": 1, "reasoning": "..."}},
        ...
    ],
    "recall_score": 0.0-1.0
}}
"""

# Faithfulness Evaluation (Step 1: Statement Breakdown)
FAITHFULNESS_BREAKDOWN_PROMPT: Final[str] = """
Goal: Break down the answer into atomic statements that can be verified.

Answer:
{answer}

Rules:
1. Each statement should be self-contained
2. No pronouns - use full entity names
3. Each statement should express a single fact

Return JSON:
{{
    "statements": [
        "Statement 1 with full context",
        "Statement 2 with full context",
        ...
    ]
}}
"""

# Faithfulness Evaluation (Step 2: Verification)
FAITHFULNESS_VERIFY_PROMPT: Final[str] = """
Goal: Verify each statement against the provided context.

Context:
{context}

Statements to verify:
{statements}

For each statement, return:
- 1 if the statement can be directly inferred from the context
- 0 if the statement cannot be directly inferred from the context

Return JSON:
{{
    "verifications": [
        {{"statement": "...", "verdict": 1, "evidence": "quote from context"}},
        {{"statement": "...", "verdict": 0, "evidence": null}},
        ...
    ],
    "faithfulness_score": 0.0-1.0
}}
"""

# Answer Correctness Evaluation
ANSWER_CORRECTNESS_PROMPT: Final[str] = """
Goal: Compare the generated answer against the ground truth.

Ground Truth:
{ground_truth}

Generated Answer:
{answer}

Classify each statement in the answer:
- TP (True Positive): Present in answer AND supported by ground truth
- FP (False Positive): Present in answer but NOT in ground truth
- FN (False Negative): In ground truth but NOT present in answer

Return JSON:
{{
    "classifications": [
        {{"statement": "...", "classification": "TP", "reasoning": "..."}},
        ...
    ],
    "true_positives": 0,
    "false_positives": 0,
    "false_negatives": 0,
    "precision": 0.0-1.0,
    "recall": 0.0-1.0,
    "f1_score": 0.0-1.0
}}
"""
```

### Benchmark Dataset Structure

```python
"""Benchmark dataset for GraphRAG evaluation.

Test cases covering different query types and expected behaviors.
"""

from __future__ import annotations

from typing import Final

BENCHMARK_DATASET: Final[list[dict]] = [
    # Greeting and scope tests
    {
        "category": "greeting",
        "question": "Hello",
        "expected_behavior": "greeting_with_scope",
        "expected_response_contains": ["requirements", "help"],
    },
    {
        "category": "scope",
        "question": "What can you help me with?",
        "expected_behavior": "explain_capabilities",
        "expected_response_contains": ["requirements", "standards", "traceability"],
    },
    {
        "category": "out_of_scope",
        "question": "What is the weather like today?",
        "expected_behavior": "politely_decline",
        "expected_response_contains": ["requirements", "cannot", "outside"],
    },
    
    # Entity lookup tests
    {
        "category": "entity_lookup",
        "question": "What is requirements traceability?",
        "expected_tool": "graphrag_explore_entity",
        "ground_truth": "Requirements traceability is the ability to trace requirements throughout the development lifecycle",
        "expected_response_contains": ["traceability", "requirements", "lifecycle"],
    },
    {
        "category": "entity_lookup",
        "question": "Tell me about ISO 26262",
        "expected_tool": "graphrag_explore_entity",
        "ground_truth": "ISO 26262 is the automotive functional safety standard",
        "expected_response_contains": ["ISO 26262", "automotive", "safety"],
    },
    
    # Standards queries
    {
        "category": "standards",
        "question": "What standards apply to medical devices?",
        "expected_tool": "graphrag_standards",
        "ground_truth": "FDA regulations, IEC 62304, ISO 13485 apply to medical devices",
        "expected_response_contains": ["FDA", "medical", "regulatory"],
    },
    {
        "category": "standards",
        "question": "What is DO-178C used for?",
        "expected_tool": "graphrag_standards",
        "ground_truth": "DO-178C is used for software certification in airborne systems",
        "expected_response_contains": ["DO-178C", "aerospace", "software", "safety"],
    },
    
    # Relationship queries
    {
        "category": "relationship",
        "question": "How does the V-Model relate to requirements validation?",
        "expected_tool": "graphrag_retrieve",
        "expected_response_contains": ["V-Model", "validation", "verification"],
    },
    {
        "category": "relationship",
        "question": "What challenges does requirements traceability address?",
        "expected_tool": "graphrag_retrieve",
        "expected_response_contains": ["traceability", "challenge", "address"],
    },
    
    # Definition tests
    {
        "category": "definitions",
        "question": "Define baseline",
        "expected_tool": "graphrag_definitions",
        "ground_truth": "A baseline is a formally reviewed and agreed upon set of requirements",
        "expected_response_contains": ["baseline", "definition"],
    },
    {
        "category": "definitions",
        "question": "What is an atomic requirement?",
        "expected_tool": "graphrag_definitions",
        "expected_response_contains": ["atomic", "requirement", "single"],
    },
    
    # Aggregation queries (text2cypher)
    {
        "category": "aggregation",
        "question": "How many entities of each type are in the graph?",
        "expected_tool": "graphrag_text2cypher",
        "expected_cypher_contains": ["count", "entity_type"],
    },
    {
        "category": "aggregation",
        "question": "Which chapter has the most articles?",
        "expected_tool": "graphrag_text2cypher",
        "expected_cypher_contains": ["Chapter", "count", "ORDER BY"],
    },
    
    # Complex synthesis (RAG chat)
    {
        "category": "synthesis",
        "question": "How should I approach requirements management for an automotive safety-critical system?",
        "expected_tool": "graphrag_chat",
        "expected_response_contains": ["ISO 26262", "traceability", "safety", "ASIL"],
    },
    
    # Edge cases - missing data
    {
        "category": "edge_case",
        "question": "What is the price of Jama Connect?",
        "expected_behavior": "acknowledge_missing",
        "expected_response_contains": ["not available", "pricing", "contact"],
    },
]
```

---

## Development Commands

```bash
# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Create project with uv (Python package manager)
uv init jama-mcp-server-graphrag
cd jama-mcp-server-graphrag

# Install all dependencies including dev tools
uv sync --group dev

# Run the environment test to verify setup
python -m jama_mcp_server_graphrag.test_environment

# ============================================================================
# RUNNING THE SERVER
# ============================================================================

# Run MCP server in stdio mode (for Claude Desktop integration)
uv run python -m jama_mcp_server_graphrag.server

# Run MCP server in HTTP mode (for remote access)
uv run python -m jama_mcp_server_graphrag.server --transport http --port 8000

# Run with fastmcp CLI
uv run fastmcp run src/jama_mcp_server_graphrag/server.py

# ============================================================================
# DOCKER
# ============================================================================

# Build Docker image
docker build -t jama-mcp-server-graphrag:latest .

# Run Docker container
docker run -p 8000:8000 --env-file .env jama-mcp-server-graphrag:latest

# Run with Docker Compose (includes health checks)
docker compose up -d

# View logs
docker compose logs -f

# Stop containers
docker compose down

# ============================================================================
# TESTING
# ============================================================================

# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=jama_mcp_server_graphrag --cov-report=html

# Run a single test file
uv run pytest tests/test_tools.py -v

# Run evaluation benchmark
uv run pytest tests/benchmark/ -v --benchmark

# Test with MCP Inspector
npx @modelcontextprotocol/inspector uv run python -m jama_mcp_server_graphrag.server

# ============================================================================
# LINTING & FORMATTING (ruff)
# ============================================================================

uv run ruff check .                    # Check for issues
uv run ruff check . --fix              # Auto-fix issues
uv run ruff format .                   # Format code (Black-compatible)
uv run ruff format --check .           # Check formatting without changing

# ============================================================================
# TYPE CHECKING (ty)
# ============================================================================

uv run ty check .                      # Full type check
uv run ty check src/jama_mcp_server_graphrag/server.py  # Check single file

# ============================================================================
# CI/CD LOCAL TESTING
# ============================================================================

# Run the full CI pipeline locally with act (GitHub Actions local runner)
act push

# Test Vercel deployment locally
vercel dev
```

---

## Project Structure

```
jama-mcp-server-graphrag/
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Continuous integration
│       └── deploy.yml                 # Deployment to Vercel
├── .vscode/
│   ├── settings.json                  # Python interpreter, Ruff, format-on-save
│   ├── extensions.json                # Recommended extensions
│   └── launch.json                    # Debug configurations
├── src/
│   └── jama_mcp_server_graphrag/
│       ├── __init__.py
│       ├── server.py                  # FastMCP server entry point
│       ├── api.py                     # FastAPI REST API entry point
│       ├── config.py                  # Configuration management (dataclass)
│       ├── exceptions.py              # Custom exception hierarchy
│       ├── validators.py              # Input validation utilities
│       ├── formatters.py              # Response formatting utilities
│       ├── neo4j_client.py            # Neo4j driver best practices wrapper
│       ├── llm_client.py              # LLM initialization utilities
│       ├── prompts.py                 # Prompt templates (RAG, Text2Cypher, etc.)
│       ├── core/                      # Shared GraphRAG logic (used by tools/ and routes/)
│       │   ├── __init__.py
│       │   ├── retrieval.py           # Vector, hybrid, graph-enriched search
│       │   ├── text2cypher.py         # Natural language to Cypher
│       │   ├── generation.py          # Answer generation with citations
│       │   ├── entities.py            # Entity exploration logic
│       │   ├── definitions.py         # Definition/glossary lookup logic
│       │   ├── standards.py           # Standards lookup logic
│       │   └── schema.py              # Schema exploration logic
│       ├── tools/                     # MCP tool implementations (thin wrappers around core/)
│       │   ├── __init__.py            # Tool registration exports
│       │   ├── vector_search.py       # Basic vector similarity search
│       │   ├── hybrid_search.py       # Combined vector + fulltext search
│       │   ├── graph_retrieval.py     # Vector + entity enrichment (GraphRAG)
│       │   ├── entity_explorer.py     # Deep-dive entity exploration
│       │   ├── standards.py           # Standards/compliance lookup
│       │   ├── definitions.py         # Definition term lookup
│       │   ├── text2cypher.py         # Natural language to Cypher
│       │   ├── schema.py              # Schema exploration tool
│       │   ├── chat.py                # Full RAG chat workflow
│       │   └── evaluate.py            # Evaluation tool (RAGAS metrics)
│       ├── routes/                    # FastAPI route handlers (thin wrappers around core/)
│       │   ├── __init__.py
│       │   ├── chat.py                # POST /chat - Main chat endpoint
│       │   ├── search.py              # POST /search - Search endpoints
│       │   ├── schema.py              # GET /schema - Schema exploration
│       │   ├── definitions.py         # GET /definitions - Definition lookup
│       │   ├── standards.py           # GET /standards - Standards lookup
│       │   └── health.py              # GET /health - Health check
│       ├── agentic/
│       │   ├── __init__.py
│       │   ├── router.py              # Retriever router
│       │   ├── stepback.py            # Step-back prompting
│       │   ├── critic.py              # Answer critic
│       │   └── query_updater.py       # Multi-query context updater
│       ├── resources/
│       │   ├── __init__.py
│       │   ├── schema.py              # Schema information resources
│       │   └── indexes.py             # Index metadata resources
│       └── workflows/
│           ├── __init__.py
│           ├── rag_workflow.py        # LangGraph RAG workflow
│           ├── agentic_workflow.py    # Agentic RAG with routing
│           └── state.py               # Workflow state definitions
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_environment.py            # Environment validation tests
│   ├── test_config.py                 # Configuration tests
│   ├── test_neo4j_client.py           # Neo4j client best practices tests
│   ├── test_api.py                    # FastAPI endpoint tests
│   ├── test_core/                     # Core logic tests
│   │   ├── __init__.py
│   │   ├── test_retrieval.py
│   │   ├── test_text2cypher.py
│   │   └── test_generation.py
│   ├── test_tools/
│   │   ├── __init__.py
│   │   ├── test_vector_search.py
│   │   ├── test_hybrid_search.py
│   │   ├── test_graph_retrieval.py
│   │   ├── test_entity_explorer.py
│   │   ├── test_standards.py
│   │   ├── test_definitions.py
│   │   └── test_text2cypher.py
│   ├── test_agentic/
│   │   ├── __init__.py
│   │   ├── test_router.py
│   │   ├── test_stepback.py
│   │   └── test_critic.py
│   ├── test_workflows/
│   │   ├── __init__.py
│   │   └── test_rag_workflow.py
│   └── benchmark/
│       ├── __init__.py
│       ├── dataset.py                 # Benchmark test cases
│       ├── test_retrieval_accuracy.py # Retrieval evaluation
│       └── test_answer_quality.py     # Answer quality evaluation
├── docker/
│   ├── Dockerfile                     # Production container
│   └── Dockerfile.dev                 # Development container
├── .env.example                       # Environment variable template
├── .gitignore
├── .dockerignore
├── docker-compose.yml                 # Local development environment
├── pyproject.toml                     # Project configuration with ruff/ty
├── vercel.json                        # Vercel deployment configuration
├── README.md                          # Project documentation
├── CLAUDE.md                          # Claude Code instructions
├── SPECIFICATION.md                   # This file - detailed implementation spec
├── fastmcp.json                       # FastMCP deployment configuration
└── LICENSE
```

---

## Architecture

### Design Principles

The project follows **production-ready patterns** demonstrated in the reference repository and **Essential GraphRAG** best practices:

1. **Structured Logging**: Module-level loggers with context
2. **Custom Exception Hierarchy**: Exception chaining for debugging
3. **Configuration Management**: Immutable dataclasses with validation
4. **Dependency Injection**: Factory functions for testability
5. **Input Validation**: Pydantic models with constraints
6. **Resource Lifecycle Management**: Proper connection handling
7. **Closure-based Abstractions**: Loose coupling for flexibility
8. **Hybrid Retrieval**: Combined vector + keyword search
9. **Agentic Patterns**: Retriever routing, answer validation
10. **Evaluation-Driven**: RAGAS metrics for quality assurance

### Module Structure (10 Sections per File)

Each core module should follow this structure:

```python
"""Module docstring with purpose, patterns, and usage examples."""

from __future__ import annotations

# =============================================================================
# 1. IMPORTS - Runtime imports, then TYPE_CHECKING block
# =============================================================================

# =============================================================================
# 2. LOGGING CONFIGURATION - Module-level logger
# =============================================================================

# =============================================================================
# 3. CONSTANTS - Final types for immutable values
# =============================================================================

# =============================================================================
# 4. EXCEPTIONS - Custom exception hierarchy
# =============================================================================

# =============================================================================
# 5. CONFIGURATION - Immutable dataclass with validation
# =============================================================================

# =============================================================================
# 6. TYPE DEFINITIONS - TypedDicts, Protocols, type aliases
# =============================================================================

# =============================================================================
# 7. HELPER FUNCTIONS - Pure utility functions
# =============================================================================

# =============================================================================
# 8. FACTORY FUNCTIONS - Dependency injection creators
# =============================================================================

# =============================================================================
# 9. CORE LOGIC - Main business logic implementation
# =============================================================================

# =============================================================================
# 10. EXPORTS - Module-level __all__ definition
# =============================================================================
```

---

## Docker Configuration

### Dockerfile

```dockerfile
# Production Dockerfile for GraphRAG MCP Server
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Builder stage - install dependencies
# ============================================================================
FROM base AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev dependencies for production)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ============================================================================
# Production stage
# ============================================================================
FROM base AS production

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose port for HTTP transport
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - HTTP mode for cloud deployment
CMD ["python", "-m", "jama_mcp_server_graphrag.server", "--transport", "http", "--port", "8000"]
```

### docker-compose.yml

```yaml
# Docker Compose for local development
version: "3.9"

services:
  graphrag-mcp:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: jama-mcp-server-graphrag
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=${NEO4J_URI}
      - NEO4J_USERNAME=${NEO4J_USERNAME}
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - NEO4J_DATABASE=${NEO4J_DATABASE:-neo4j}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o}
      - EMBEDDING_MODEL=${EMBEDDING_MODEL:-text-embedding-3-small}
      - VECTOR_INDEX_NAME=${VECTOR_INDEX_NAME:-chunk_embeddings}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    env_file:
      - .env
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    networks:
      - graphrag-network

  # Optional: Local Neo4j for development (comment out if using remote)
  # neo4j:
  #   image: neo4j:5.15-community
  #   container_name: graphrag-neo4j
  #   ports:
  #     - "7474:7474"
  #     - "7687:7687"
  #   environment:
  #     - NEO4J_AUTH=neo4j/your-password
  #     - NEO4J_PLUGINS=["apoc", "graph-data-science"]
  #   volumes:
  #     - neo4j_data:/data
  #     - neo4j_logs:/logs
  #   networks:
  #     - graphrag-network

networks:
  graphrag-network:
    driver: bridge

# volumes:
#   neo4j_data:
#   neo4j_logs:
```

### .dockerignore

```
# Git
.git
.gitignore

# Python
__pycache__
*.py[cod]
*$py.class
*.so
.Python
.venv
venv
ENV
.eggs
*.egg-info
.pytest_cache
.coverage
htmlcov
.mypy_cache
.ruff_cache

# IDE
.vscode
.idea
*.swp
*.swo

# Project specific
.env
.env.*
!.env.example
*.log
tests/
docs/
*.md
!README.md

# Build artifacts
dist
build
```

---

## GitHub Actions CI/CD

### `.github/workflows/ci.yml`

```yaml
# Continuous Integration Pipeline
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.12"

jobs:
  lint:
    name: Lint & Format Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --group dev

      - name: Run Ruff linter
        run: uv run ruff check . --output-format=github

      - name: Run Ruff formatter check
        run: uv run ruff format --check .

  type-check:
    name: Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --group dev

      - name: Run type checking
        run: uv run ty check .

  test:
    name: Test
    runs-on: ubuntu-latest
    needs: [lint, type-check]
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --group dev

      - name: Run tests with coverage
        run: uv run pytest --cov=jama_mcp_server_graphrag --cov-report=xml --cov-report=html
        env:
          NEO4J_URI: ${{ secrets.NEO4J_URI }}
          NEO4J_USERNAME: ${{ secrets.NEO4J_USERNAME }}
          NEO4J_PASSWORD: ${{ secrets.NEO4J_PASSWORD }}
          NEO4J_DATABASE: ${{ secrets.NEO4J_DATABASE }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          file: ./coverage.xml
          fail_ci_if_error: false

      - name: Upload coverage HTML report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: htmlcov/

  build-docker:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [test]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: jama-mcp-server-graphrag:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Test Docker image
        run: |
          docker run --rm -d --name test-container \
            -e NEO4J_URI=neo4j://localhost:7687 \
            -e NEO4J_USERNAME=neo4j \
            -e NEO4J_PASSWORD=test \
            -e OPENAI_API_KEY=sk-test \
            jama-mcp-server-graphrag:${{ github.sha }} \
            sleep 30
          
          # Wait for container to start
          sleep 5
          
          # Verify Python imports work
          docker exec test-container python -c "from jama_mcp_server_graphrag import server; print('Import OK')"
          
          # Cleanup
          docker stop test-container
```

### `.github/workflows/deploy.yml`

```yaml
# Deployment Pipeline to Vercel
name: Deploy

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.12"

jobs:
  deploy:
    name: Deploy to Vercel
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --no-dev

      - name: Install Vercel CLI
        run: npm install -g vercel

      - name: Pull Vercel Environment
        run: vercel pull --yes --environment=production --token=${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}

      - name: Build Project
        run: vercel build --prod --token=${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}

      - name: Deploy to Vercel
        run: vercel deploy --prebuilt --prod --token=${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}

      - name: Verify Deployment
        run: |
          DEPLOY_URL=$(vercel ls --token=${{ secrets.VERCEL_TOKEN }} | grep -E '^https://' | head -1 | awk '{print $1}')
          echo "Deployment URL: $DEPLOY_URL"
          
          # Wait for deployment to be ready
          sleep 30
          
          # Health check
          curl -f "$DEPLOY_URL/health" || echo "Health check endpoint not available"
```

---

## Vercel Configuration

### `vercel.json`

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "version": 2,
  "name": "jama-mcp-server-graphrag",
  "builds": [
    {
      "src": "src/jama_mcp_server_graphrag/server.py",
      "use": "@vercel/python",
      "config": {
        "maxLambdaSize": "50mb",
        "runtime": "python3.12"
      }
    }
  ],
  "routes": [
    {
      "src": "/health",
      "dest": "src/jama_mcp_server_graphrag/server.py"
    },
    {
      "src": "/(.*)",
      "dest": "src/jama_mcp_server_graphrag/server.py"
    }
  ],
  "env": {
    "NEO4J_URI": "@neo4j_uri",
    "NEO4J_USERNAME": "@neo4j_username",
    "NEO4J_PASSWORD": "@neo4j_password",
    "NEO4J_DATABASE": "@neo4j_database",
    "OPENAI_API_KEY": "@openai_api_key",
    "OPENAI_MODEL": "@openai_model",
    "EMBEDDING_MODEL": "@embedding_model",
    "VECTOR_INDEX_NAME": "@vector_index_name"
  },
  "functions": {
    "src/jama_mcp_server_graphrag/server.py": {
      "memory": 1024,
      "maxDuration": 60
    }
  }
}
```

### Vercel Environment Setup

```bash
# Set up Vercel secrets (run these commands in your terminal)
vercel secrets add neo4j_uri "neo4j+s://your-neo4j-uri"
vercel secrets add neo4j_username "neo4j"
vercel secrets add neo4j_password "your-password"
vercel secrets add neo4j_database "neo4j"
vercel secrets add openai_api_key "sk-your-openai-key"
vercel secrets add openai_model "gpt-4o"
vercel secrets add embedding_model "text-embedding-3-small"
vercel secrets add vector_index_name "chunk_embeddings"
```

---

## Dependencies

### `pyproject.toml`

```toml
[project]
name = "jama-mcp-server-graphrag"
version = "0.1.0"
description = "GraphRAG MCP Server for Requirements Management Knowledge Graph"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
keywords = [
    "mcp",
    "graphrag",
    "neo4j",
    "langchain",
    "langgraph",
    "requirements-management",
    "knowledge-graph",
    "rag",
    "agentic-ai",
]
authors = [{ name = "Your Name", email = "your.email@example.com" }]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "mcp>=1.0.0",
    "fastmcp>=0.4.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-neo4j>=0.2.0",
    "langgraph>=0.2.0",
    "neo4j>=5.15.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "httpx>=0.27.0",
    "python-dotenv>=1.0.0",
    "structlog>=24.0.0",
    "tenacity>=8.2.0",
    "tiktoken>=0.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "pytest-benchmark>=4.0.0",
    "ruff>=0.6.0",
    "ty>=0.0.1a0",
    "pre-commit>=3.5.0",
    "ragas>=0.1.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/jama-mcp-server-graphrag"
Repository = "https://github.com/yourusername/jama-mcp-server-graphrag"
Documentation = "https://github.com/yourusername/jama-mcp-server-graphrag#readme"

[project.scripts]
jama-mcp-server-graphrag = "jama_mcp_server_graphrag.server:main"
jama-graphrag-api = "jama_mcp_server_graphrag.api:run"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/jama_mcp_server_graphrag"]

# =============================================================================
# Ruff Configuration
# =============================================================================
[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings  
    "F",      # Pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "D",      # pydocstyle
    "UP",     # pyupgrade
    "ANN",    # flake8-annotations
    "ASYNC",  # flake8-async
    "S",      # flake8-bandit (security)
    "B",      # flake8-bugbear
    "A",      # flake8-builtins
    "COM",    # flake8-commas
    "C4",     # flake8-comprehensions
    "DTZ",    # flake8-datetimez
    "T10",    # flake8-debugger
    "ISC",    # flake8-implicit-str-concat
    "ICN",    # flake8-import-conventions
    "LOG",    # flake8-logging
    "G",      # flake8-logging-format
    "PIE",    # flake8-pie
    "PT",     # flake8-pytest-style
    "Q",      # flake8-quotes
    "RSE",    # flake8-raise
    "RET",    # flake8-return
    "SLF",    # flake8-self
    "SIM",    # flake8-simplify
    "TID",    # flake8-tidy-imports
    "TCH",    # flake8-type-checking
    "ARG",    # flake8-unused-arguments
    "PTH",    # flake8-use-pathlib
    "PL",     # Pylint
    "TRY",    # tryceratops
    "FLY",    # flynt
    "PERF",   # Perflint
    "RUF",    # Ruff-specific rules
]

ignore = [
    "D100",   # Missing docstring in public module
    "D104",   # Missing docstring in public package
    "D107",   # Missing docstring in __init__
    "ANN101", # Missing type annotation for self
    "ANN102", # Missing type annotation for cls
    "ANN401", # Dynamically typed expressions (Any)
    "S101",   # Use of assert (needed for tests)
    "COM812", # Trailing comma missing (conflicts with formatter)
    "ISC001", # Single line implicit string concatenation (conflicts with formatter)
]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = [
    "S101",   # Allow assert in tests
    "ANN",    # Don't require annotations in tests
    "D",      # Don't require docstrings in tests
    "PLR2004", # Allow magic values in tests
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

# =============================================================================
# Type Checking Configuration (ty)
# =============================================================================
[tool.ty]
python_version = "3.12"
strict = true

# =============================================================================
# Pytest Configuration
# =============================================================================
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = [
    "-ra",
    "-q",
    "--strict-markers",
]
markers = [
    "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    "benchmark: marks tests as benchmark tests",
    "slow: marks tests as slow",
]

[tool.coverage.run]
source = ["src/jama_mcp_server_graphrag"]
branch = true
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
]
fail_under = 80
show_missing = true
```

---

## Implementation Guide

### Phase 1: Core Infrastructure

#### 1.1 Configuration (`config.py`)

```python
"""Configuration management for GraphRAG MCP Server.

Provides immutable configuration using dataclasses with validation,
environment variable loading, and sensible defaults.

Neo4j Connection Best Practices:
- Use neo4j+s:// for production (Aura, clusters) - enables TLS and routing
- Use neo4j:// only for local development without TLS
- Never use bolt:// with clusters (no routing support)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Valid Neo4j URI schemes
VALID_NEO4J_SCHEMES: Final[tuple[str, ...]] = (
    "neo4j://",
    "neo4j+s://",
    "neo4j+ssc://",
    "bolt://",
    "bolt+s://",
    "bolt+ssc://",
)

# Recommended schemes for production (TLS-enabled)
SECURE_NEO4J_SCHEMES: Final[tuple[str, ...]] = (
    "neo4j+s://",
    "neo4j+ssc://",
    "bolt+s://",
    "bolt+ssc://",
)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Immutable application configuration.
    
    Attributes:
        neo4j_uri: Neo4j connection URI (prefer neo4j+s:// for production).
        neo4j_username: Neo4j username.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        openai_api_key: OpenAI API key.
        chat_model: Chat model name for generation.
        embedding_model: Embedding model name.
        vector_index_name: Name of the vector index in Neo4j.
        similarity_k: Default number of results for similarity search.
        log_level: Logging level.
        neo4j_max_connection_pool_size: Max connections (reduce for serverless).
        neo4j_connection_acquisition_timeout: Timeout for acquiring connections.
    """
    
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str = "neo4j"
    openai_api_key: str = ""
    chat_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    vector_index_name: str = "chunk_embeddings"
    similarity_k: int = 6
    log_level: str = "INFO"
    # Neo4j driver settings optimized for serverless (Vercel/Lambda)
    neo4j_max_connection_pool_size: int = 5
    neo4j_connection_acquisition_timeout: float = 30.0
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate Neo4j URI scheme
        if not any(self.neo4j_uri.startswith(scheme) for scheme in VALID_NEO4J_SCHEMES):
            msg = f"Invalid Neo4j URI scheme. Must start with one of: {VALID_NEO4J_SCHEMES}"
            raise ConfigurationError(msg)
        
        # Warn if using insecure scheme in production-like URI
        is_secure = any(self.neo4j_uri.startswith(s) for s in SECURE_NEO4J_SCHEMES)
        is_production_uri = (
            "aura" in self.neo4j_uri.lower() or 
            "neo4j.io" in self.neo4j_uri.lower() or
            not any(local in self.neo4j_uri for local in ["localhost", "127.0.0.1", "host.docker.internal"])
        )
        
        if is_production_uri and not is_secure:
            logger.warning(
                "Using insecure Neo4j connection scheme for production URI. "
                "Consider using neo4j+s:// for TLS encryption and certificate validation."
            )
        
        # Validate similarity_k
        if not 1 <= self.similarity_k <= 100:
            msg = "similarity_k must be between 1 and 100"
            raise ConfigurationError(msg)
        
        # Validate connection pool size
        if not 1 <= self.neo4j_max_connection_pool_size <= 100:
            msg = "neo4j_max_connection_pool_size must be between 1 and 100"
            raise ConfigurationError(msg)


def get_config() -> AppConfig:
    """Load configuration from environment variables.
    
    Returns:
        AppConfig instance with values from environment.
        
    Raises:
        ConfigurationError: If required environment variables are missing.
    """
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        msg = f"Missing required environment variables: {', '.join(missing)}"
        raise ConfigurationError(msg)
    
    return AppConfig(
        neo4j_uri=os.environ["NEO4J_URI"],
        neo4j_username=os.environ["NEO4J_USERNAME"],
        neo4j_password=os.environ["NEO4J_PASSWORD"],
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chat_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        vector_index_name=os.getenv("VECTOR_INDEX_NAME", "chunk_embeddings"),
        similarity_k=int(os.getenv("SIMILARITY_K", "6")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        neo4j_max_connection_pool_size=int(os.getenv("NEO4J_MAX_POOL_SIZE", "5")),
        neo4j_connection_acquisition_timeout=float(os.getenv("NEO4J_CONNECTION_TIMEOUT", "30.0")),
    )
```

#### 1.2 Exceptions (`exceptions.py`)

```python
"""Custom exception hierarchy for GraphRAG MCP Server.

Provides specific exceptions for different failure modes with
proper exception chaining support.
"""

from __future__ import annotations


class GraphRAGError(Exception):
    """Base exception for all GraphRAG errors."""


class ConfigurationError(GraphRAGError):
    """Raised when configuration is invalid or missing."""


class ConnectionError(GraphRAGError):
    """Raised when connection to external service fails."""


class Neo4jConnectionError(ConnectionError):
    """Raised when Neo4j connection fails."""


class InputValidationError(GraphRAGError):
    """Raised when input validation fails."""


class GraphTraversalError(GraphRAGError):
    """Raised when graph traversal operations fail."""


class VectorSearchError(GraphRAGError):
    """Raised when vector similarity search fails."""


class LLMError(GraphRAGError):
    """Raised when LLM operations fail."""


class Text2CypherError(LLMError):
    """Raised when text to Cypher generation fails."""


class EvaluationError(GraphRAGError):
    """Raised when evaluation operations fail."""
```

#### 1.3 Neo4j Client (`neo4j_client.py`)

```python
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
from typing import TYPE_CHECKING, Any, Final

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from jama_mcp_server_graphrag.exceptions import Neo4jConnectionError

if TYPE_CHECKING:
    from neo4j import Driver, Session
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
        logger.info("Neo4j connectivity verified successfully")
        
        return driver
        
    except AuthError as e:
        logger.error("Neo4j authentication failed - check credentials")
        raise Neo4jConnectionError(f"Authentication failed: {e}") from e
    except ServiceUnavailable as e:
        logger.error("Neo4j service unavailable - check URI and network")
        raise Neo4jConnectionError(f"Service unavailable: {e}") from e
    except Exception as e:
        logger.exception("Failed to create Neo4j driver")
        raise Neo4jConnectionError(f"Driver creation failed: {e}") from e


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
    def _execute(tx, query: str, params: dict[str, Any] | None):
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
    def _execute(tx, query: str, params: dict[str, Any] | None):
        result = tx.run(query, params or {})
        return [record.data() for record in result]
    
    with driver.session(database=database) as session:
        return session.execute_write(_execute, query, parameters)


def execute_read_with_bookmark(
    driver: Driver,
    query: str,
    parameters: dict[str, Any] | None = None,
    bookmarks: list | None = None,
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
    def _execute(tx, query: str, params: dict[str, Any] | None):
        result = tx.run(query, params or {})
        return [record.data() for record in result]
    
    with driver.session(database=database, bookmarks=bookmarks) as session:
        results = session.execute_read(_execute, query, parameters)
        return results, session.last_bookmark()


# Example usage pattern for serverless (Vercel/Lambda)
# =============================================================================
# # Module-level driver (created once, reused across requests)
# _driver: Driver | None = None
# 
# def get_driver(config: AppConfig) -> Driver:
#     """Get or create the singleton driver instance."""
#     global _driver
#     if _driver is None:
#         _driver = create_driver(config)
#     return _driver
# =============================================================================
```

#### 1.4 Server Entry Point (`server.py`)

```python
"""FastMCP server entry point for Requirements Management GraphRAG.

This module initializes the MCP server with all GraphRAG tools,
managing the lifecycle of Neo4j connections and vector stores.

Neo4j Driver Best Practices Applied:
- Single driver instance created once and reused
- Connectivity verified at startup (fail fast)
- Connection pool sized for serverless (Vercel)
- Proper cleanup on shutdown
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP

from jama_mcp_server_graphrag.config import get_config
from jama_mcp_server_graphrag.exceptions import Neo4jConnectionError
from jama_mcp_server_graphrag.tools import (
    register_chat_tool,
    register_entity_explorer_tool,
    register_evaluate_tool,
    register_definitions_tool,
    register_graph_retrieval_tool,
    register_hybrid_search_tool,
    register_schema_tool,
    register_standards_tool,
    register_text2cypher_tool,
    register_vector_search_tool,
)

if TYPE_CHECKING:
    from jama_mcp_server_graphrag.config import AppConfig

logger: logging.Logger = logging.getLogger(__name__)


def create_graph(config: AppConfig) -> Neo4jGraph:
    """Create Neo4j graph connection with best practices.
    
    Applies Neo4j driver best practices:
    - Uses neo4j+s:// scheme for secure connections
    - Verifies connectivity immediately (fail fast)
    - Configures connection pool for serverless environments
    
    Args:
        config: Application configuration.
        
    Returns:
        Connected Neo4jGraph instance.
        
    Raises:
        Neo4jConnectionError: If connection verification fails.
    """
    logger.info("Creating Neo4j connection to %s", config.neo4j_uri.split("@")[-1])
    
    try:
        graph = Neo4jGraph(
            url=config.neo4j_uri,
            username=config.neo4j_username,
            password=config.neo4j_password,
            database=config.neo4j_database,
        )
        
        # Best Practice: Verify connectivity immediately
        # This catches configuration errors at startup, not on first query
        graph.query("RETURN 1 AS connected")
        logger.info("Neo4j connectivity verified successfully")
        
        return graph
        
    except Exception as e:
        logger.exception("Failed to connect to Neo4j")
        raise Neo4jConnectionError(
            f"Failed to connect to Neo4j at {config.neo4j_uri}: {e}"
        ) from e


def create_vector_store(config: AppConfig, graph: Neo4jGraph) -> Neo4jVector:
    """Create vector store for similarity search.
    
    Args:
        config: Application configuration.
        graph: Neo4j graph connection.
        
    Returns:
        Configured Neo4jVector instance.
    """
    logger.info("Initializing vector store with index '%s'", config.vector_index_name)
    
    embedding_model = OpenAIEmbeddings(model=config.embedding_model)
    
    vector_store = Neo4jVector.from_existing_index(
        embedding=embedding_model,
        graph=graph,
        index_name=config.vector_index_name,
    )
    
    logger.info("Vector store initialized successfully")
    return vector_store


@asynccontextmanager
async def server_lifespan(mcp_server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage server lifecycle and resources.
    
    Implements Neo4j driver best practices:
    - Creates single driver instance (reused across all requests)
    - Verifies connectivity at startup (fail fast on bad config)
    - Properly cleans up resources on shutdown
    
    Critical for serverless (Vercel): Driver creation is expensive
    (can take seconds). By creating once in lifespan, we avoid
    cold start penalties on every request.
    
    Args:
        mcp_server: The FastMCP server instance.
        
    Yields:
        Dictionary containing shared resources (config, graph, vector_store).
    """
    logger.info("Starting Requirements GraphRAG MCP Server")
    
    config: AppConfig = get_config()
    
    # Log configuration (without sensitive data)
    logger.info(
        "Configuration loaded: database=%s, model=%s, embedding=%s, pool_size=%d",
        config.neo4j_database,
        config.chat_model,
        config.embedding_model,
        config.neo4j_max_connection_pool_size,
    )
    
    # Initialize Neo4j graph connection
    # Best Practice: Create once, reuse for all requests
    graph: Neo4jGraph = create_graph(config)
    
    # Initialize vector store for similarity search
    vector_store: Neo4jVector = create_vector_store(config, graph)
    
    logger.info("All resources initialized - server ready to accept requests")
    
    # Yield resources for tools to access
    yield {
        "config": config,
        "graph": graph,
        "vector_store": vector_store,
    }
    
    # Cleanup on shutdown
    logger.info("Shutting down GraphRAG MCP Server - cleaning up resources")
    # Note: Neo4jGraph handles its own cleanup via __del__


# Initialize the MCP server
mcp = FastMCP(
    name="jama_graphrag_mcp",
    instructions="""
    Requirements Management GraphRAG Server - Intelligent access to requirements 
    engineering knowledge including best practices, industry standards, tools, 
    and methodologies.
    
    Available capabilities:
    
    1. **Semantic Search** (`graphrag_vector_search`)
       Find relevant content using vector similarity on chunk embeddings.
       Best for: "What does the guide say about traceability?"
    
    2. **Hybrid Search** (`graphrag_hybrid_search`)
       Combined vector + keyword search for improved accuracy.
       Best for: "ISO 26262 ASIL requirements" (specific technical terms)
    
    3. **Graph-Enriched Retrieval** (`graphrag_retrieve`)
       Combine semantic search with entity relationships for richer context.
       Best for: "Explain requirements validation with related concepts"
    
    4. **Entity Explorer** (`graphrag_explore_entity`)
       Deep dive into specific concepts, challenges, or best practices.
       Best for: "Tell me about ISO 26262" or "What is requirements traceability?"
    
    5. **Standards Lookup** (`graphrag_standards`)
       Find regulatory standards and compliance requirements.
       Best for: "What standards apply to medical devices?"
    
    6. **Definitions** (`graphrag_definitions`)
       Look up definitions for requirements management terminology.
       Best for: "Define 'baseline'" or "What is an atomic requirement?"
    
    7. **Text2Cypher** (`graphrag_text2cypher`)
       Generate Cypher queries from natural language for complex retrieval.
       Best for: "List all tools by vendor" or "Which chapter has the most entities?"
    
    8. **RAG Chat** (`graphrag_chat`)
       Full conversational Q&A with retrieval-augmented generation.
       Best for: Complex questions requiring synthesis across sources.
    
    9. **Evaluate** (`graphrag_evaluate`)
       Run evaluation metrics on retrieval and answer quality.
       Best for: Testing system performance.
    
    **Recommended Workflow:**
    1. Start with schema exploration to understand available data
    2. Use definitions lookup for unfamiliar terms
    3. Use entity explorer for deep concept understanding
    4. Use RAG chat for complex, multi-faceted questions
    
    **Domain Coverage:**
    - Requirements writing, gathering, and management processes
    - Traceability and impact analysis
    - Validation and verification
    - Industry standards (ISO, FDA, DO-178C, INCOSE, MIL-STD)
    - Industry applications (Automotive, Medical, Aerospace, Defense, etc.)
    - Tools (Jama Connect, DOORS, traditional approaches)
    - Methodologies (Agile, V-Model, Waterfall)
    """,
    lifespan=server_lifespan,
)

# Register all tools
register_vector_search_tool(mcp)
register_hybrid_search_tool(mcp)
register_graph_retrieval_tool(mcp)
register_entity_explorer_tool(mcp)
register_standards_tool(mcp)
register_definitions_tool(mcp)
register_text2cypher_tool(mcp)
register_chat_tool(mcp)
register_schema_tool(mcp)
register_evaluate_tool(mcp)


def main() -> None:
    """Main entry point for the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Requirements GraphRAG MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run the server
    if args.transport == "http":
        mcp.run(transport="streamable_http", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
```

---

## Environment Configuration

### `.env.example`

```bash
# =============================================================================
# Neo4j Connection
# =============================================================================
# Use neo4j+s:// for production (Aura, clusters) - enables TLS and routing
# Use neo4j:// only for local development without TLS
# WARNING: bolt:// does not support cluster routing - avoid for clusters
NEO4J_URI=neo4j+s://your-neo4j-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password-here
NEO4J_DATABASE=neo4j

# Neo4j Driver Configuration (optimized for serverless)
# Smaller pool size reduces cold start time in Vercel/Lambda
NEO4J_MAX_POOL_SIZE=5
# Longer timeout for cold starts in serverless environments
NEO4J_CONNECTION_TIMEOUT=30.0

# =============================================================================
# OpenAI
# =============================================================================
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# =============================================================================
# Vector Index Configuration (aligned with actual database)
# =============================================================================
VECTOR_INDEX_NAME=chunk_embeddings
SIMILARITY_K=6

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL=INFO
```

---

## Quality Checklist

Before considering the implementation complete, verify:

### Strategic Design
- [ ] Tools enable complete GraphRAG workflows
- [ ] Hybrid search combines vector and keyword retrieval
- [ ] Agentic patterns (router, critic) are implemented
- [ ] Tool names are action-oriented and discoverable
- [ ] Response formats are efficient for LLM context
- [ ] Error messages guide toward correct usage

### Implementation Quality
- [ ] All tools have descriptive names and documentation
- [ ] Return types are consistent across operations
- [ ] Error handling covers all external calls
- [ ] All network operations use async/await
- [ ] Common functionality is extracted into reusable functions
- [ ] Text2Cypher includes few-shot examples and terminology mappings

### Neo4j Driver Best Practices
- [ ] Uses `neo4j+s://` scheme for production connections
- [ ] Single driver instance created once (in lifespan handler)
- [ ] Connectivity verified at startup (fail fast)
- [ ] Connection pool sized for serverless (5-10 connections)
- [ ] Query parameters used (no string concatenation)
- [ ] Explicit transaction functions used (`execute_read`/`execute_write`)
- [ ] Results processed within transaction scope

### Tool Configuration
- [ ] All tools implement `name` and `annotations`
- [ ] Annotations correctly set (readOnlyHint, etc.)
- [ ] All tools use Pydantic BaseModel for input validation
- [ ] All fields have types, descriptions, and constraints

### Code Quality
- [ ] ruff check passes with no errors
- [ ] ty check passes with no errors
- [ ] All tests pass
- [ ] Coverage >= 80%

### Evaluation
- [ ] Benchmark dataset covers all tool categories
- [ ] RAGAS metrics (recall, faithfulness, correctness) implemented
- [ ] Evaluation tool available via MCP

### DevOps
- [ ] Dockerfile builds successfully
- [ ] docker-compose runs locally
- [ ] GitHub Actions CI passes
- [ ] Vercel deployment configured

### Documentation
- [ ] README.md with installation and usage instructions
- [ ] CLAUDE.md with development commands
- [ ] All public functions have docstrings
- [ ] Example usage provided

---

## Next Steps After Implementation

1. **Test with MCP Inspector**: `npx @modelcontextprotocol/inspector`
2. **Install in Claude Desktop**: Update `claude_desktop_config.json`
3. **Run evaluation benchmark**: Test retrieval accuracy
4. **Deploy to Vercel**: Push to main branch
5. **Performance optimization**: Profile and optimize hot paths
6. **Add monitoring**: Structured logging and metrics
7. **Documentation**: API docs, architecture diagrams

---

## References

- [FastMCP Documentation](https://gofastmcp.com)
- [MCP Specification](https://modelcontextprotocol.io)
- [LangChain Neo4j](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Neo4j GraphRAG](https://neo4j.com/labs/genai-ecosystem/neo4j-graphrag/)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)

### Neo4j Driver Best Practices Reference

- [Neo4j Driver Best Practices (Neo4j Blog)](https://neo4j.com/blog/developer/neo4j-driver-best-practices/)
  - Connection URI schemes (`neo4j+s://` for production)
  - Driver instance management (create once, reuse)
  - Connectivity verification at startup
  - Explicit transaction functions (`execute_read`/`execute_write`)
  - Query parameters for security and performance
  - Result processing within transaction scope
  - Bookmarking for causal consistency
  - Serverless optimization (connection pool sizing)
- [Neo4j Python Driver Documentation](https://neo4j.com/docs/python-manual/current/)
- [Cypher Injection Prevention](https://neo4j.com/developer/kb/protecting-against-cypher-injection/)

### Essential GraphRAG Reference

- **"Essential GraphRAG" by Tomaž Bratanič and Oskar Hane (Manning, 2025)**
  - Chapter 2: Vector Similarity Search and Hybrid Search
  - Chapter 3: Advanced Vector Retrieval Strategies (Step-Back, Parent Document)
  - Chapter 4: Text2Cypher Best Practices (Few-Shot, Terminology Mapping)
  - Chapter 5: Agentic RAG (Router, Critic, Query Updating)
  - Chapter 7: Microsoft's GraphRAG (Community Detection, Global/Local Search)
  - Chapter 8: RAG Application Evaluation (RAGAS Metrics)
- [GitHub Repository: kg-rag](https://github.com/tomasonjo/kg-rag)

### Domain-Specific Resources

- [INCOSE Guide for Writing Requirements](https://www.incose.org/)
- [ISO 26262 - Automotive Functional Safety](https://www.iso.org/standard/68383.html)
- [FDA Design Controls Guidance](https://www.fda.gov/medical-devices)
- [DO-178C - Software Considerations in Airborne Systems](https://www.rtca.org/)
- [Jama Software Requirements Management Guide](https://www.jamasoftware.com/requirements-management-guide/)

---

## Summary of MCP Tools

| Tool Name | Purpose | Best For |
|-----------|---------|----------|
| `graphrag_vector_search` | Basic semantic search | Quick content lookup |
| `graphrag_hybrid_search` | Vector + keyword search | Technical terms, acronyms |
| `graphrag_retrieve` | Graph-enriched retrieval | Understanding concept relationships |
| `graphrag_explore_entity` | Entity deep-dive | Learning about specific concepts |
| `graphrag_standards` | Standards lookup | Compliance and regulatory questions |
| `graphrag_definitions` | Term definitions | Understanding terminology |
| `graphrag_text2cypher` | Natural language to Cypher | Complex queries, aggregations |
| `graphrag_chat` | Full RAG Q&A | Complex, multi-faceted questions |
| `graphrag_schema` | Schema exploration | Understanding graph structure |
| `graphrag_evaluate` | Evaluation metrics | Testing system performance |

---

## REST API Endpoints (React Frontend)

The REST API exposes the same GraphRAG capabilities as the MCP tools, designed for consumption by a React frontend chatbot application.

### API Entry Point (`api.py`)

```python
"""FastAPI REST API entry point for React frontend integration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jama_mcp_server_graphrag.config import AppConfig
from jama_mcp_server_graphrag.neo4j_client import create_driver
from jama_mcp_server_graphrag.routes import chat, definitions, health, schema, search, standards

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle - initialize and cleanup resources."""
    config = AppConfig.from_env()
    driver = create_driver(config)
    
    # Store in app state for route access
    app.state.config = config
    app.state.driver = driver
    
    yield
    
    driver.close()


app = FastAPI(
    title="Jama GraphRAG API",
    description="GraphRAG backend for Jama Requirements Management knowledge base",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router, tags=["health"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(schema.router, prefix="/api/v1", tags=["schema"])
app.include_router(definitions.router, prefix="/api/v1", tags=["definitions"])
app.include_router(standards.router, prefix="/api/v1", tags=["standards"])
```

### Endpoint Specifications

#### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "neo4j": "connected",
  "version": "1.0.0"
}
```

#### Chat (Main RAG Endpoint)

```
POST /api/v1/chat
```

Request:
```json
{
  "message": "What is requirements traceability?",
  "conversation_id": "optional-uuid-for-history",
  "options": {
    "retrieval_strategy": "hybrid",
    "include_sources": true,
    "max_sources": 5
  }
}
```

Response:
```json
{
  "answer": "Requirements traceability is the ability to...",
  "sources": [
    {
      "title": "Understanding Requirements Traceability",
      "url": "https://www.jamasoftware.com/...",
      "chunk_id": "chunk-123",
      "relevance_score": 0.92
    }
  ],
  "entities": [
    {"name": "Requirements Traceability", "type": "Concept"},
    {"name": "Traceability Matrix", "type": "Artifact"}
  ],
  "conversation_id": "uuid"
}
```

#### Search Endpoints

```
POST /api/v1/search/vector
POST /api/v1/search/hybrid
POST /api/v1/search/graph
```

Request (all search endpoints):
```json
{
  "query": "ISO 26262 compliance",
  "limit": 10,
  "filters": {
    "chapter": "Meeting Regulatory Compliance",
    "entity_types": ["Standard", "Concept"]
  }
}
```

Response:
```json
{
  "results": [
    {
      "content": "...",
      "title": "...",
      "url": "...",
      "score": 0.89,
      "entities": [...]
    }
  ],
  "total": 10,
  "strategy": "hybrid"
}
```

#### Text2Cypher

```
POST /api/v1/search/cypher
```

Request:
```json
{
  "question": "Which chapters have the most articles about medical devices?",
  "validate": true
}
```

Response:
```json
{
  "cypher": "MATCH (c:Chapter)-[:CONTAINS]->(a:Article) WHERE ...",
  "results": [...],
  "explanation": "This query finds chapters containing articles about medical devices..."
}
```

#### Schema

```
GET /api/v1/schema
GET /api/v1/schema/entities
GET /api/v1/schema/relationships
```

#### Definitions

```
GET /api/v1/definitions
GET /api/v1/definitions/{term}
GET /api/v1/definitions/search?q=trace
```

#### Standards

```
GET /api/v1/standards
GET /api/v1/standards/{name}
GET /api/v1/standards/industry/{industry}
```

### Route Implementation Pattern

Routes should be thin wrappers around `core/` modules:

```python
# routes/chat.py
"""Chat route handler - wraps core chat workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from jama_mcp_server_graphrag.core.generation import generate_answer
from jama_mcp_server_graphrag.core.retrieval import hybrid_search

if TYPE_CHECKING:
    from neo4j import Driver

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None
    options: dict | None = None


class SourceInfo(BaseModel):
    """Source citation information."""

    title: str
    url: str | None
    chunk_id: str | None
    relevance_score: float


class EntityInfo(BaseModel):
    """Related entity information."""

    name: str
    type: str | None = None


class ImageInfo(BaseModel):
    """Image from knowledge base.

    Images are retrieved from articles related to the search results.
    URLs point to externally hosted images (Jama CDN).
    """

    url: str
    alt_text: str = ""
    context: str = ""
    source_title: str = ""


class ChatResponse(BaseModel):
    """Chat response model.

    Includes answer, sources, entities, and relevant images from the
    knowledge graph. Images are limited to 5 per response.
    """

    answer: str
    sources: list[SourceInfo]
    entities: list[EntityInfo]
    images: list[ImageInfo] = []
    conversation_id: str | None


def get_driver(request: Request) -> Driver:
    """Dependency to get Neo4j driver from app state."""
    return request.app.state.driver


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    driver: Driver = Depends(get_driver),
) -> ChatResponse:
    """Process a chat message and return RAG response."""
    # Use core modules - same logic as MCP tools
    contexts = await hybrid_search(driver, request.message, limit=5)
    answer, sources, entities = await generate_answer(
        query=request.message,
        contexts=contexts,
    )
    
    return ChatResponse(
        answer=answer,
        sources=sources,
        entities=entities,
        images=images,
        conversation_id=request.conversation_id,
    )
```

### Running the REST API

```bash
# Development
uv run uvicorn jama_mcp_server_graphrag.api:app --reload --port 8000

# Production
uv run uvicorn jama_mcp_server_graphrag.api:app --host 0.0.0.0 --port 8000

# With Docker
docker compose up api
```

### OpenAPI Documentation

When running, access interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Core Module Pattern

The `core/` directory contains shared logic used by both MCP tools and REST API routes. This avoids code duplication and ensures consistent behavior across interfaces.

```python
# core/retrieval.py
"""Core retrieval logic - shared by tools/ and routes/."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import Driver


async def vector_search(
    driver: Driver,
    query: str,
    *,
    limit: int = 10,
    filters: dict | None = None,
) -> list[dict]:
    """Execute vector similarity search."""
    # Implementation here - used by both MCP tool and REST route
    ...


async def hybrid_search(
    driver: Driver,
    query: str,
    *,
    limit: int = 10,
    keyword_weight: float = 0.3,
) -> list[dict]:
    """Execute hybrid vector + keyword search."""
    ...


async def graph_enriched_search(
    driver: Driver,
    query: str,
    *,
    limit: int = 10,
    traversal_depth: int = 1,
) -> list[dict]:
    """Execute vector search with graph traversal enrichment."""
    ...
```

### Usage in MCP Tools vs REST Routes

```python
# tools/hybrid_search.py (MCP)
from jama_mcp_server_graphrag.core.retrieval import hybrid_search

@mcp.tool()
async def graphrag_hybrid_search(query: str, limit: int = 10) -> list[dict]:
    driver = mcp.state["driver"]
    return await hybrid_search(driver, query, limit=limit)


# routes/search.py (REST)
from jama_mcp_server_graphrag.core.retrieval import hybrid_search

@router.post("/search/hybrid")
async def search_hybrid(request: SearchRequest, driver = Depends(get_driver)):
    return await hybrid_search(driver, request.query, limit=request.limit)
```

This pattern ensures:
1. **Single source of truth** for GraphRAG logic
2. **Consistent behavior** across MCP and REST interfaces
3. **Easier testing** - test core logic once
4. **DRY principle** - no duplicated code
