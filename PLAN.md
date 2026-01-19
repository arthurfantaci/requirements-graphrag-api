# Comprehensive Update Plan: jama-mcp-server-graphrag

> **STATUS: COMPLETED**
>
> This plan was fully implemented between 2026-01-17 and 2026-01-19.
> All phases have been completed and verified. This document is retained
> for historical reference.
>
> **Post-Implementation Updates (2026-01-19):**
> - Added images to GraphRAG chat responses
> - Fixed Neo4j 5.x syntax (`size()` → `COUNT {}` for patterns)
> - Updated MCP tool docstrings with image handling guidance

**Date**: 2026-01-17
**Purpose**: Update the project to support the new Neo4j graph data model and vector retrieval patterns

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Data Model Changes](#2-data-model-changes)
3. [Implementation Plan](#3-implementation-plan)
4. [File-by-File Update Guide](#4-file-by-file-update-guide)
5. [Ruff LSP Migration](#5-ruff-lsp-migration)
6. [Testing Strategy](#6-testing-strategy)
7. [Rollout Checklist](#7-rollout-checklist)

---

## 1. Executive Summary

### Scope of Changes

The Neo4j knowledge graph has been restructured with:
- **Reversed relationship directions** for chunk-entity connections
- **New node types** for media content (Image, Video, Webinar)
- **Renamed node label**: `GlossaryTerm` → `Definition`
- **Updated embedding model**: `text-embedding-ada-002` → `text-embedding-3-small`
- **Different retrieval library**: Using `neo4j-graphrag` VectorRetriever instead of LangChain's Neo4jVector

### Impact Assessment

| Component | Impact Level | Effort |
|-----------|-------------|--------|
| `core/retrieval.py` | **HIGH** | Major rewrite |
| `core/glossary.py` | **HIGH** | Replace GlossaryTerm with Definition |
| `core/text2cypher.py` | **MEDIUM** | Update schema context |
| `server.py` | **MEDIUM** | Update initialization |
| `api.py` | **MEDIUM** | Update initialization |
| `config.py` | **LOW** | Change default embedding model |
| Routes | **LOW** | Minor query adjustments |
| `.vscode/settings.json` | **LOW** | Remove deprecated setting |

---

## 2. Data Model Changes

### 2.1 Node Labels Comparison

| Label | OLD Count | NEW Count | Status |
|-------|-----------|-----------|--------|
| Chapter | 15 | 15 | Unchanged |
| Article | 103 | 103 | Properties changed |
| Chunk | 804 | 2,159 | Properties: `text`, `embedding`, `index` |
| Concept | 1,199 | 1,523 | Increased |
| Challenge | 680 | 839 | Increased |
| Bestpractice | 444 | 330 | Decreased |
| Artifact | 434 | 601 | Increased |
| Role | 194 | 181 | Decreased |
| Processstage | 190 | 285 | Increased |
| Standard | 111 | 123 | Increased |
| Tool | 79 | 159 | Increased |
| Methodology | 47 | 30 | Decreased |
| Industry | 14 | 18 | Increased |
| GlossaryTerm | 134 | **0** | **REMOVED** |
| **Definition** | N/A | **134** | **NEW** |
| **Image** | N/A | **163** | **NEW** |
| **Video** | N/A | **1** | **NEW** |
| **Webinar** | N/A | **38** | **NEW** |

### 2.2 Relationship Changes

#### Removed Relationships
- `MENTIONS_ENTITY` (Chunk → Entity)
- `MENTIONS` (Article → Entity)
- `MENTIONS_TERM` (Chunk → GlossaryTerm)
- `HAS_CHUNK` (Article → Chunk)
- `RELATED_TO_TERM` (Entity → GlossaryTerm)
- `DEFINED_BY` (Entity → GlossaryTerm)

#### New Relationships
| Relationship | Pattern | Count | Notes |
|--------------|---------|-------|-------|
| `FROM_ARTICLE` | (Chunk)-[:FROM_ARTICLE]->(Article) | 2,159 | Reversed from HAS_CHUNK |
| `NEXT_CHUNK` | (Chunk)-[:NEXT_CHUNK]->(Chunk) | 2,056 | Sequential ordering |
| `MENTIONED_IN` | (Entity)-[:MENTIONED_IN]->(Chunk) | 8,524 | Reversed direction! |
| `HAS_IMAGE` | (Article)-[:HAS_IMAGE]->(Image) | 163 | Media support |
| `HAS_VIDEO` | (Article)-[:HAS_VIDEO]->(Video) | 1 | Media support |
| `HAS_WEBINAR` | (Article)-[:HAS_WEBINAR]->(Webinar) | 38 | Media support |
| `REFERENCES` | (Article)-[:REFERENCES]->(Article) | 29 | Cross-references |

#### Modified Relationships
| Relationship | Count | Changes |
|--------------|-------|---------|
| `ADDRESSES` | 703 | Optional `effectiveness` property |
| `RELATED_TO` | 807 | Optional `relationship_nature` property |
| `PRODUCES` | 97 | (Processstage/Role) → Artifact |
| `USED_BY` | 132 | Multiple source types |
| `REQUIRES` | 434 | Multiple patterns |
| `COMPONENT_OF` | 283 | Hierarchical structure |
| `DEFINES` | 77 | Standard → Concept/Artifact |
| `APPLIES_TO` | 139 | Multiple patterns |

### 2.3 Chunk Node Changes

**OLD Chunk Structure:**
```
(:Chunk)
  - heading
  - chunk_type
  - (embedding stored separately or via Article)
```

**NEW Chunk Structure:**
```
(:Chunk)
  - text        # Actual content for retrieval
  - embedding   # 1536-dim vector
  - index       # Sequential position
```

### 2.4 Vector Index Configuration

```json
{
  "name": "chunk_embeddings",
  "type": "VECTOR",
  "entityType": "NODE",
  "labelsOrTypes": ["Chunk"],
  "properties": ["embedding"],
  "indexProvider": "vector-3.0",
  "state": "ONLINE"
}
```

### 2.5 Embedding Model Change

| Aspect | OLD | NEW |
|--------|-----|-----|
| Model | `text-embedding-ada-002` | `text-embedding-3-small` |
| Dimensions | 1536 | 1536 |
| Provider | OpenAI | OpenAI |
| Library | LangChain `OpenAIEmbeddings` | `neo4j_graphrag.embeddings.OpenAIEmbeddings` |

---

## 3. Implementation Plan

### Phase 1: Configuration & Dependencies (Priority: HIGH)

#### 3.1.1 Update `pyproject.toml`

Add `neo4j-graphrag` dependency:

```toml
dependencies = [
    # ... existing deps
    "neo4j-graphrag>=1.0.0",  # Add this
]
```

#### 3.1.2 Update `config.py`

```python
# Change default embedding model
embedding_model: str = "text-embedding-3-small"  # was text-embedding-ada-002
```

### Phase 2: Core Retrieval Rewrite (Priority: HIGH)

#### 3.2.1 Rewrite `core/retrieval.py`

**Key Changes:**
1. Switch from `Neo4jVector` to `neo4j_graphrag.retrievers.VectorRetriever`
2. Update entity traversal to use reversed `MENTIONED_IN` direction
3. Add new retrieval query that fetches chunk text directly
4. Update graph-enriched search patterns

**New Retrieval Pattern:**
```python
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

# Vector search on Chunk nodes
retriever = VectorRetriever(
    driver=driver,
    index_name="chunk_embeddings",
    embedder=OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key),
    return_properties=["text"],
)
results = retriever.search(query_text=query, top_k=limit)
```

**Entity Traversal (Reversed Direction):**
```cypher
# OLD: MATCH (c:Chunk)-[:MENTIONS_ENTITY]->(entity)
# NEW:
MATCH (entity)-[:MENTIONED_IN]->(c:Chunk)
WHERE elementId(c) IN $chunk_ids
RETURN labels(entity)[0] AS label, entity.name AS name
```

**Article Context:**
```cypher
# OLD: MATCH (article:Article)-[:HAS_CHUNK]->(chunk)
# NEW:
MATCH (chunk:Chunk)-[:FROM_ARTICLE]->(article:Article)
RETURN article.article_title AS title, article.url AS url
```

### Phase 3: Glossary Module Update (Priority: HIGH)

#### 3.3.1 Rename/Rewrite `core/glossary.py`

**Option A**: Rename to `core/definitions.py` (Recommended)
**Option B**: Keep filename, update all queries

**Query Changes:**
```cypher
# OLD:
MATCH (t:GlossaryTerm)
WHERE toLower(t.term) CONTAINS toLower($term)
RETURN t.term, t.definition

# NEW:
MATCH (d:Definition)
WHERE toLower(d.term) CONTAINS toLower($term)
RETURN d.term, d.definition, d.url, d.term_id
```

**Note**: `Definition` nodes have additional properties: `url`, `term_id`

### Phase 4: Text2Cypher Schema Update (Priority: MEDIUM)

#### 3.4.1 Update `core/text2cypher.py`

Update the schema context in the system prompt:

```python
SCHEMA_CONTEXT = """
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
- (Chunk)-[:FROM_ARTICLE]->(Article)
- (Chunk)-[:NEXT_CHUNK]->(Chunk)
- (Entity)-[:MENTIONED_IN]->(Chunk)
- (Article)-[:HAS_IMAGE]->(Image)
- (Article)-[:HAS_VIDEO]->(Video)
- (Article)-[:HAS_WEBINAR]->(Webinar)
- (Article)-[:REFERENCES]->(Article)
- (Concept)-[:ADDRESSES]->(Challenge)
- (Concept)-[:REQUIRES]->(Concept|Artifact)
- (Concept)-[:COMPONENT_OF]->(Concept)
- (Standard)-[:DEFINES]->(Concept|Artifact)
- (Standard)-[:APPLIES_TO]->(Industry)
"""
```

Update few-shot examples to reflect new patterns.

### Phase 5: Server & API Initialization (Priority: MEDIUM)

#### 3.5.1 Update `server.py`

```python
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

# In lifespan handler
async with lifespan_context() as resources:
    embedder = OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=config.openai_api_key,
    )
    retriever = VectorRetriever(
        driver=driver,
        index_name=config.vector_index_name,
        embedder=embedder,
        return_properties=["text"],
    )
    resources.retriever = retriever
```

#### 3.5.2 Update `api.py`

Same pattern as server.py for FastAPI initialization.

### Phase 6: Route Updates (Priority: LOW)

#### 3.6.1 Update `routes/glossary.py`

- Change all `GlossaryTerm` references to `Definition`
- Update response models to include new properties (`url`, `term_id`)

#### 3.6.2 Update `routes/schema.py`

- Update schema introspection queries
- Include new node labels (Image, Video, Webinar, Definition)
- Include new relationships

### Phase 7: MCP Tool Updates (Priority: LOW)

Update tool implementations to use new core functions:
- `graphrag_lookup_term` → Use Definition queries
- `graphrag_graph_enriched_search` → Use new traversal patterns

---

## 4. File-by-File Update Guide

### 4.1 `src/jama_mcp_server_graphrag/config.py`

**Changes:**
```python
# Line ~XX: Change default embedding model
embedding_model: str = field(default="text-embedding-3-small")
```

### 4.2 `src/jama_mcp_server_graphrag/core/retrieval.py`

**Complete Rewrite Required**

Key function changes:

| Function | Change Type | Notes |
|----------|-------------|-------|
| `vector_search()` | Major | Use VectorRetriever |
| `hybrid_search()` | Major | Update fulltext index usage |
| `graph_enriched_search()` | Major | Reverse MENTIONED_IN direction |
| `get_entities_from_chunks()` | Major | New Cypher pattern |
| `explore_entity()` | Minor | Update relationship patterns |

### 4.3 `src/jama_mcp_server_graphrag/core/glossary.py`

**Rename to `core/definitions.py` (Recommended)**

| Function | OLD Query | NEW Query |
|----------|-----------|-----------|
| `lookup_term()` | `MATCH (t:GlossaryTerm)` | `MATCH (d:Definition)` |
| `search_terms()` | `MATCH (t:GlossaryTerm)` | `MATCH (d:Definition)` |
| `list_terms()` | `MATCH (t:GlossaryTerm)` | `MATCH (d:Definition)` |

### 4.4 `src/jama_mcp_server_graphrag/core/text2cypher.py`

**Updates:**
1. Replace schema context string
2. Update 8+ few-shot examples
3. Add new relationship patterns

### 4.5 `src/jama_mcp_server_graphrag/server.py`

**Updates:**
1. Import new retriever classes
2. Update lifespan initialization
3. Update resource holder class
4. Modify tool implementations

### 4.6 `src/jama_mcp_server_graphrag/api.py`

**Same updates as server.py**

### 4.7 `src/jama_mcp_server_graphrag/routes/glossary.py`

**Rename to `routes/definitions.py`**
- Update endpoint paths: `/glossary` → `/definitions`
- Update Pydantic models
- Update query functions

### 4.8 `src/jama_mcp_server_graphrag/routes/schema.py`

**Updates:**
1. Add new node labels to schema response
2. Update relationship counts
3. Include media types

---

## 5. Ruff LSP Migration

### 5.1 Issue

The VS Code extension "ruff" has deprecated the legacy `ruff-lsp` server. The following setting is no longer supported:

```json
"ruff.format.args": []
```

### 5.2 Solution

**Update `.vscode/settings.json`:**

```diff
  // Ruff extension settings
  "ruff.path": ["${workspaceFolder}/.venv/bin/ruff"],
  "ruff.importStrategy": "fromEnvironment",
  "ruff.lint.enable": true,
- "ruff.format.args": [],
```

**Remove line 20**: `"ruff.format.args": [],`

### 5.3 Migration Reference

For additional migration details, see:
- [Ruff Editor Migration Guide](https://docs.astral.sh/ruff/editors/migration/)
- [GitHub Discussion #15991](https://github.com/astral-sh/ruff/discussions/15991)

### 5.4 New Ruff Settings (Optional Enhancements)

The new native Ruff extension supports these settings:

```json
{
  "ruff.nativeServer": "on",
  "ruff.configuration": "${workspaceFolder}/pyproject.toml",
  "ruff.lint.run": "onSave"
}
```

---

## 6. Testing Strategy

### 6.1 Unit Tests to Update

| Test File | Changes Required |
|-----------|------------------|
| `tests/test_retrieval.py` | Mock new VectorRetriever |
| `tests/test_glossary.py` | Rename to `test_definitions.py` |
| `tests/test_text2cypher.py` | Update expected queries |
| `tests/test_api.py` | Update endpoint paths |

### 6.2 Integration Tests

Create `tests/integration/test_new_schema.py`:

```python
@pytest.mark.integration
def test_vector_search_returns_chunk_text():
    """Verify vector search returns Chunk.text property."""
    results = vector_search(driver, "requirements tracing")
    assert results.items[0].content.get("text") is not None

@pytest.mark.integration
def test_entity_mentioned_in_direction():
    """Verify MENTIONED_IN traversal works from entity to chunk."""
    results = get_entities_from_chunks(driver, chunk_ids)
    assert len(results) > 0

@pytest.mark.integration
def test_definition_lookup():
    """Verify Definition node queries work."""
    result = lookup_term(driver, "requirement")
    assert result is not None
```

### 6.3 Validation Script

Use the provided `test_query.py` as reference for testing new retrieval patterns:

```bash
uv run python test_query.py "What is impact analysis?"
```

---

## 7. Rollout Checklist

### Pre-Implementation

- [x] Backup current working code (create git branch)
- [x] Verify Neo4j database connectivity
- [x] Confirm vector index is ONLINE
- [x] Document current API responses for comparison

### Phase 1: Configuration

- [x] Update `pyproject.toml` with new dependencies
- [x] Run `uv sync` to install `neo4j-graphrag`
- [x] Update `config.py` default embedding model
- [x] Remove deprecated ruff setting from `.vscode/settings.json`

### Phase 2: Core Changes

- [x] Rewrite `core/retrieval.py`
- [x] Rename/rewrite `core/glossary.py` → `core/definitions.py`
- [x] Update `core/text2cypher.py` schema context
- [x] Update `core/standards.py` if needed

### Phase 3: Server Updates

- [x] Update `server.py` initialization
- [x] Update `api.py` initialization
- [x] Update MCP tool implementations

### Phase 4: Route Updates

- [x] Update/rename `routes/glossary.py`
- [x] Update `routes/schema.py`
- [x] Update other routes as needed

### Phase 5: Testing

- [x] Run `uv run ruff check src/`
- [x] Run `uv run pytest`
- [x] Test MCP server with Inspector: `npx @modelcontextprotocol/inspector`
- [x] Test REST API: `curl http://localhost:8000/docs`
- [x] Verify vector search returns correct results
- [x] Verify entity traversal works

### Phase 6: Documentation

- [x] Update CLAUDE.md if needed
- [x] Update SPECIFICATION.md with new schema
- [ ] Update README.md examples *(deferred)*

### Post-Implementation

- [x] Commit changes with conventional commit message
- [x] Create pull request
- [x] Deploy to staging environment
- [x] Verify production deployment

---

## Appendix A: New Cypher Query Patterns

### A.1 Vector Search with Article Context

```cypher
// After retrieving chunks via vector search
MATCH (c:Chunk)-[:FROM_ARTICLE]->(a:Article)
WHERE elementId(c) IN $chunk_ids
RETURN c.text AS text, a.article_title AS title, a.url AS url
```

### A.2 Entity Extraction from Chunks

```cypher
MATCH (entity)-[:MENTIONED_IN]->(c:Chunk)
WHERE elementId(c) IN $chunk_ids
WITH entity, labels(entity)[0] AS label, count(*) AS mentions
RETURN label, entity.name AS name, entity.display_name AS display_name,
       entity.definition AS definition, mentions
ORDER BY mentions DESC, label, name
LIMIT 20
```

### A.3 Definition Lookup

```cypher
MATCH (d:Definition)
WHERE toLower(d.term) CONTAINS toLower($search_term)
RETURN d.term AS term, d.definition AS definition,
       d.url AS url, d.term_id AS term_id
ORDER BY size(d.term)
LIMIT 10
```

### A.4 Media Retrieval

```cypher
MATCH (a:Article)-[:HAS_IMAGE]->(img:Image)
WHERE a.article_id = $article_id
RETURN img.url AS url, img.alt_text AS alt_text, img.context AS context
```

### A.5 Related Entities (New Pattern)

```cypher
MATCH (n {name: $entity_name})-[r]-(related)
WHERE NOT related:Chunk AND NOT related:Article
WITH type(r) AS rel_type,
     labels(related)[0] AS related_label,
     related.name AS related_name,
     related.display_name AS related_display,
     startNode(r) = n AS outgoing
RETURN rel_type,
       CASE WHEN outgoing THEN '->' ELSE '<-' END AS direction,
       related_label, related_name, related_display
ORDER BY rel_type, related_label
```

---

## Appendix B: Dependencies

### Current Dependencies (relevant)
```toml
langchain = "^0.3"
langchain-neo4j = "^0.2"
langchain-openai = "^0.2"
```

### New Dependencies to Add
```toml
neo4j-graphrag = "^1.0"
```

Note: The `neo4j-graphrag` package provides `VectorRetriever` and `OpenAIEmbeddings` which offer direct Neo4j integration without LangChain abstraction layer.

---

*Plan created: 2026-01-17*
*Last updated: 2026-01-17*
