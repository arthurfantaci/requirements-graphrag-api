# Jama MCP Server GraphRAG

## Project Overview

Production-ready GraphRAG backend for the **Jama Software "Essential Guide to Requirements Management and Traceability"** knowledge base. This project serves two purposes:

1. **MCP Server**: Exposes GraphRAG tools to Claude Desktop via Model Context Protocol
2. **REST API Backend**: Powers a React frontend chatbot for querying requirements management content

Both interfaces share the same core GraphRAG logic, Neo4j connection, and retrieval workflows.

## Architecture

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

## Tech Stack

- Python 3.12+ with uv package manager
- FastMCP for MCP server implementation
- FastAPI for REST API endpoints
- LangChain + langchain-neo4j for GraphRAG
- LangGraph for agentic workflows
- Neo4j AuraDB (neo4j+s:// connection)
- OpenAI embeddings (text-embedding-3-small)
- Docker for containerization
- Vercel for serverless deployment

## Project Structure

```
src/jama_mcp_server_graphrag/
├── server.py              # FastMCP entry point with MCP tools
├── api.py                 # FastAPI REST endpoints for React frontend
├── config.py              # Immutable dataclass configuration
├── exceptions.py          # Custom exception hierarchy
├── neo4j_client.py        # Neo4j driver best practices wrapper
├── observability.py       # LangSmith tracing integration
├── mlflow_tracking.py     # MLflow experiment tracking
├── observability_comparison.py  # Platform comparison utilities
├── token_counter.py       # Token counting and cost estimation
├── core/                  # Shared GraphRAG logic
│   ├── retrieval.py       # Vector, hybrid, graph-enriched search
│   ├── text2cypher.py     # Natural language to Cypher
│   ├── generation.py      # Answer generation with citations
│   ├── definitions.py     # Definition/glossary term lookups
│   └── standards.py       # Standards reference queries
├── evaluation/            # Evaluation framework
│   ├── metrics.py         # RAGAS metrics integration
│   ├── domain_metrics.py  # Domain-specific metrics (citation, traceability)
│   ├── cost_metrics.py    # Cost and budget tracking
│   ├── datasets.py        # Evaluation dataset utilities
│   └── runner.py          # Evaluation execution runner
├── prompts/               # Prompt templates and management
│   ├── catalog.py         # Prompt catalog with versioning
│   ├── definitions.py     # Definition-related prompts
│   └── evaluation.py      # Evaluation prompts
├── routes/                # FastAPI route handlers
│   ├── chat.py            # Chat/RAG endpoints
│   ├── search.py          # Search endpoints
│   ├── definitions.py     # Definition lookups
│   ├── standards.py       # Standards queries
│   ├── schema.py          # Schema introspection
│   └── health.py          # Health checks
├── agentic/               # Agentic RAG patterns
│   ├── router.py          # Query routing logic
│   ├── stepback.py        # Step-back prompting
│   ├── critic.py          # Answer validation
│   └── query_updater.py   # Query refinement
└── workflows/             # LangGraph workflows
    ├── rag_workflow.py    # Standard RAG workflow
    ├── agentic_workflow.py # Agentic workflow with routing
    └── state.py           # Workflow state definitions
```

## Commands

```bash
# Development
uv sync                           # Install dependencies
uv run pytest                     # Run tests
uv run pytest --cov --cov-report=html  # Coverage report
uv run ruff check src/            # Lint
uv run ruff format src/           # Format

# Run MCP Server (for Claude Desktop)
uv run jama-mcp-server-graphrag

# Run REST API (for React frontend)
uv run uvicorn jama_mcp_server_graphrag.api:app --reload

# Docker
docker build -t jama-mcp-graphrag:latest .
docker compose up -d

# Testing
npx @modelcontextprotocol/inspector  # Test MCP server
curl http://localhost:8000/docs      # OpenAPI docs for REST API
```

## Code Style

- Use `from __future__ import annotations` in all modules
- Prefer `dataclass(frozen=True, slots=True)` for immutable configs
- Use explicit type hints everywhere
- Follow Neo4j driver best practices (see neo4j_client.py)
- Use query parameters, never string concatenation for Cypher
- Process Neo4j results within transaction scope
- Keep core logic in `core/`, thin wrappers in `tools/` and `routes/`

## Neo4j Best Practices (CRITICAL)

- Always use `neo4j+s://` for production URIs
- Create driver once in lifespan, reuse across requests
- Verify connectivity at startup with `driver.verify_connectivity()`
- Use `execute_read()` and `execute_write()` for proper cluster routing
- Keep connection pool small (5-10) for serverless

## Testing

- Unit tests in tests/test_*.py
- Integration tests marked with @pytest.mark.integration
- API tests in tests/test_api.py
- Benchmark tests in tests/benchmark/
- Target 80% coverage minimum

## Workflow

1. Plan before implementing (use /plan command)
2. Write tests first (TDD)
3. Implement incrementally
4. Run `uv run ruff check && uv run pytest` before commits
5. Use conventional commits (feat:, fix:, docs:, etc.)

## Do Not

- Edit .env files directly (use .env.example as template)
- Use bolt:// for cluster connections
- Create new driver instances per request
- Use string concatenation in Cypher queries
- Commit sensitive credentials
- Duplicate logic between MCP tools and REST routes (use core/)

## Project Specification

See SPECIFICATION.md for complete implementation details including:
- Knowledge graph data model
- MCP tool definitions
- REST API endpoint definitions
- Neo4j driver patterns
- Docker and CI/CD configuration
- Quality checklist
