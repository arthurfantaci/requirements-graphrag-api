# Requirements GraphRAG API

## Project Overview

Monorepo containing a GraphRAG backend and React frontend for the **Jama Software "Essential Guide to Requirements Management and Traceability"** knowledge base.

## Architecture

```
requirements-graphrag-api/
├── backend/                    # Python FastAPI + GraphRAG
│   ├── src/requirements_graphrag_api/
│   │   ├── api.py             # FastAPI application
│   │   ├── config.py          # Configuration management
│   │   ├── core/              # GraphRAG logic
│   │   ├── routes/            # API endpoints
│   │   └── observability.py   # LangSmith tracing
│   ├── tests/                 # Backend tests
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/                   # React + Vite + Tailwind
│   ├── src/
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml          # Local development
└── railway.toml                # Railway deployment config
```

## Tech Stack

**Backend:**
- Python 3.12+ with uv package manager
- FastAPI for REST API
- LangChain + langchain-neo4j for GraphRAG
- Neo4j AuraDB
- OpenAI embeddings (text-embedding-3-small)
- LangSmith for observability

**Frontend:**
- React 18
- Vite
- Tailwind CSS v4

**Deployment:**
- Backend: Railway (Docker)
- Frontend: Vercel

## Commands

```bash
# Backend development
cd backend
uv sync --extra dev
uv run pytest
uv run ruff check src/
uv run uvicorn requirements_graphrag_api.api:app --reload

# Frontend development
cd frontend
npm install
npm run dev
npm run build

# Docker (both services)
docker-compose up

# LangSmith debugging (dev only)
uv run langsmith-fetch traces --limit 5 --format json
```

## Code Style

- Use `from __future__ import annotations` in all Python modules
- Prefer `dataclass(frozen=True, slots=True)` for immutable configs
- Use explicit type hints everywhere
- Follow Neo4j driver best practices
- Use query parameters, never string concatenation for Cypher

## Neo4j Best Practices (CRITICAL)

- Always use `neo4j+s://` for production URIs
- Create driver once in lifespan, reuse across requests
- Verify connectivity at startup with `driver.verify_connectivity()`
- Use `execute_read()` and `execute_write()` for proper cluster routing

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
