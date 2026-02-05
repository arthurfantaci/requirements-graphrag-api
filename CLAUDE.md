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

## LangGraph / Agentic Module

- Agentic RAG implementation: `core/agentic/` (state, tools, nodes, subgraphs, orchestrator)
- `langgraph-checkpoint-postgres` requires `psycopg-binary` (not just `psycopg`)
- Verify LangGraph via imports (`from langgraph.graph import StateGraph`) - no `__version__` attribute
- Use `neo4j_graphrag.retrievers.VectorRetriever` (not `langchain_neo4j.Neo4jVector`)
- State patterns: `Annotated[list, add_messages]` for messages, `TypedDict` with `total=False`
- Tool factory: `create_agent_tools(config, driver, retriever)` binds dependencies via closures
- Push prompts: `langsmith.Client().push_prompt(name, object=template, description=..., tags=[...])`
- Prompt structure: PromptName enum + SYSTEM string + TEMPLATE + METADATA + registry entry

## LangGraph Subgraph Patterns (Phase 3)

- Subgraph factory: `create_*_subgraph(config, driver, ...)` returns `StateGraph.compile()`
- Conditional edges: `builder.add_conditional_edges(node, condition_fn, [target_nodes])`
- Condition functions return `Literal["node_a", "node_b"]` matching target names
- State reducers: `Annotated[list[T], operator.add]` for append-only lists across invocations
- Early exit pattern: Check for empty context before LLM calls to avoid unnecessary API usage
- Tests: `tests/test_core/test_agentic/test_subgraphs.py` (19 tests)

## LangGraph Orchestrator & Checkpoints (Phase 4)

- AnyMessage in TypedDict: Keep outside TYPE_CHECKING (LangGraph needs runtime access), use `# noqa: TC002`
- Checkpointer: `AsyncPostgresSaver.from_conn_string(url)` as async context manager
- Setup tables: `await checkpointer.setup()` (LangGraph manages its own checkpoint schema)
- Thread config: `get_thread_config(thread_id)` -> `{"configurable": {"thread_id": "..."}}`
- Env var: `CHECKPOINT_DATABASE_URL` for PostgreSQL connection
- Tests: `tests/test_core/test_agentic/test_orchestrator.py` (17 tests)

## Agentic Streaming Patterns (Phase 5)

- Event types: `AgenticEventType` enum (ROUTING, SOURCES, TOKEN, PHASE, PROGRESS, ENTITIES, DONE, ERROR)
- SSE streaming: `stream_agentic_events(graph, input, config)` async generator using `graph.astream_events()`
- Routing event: Emitted by API handler (not streaming.py) to avoid duplication
- Frontend compatibility: Uses shape-based event detection (payload structure, not event type field)
- Checkpointer cleanup: `async_checkpointer_context()` for proper async cleanup in endpoints
- Tests: `tests/test_core/test_agentic/test_integration.py` (10 tests), `tests/test_routes/test_chat.py` (21 tests)

## Evaluation Framework (Phase 6)

- Agentic evaluators: `tool_selection`, `iteration_efficiency`, `critic_calibration`, `multi_hop_reasoning`
- LLM-as-judge: Use JSON output with `score` (0.0-1.0) and `reasoning` fields
- Sync wrappers: Required for LangSmith `evaluate()` - create `*_evaluator_sync` functions
- Cost tracking: `CostTracker.record_llm_call(model, input_tokens, output_tokens, operation)`
- Model pricing: Sort keys by length descending to match `gpt-4o-mini` before `gpt-4o`
- Performance hints: Auto-generated for high variance, repeated operations, slow subgraphs
- Dataset: `graphrag-agentic-eval` (16 examples across 5 categories)
- Scripts: `scripts/create_agentic_dataset.py`, `scripts/run_agentic_evaluation.py`
- Tests: `tests/test_evaluation/` (75+ tests across 3 files)

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

---

## Agentic RAG Implementation (Complete)

**Status**: ✅ All 6 phases completed (2026-02-04)

**Implementation Plan**: `docs/AGENTIC_IMPLEMENTATION_PLAN.md`

| Phase | Description | Key Files |
|-------|-------------|-----------|
| 0 | Setup | `core/agentic/__init__.py`, dependencies |
| 1 | State & Tools | `state.py`, `tools.py` (7 agent tools) |
| 2 | Prompts | 4 new prompts in `prompts/definitions.py` |
| 3 | Subgraphs | `subgraphs/rag.py`, `research.py`, `synthesis.py` |
| 4 | Orchestrator | `orchestrator.py`, `checkpoints.py` |
| 5 | Streaming | `streaming.py`, updated `routes/chat.py` |
| 6 | Evaluation | `evaluation/agentic_evaluators.py`, `performance.py`, `cost_analysis.py` |

### Agentic Commands

```bash
# Run agentic evaluation
cd backend && uv run python scripts/run_agentic_evaluation.py

# Compare experiments
uv run python scripts/compare_experiments.py -d graphrag-agentic-eval

# Create/update dataset
uv run python scripts/create_agentic_dataset.py
```
