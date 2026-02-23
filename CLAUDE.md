# Requirements GraphRAG API — Claude Code Instructions

## Project Layout
- `backend/` — FastAPI app, tests, pyproject.toml (all commands run from here)
- Python 3.13, managed with `uv` (use `uv run` prefix for all tools)
- `uv lock` must run before push (`--frozen` in CI)

## Implementation Plan
- **Authoritative plan**: `~/Projects/graphrag-api-db/docs/best-practices-plan.md` (vetted v2)
- **Tracking issue**: #212 (best practices phases 2-5b) — **ALL PHASES COMPLETE**
- **Phase handoffs**: See `memory/phase-handoffs.md` for historical context

## Conventions
- Conventional commits: `fix:`, `feat:`, `refactor:`, `docs:`
- Git workflow: Issue → Branch → PR (always use RC branch strategy for phases)
- **CLAUDE.md & memory files — NO separate workflow**: Changes that ONLY touch `CLAUDE.md` or `~/.claude/` memory files MUST NOT get their own issue, branch, or PR. Bundle them into the next phase's PR, or commit directly to the working branch. See global CLAUDE.md for full rule.
- CI triggers only on PRs targeting `main` — RC→main is where CI runs
- Ruff format hook auto-fixes imports → always re-stage after first commit attempt

## Logging (structlog — fully migrated)
- ALL modules use `structlog.get_logger()` — do NOT use `logging.getLogger(__name__)`
- Named loggers: `structlog.get_logger("audit")`, `structlog.get_logger("guardrails")`
- `import logging` retained ONLY for level constants (`logging.WARNING`, `logging.INFO`)
- Printf-style log calls (`logger.info("msg %s", arg)`) — do NOT convert to keyword-style
- Structured data as keyword args (`logger.warning("msg", key=value)`) — NOT `extra={}`
- Test log capture: `structlog.testing.capture_logs()` (not `caplog`) for all modules

## HybridRetriever (Phase 5a)
- `create_hybrid_retriever()` in `core/retrieval.py` — mirrors `create_vector_retriever()` pattern
- Custom `_hybrid_result_formatter()` required — default formatter loses element IDs for article joins
- Legacy fallback: `_legacy_hybrid_search()` behind `USE_LEGACY_HYBRID_SEARCH` config flag
- `hybrid_retriever` threaded through entire call chain (api.py → routes → handlers → orchestrator → rag → search)
- Test fixtures for search routes must set `app.state.hybrid_retriever` (even if `None`)
- Neo4j indexes: `chunk_embeddings` (vector, 1024d COSINE), `chunk_text_fulltext` (fulltext on `Chunk.text`)
- Do NOT use `chunk_fulltext` index — it indexes non-existent `heading` property (broken)

## Response Models (Phase 4)
- All 15 endpoints have typed Pydantic `response_model` — do NOT add untyped endpoints
- New response models use `ConfigDict(extra="allow")` — tighten later
- Search models: `EntityInfo.name` required, `SemanticRelationship.from_entity`/`.relationship` required
- Typed sub-models: `ChunkMetadata`, `RelatedArticle`, `IndustryStandardInfo` (not `dict[str, Any]`)
- Degenerate chunk filtering: `_is_meaningful_content()` in `core/retrieval.py` (≥20 non-whitespace chars)

## Testing
- 837+ tests, run with `uv run pytest --tb=short` from `backend/`
- Autouse fixtures (conftest.py): `_disable_langsmith_tracing`, `_reset_cost_tracker`, `_clear_singleton_caches`, `_reset_structlog`
- Async fixture gotcha: prefer sync fixtures with direct dict/object population over `await` calls
- Lazy import mocking: patch at source module (`neo4j_graphrag.retrievers.VectorRetriever`) not import site

## Community Retrieval (Phase 5b)
- `community_search()` in `core/retrieval.py` — VectorRetriever on `community_summary_embeddings` index
- Always-parallel: runs via `asyncio.gather` alongside RAG subgraph in orchestrator's `run_rag` node
- Graceful degradation: `app.state.community_index_available` set at startup via `SHOW VECTOR INDEXES`
- `community_index_available` threaded through call chain (api.py → chat.py → handlers.py → orchestrator.py)
- Context formatting: `## Community Context` section appended after `## Knowledge Graph Context` in `format_context()`
- Config: `COMMUNITY_INDEX_NAME` env var (default: `community_summary_embeddings`)
- Test fixtures for routes must set `app.state.community_index_available` (even if `False`)

## Verification Before PR
```bash
cd backend
uv run ruff check .          # lint (use --fix for import sorting)
uv run ruff format --check . # format
uv run pytest --tb=short     # tests
uv run ty check              # advisory only (continue-on-error)
```
