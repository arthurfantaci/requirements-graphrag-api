# Requirements GraphRAG API — Claude Code Instructions

## Project Layout
- `backend/` — FastAPI app, tests, pyproject.toml (all commands run from here)
- Python 3.13, managed with `uv` (use `uv run` prefix for all tools)
- `uv lock` must run before push (`--frozen` in CI)

## Conventions
- Conventional commits: `fix:`, `feat:`, `refactor:`, `docs:`
- Git workflow: Issue → Branch → PR (always use RC branch strategy for phases)
- CI triggers only on PRs targeting `main` — RC→main is where CI runs
- Ruff format hook auto-fixes imports → always re-stage after first commit attempt

## Logging (Phase 3a+)
- 14 core modules use `structlog.get_logger()` — do NOT use `logging.getLogger(__name__)`
- Unmigrated modules (auth/*, guardrails/*, evaluation/*, observability, core/agentic/*) still use stdlib logging
- Printf-style log calls (`logger.info("msg %s", arg)`) — do NOT convert to keyword-style
- Test log capture: `structlog.testing.capture_logs()` (not `caplog`) for structlog modules

## Testing
- 813+ tests, run with `uv run pytest --tb=short` from `backend/`
- Autouse fixtures (conftest.py): `_disable_langsmith_tracing`, `_reset_cost_tracker`, `_clear_singleton_caches`, `_reset_structlog`
- Async fixture gotcha: prefer sync fixtures with direct dict/object population over `await` calls
- Lazy import mocking: patch at source module (`neo4j_graphrag.retrievers.VectorRetriever`) not import site

## Verification Before PR
```bash
cd backend
uv run ruff check .          # lint (use --fix for import sorting)
uv run ruff format --check . # format
uv run pytest --tb=short     # tests
uv run ty check              # advisory only (continue-on-error)
```
