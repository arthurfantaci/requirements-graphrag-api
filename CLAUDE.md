# Requirements GraphRAG API — Claude Code Instructions

## Project Layout
- `backend/` — FastAPI app, tests, pyproject.toml (all commands run from here)
- Python 3.13, managed with `uv` (use `uv run` prefix for all tools)
- `uv lock` must run before push (`--frozen` in CI)

## Conventions
- Conventional commits: `fix:`, `feat:`, `refactor:`, `docs:`
- Git workflow: Issue → Branch → PR (always use RC branch strategy for phases)
- **CLAUDE.md exception**: Updates to `CLAUDE.md` skip the Issue step — commit directly to a branch, open a minimal PR, merge. No issue, no RC branch, no ceremony.
- CI triggers only on PRs targeting `main` — RC→main is where CI runs
- Ruff format hook auto-fixes imports → always re-stage after first commit attempt

## Logging (structlog — fully migrated)
- ALL modules use `structlog.get_logger()` — do NOT use `logging.getLogger(__name__)`
- Named loggers: `structlog.get_logger("audit")`, `structlog.get_logger("guardrails")`
- `import logging` retained ONLY for level constants (`logging.WARNING`, `logging.INFO`)
- Printf-style log calls (`logger.info("msg %s", arg)`) — do NOT convert to keyword-style
- Structured data as keyword args (`logger.warning("msg", key=value)`) — NOT `extra={}`
- Test log capture: `structlog.testing.capture_logs()` (not `caplog`) for all modules

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
