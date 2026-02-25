---
name: fastapi-conventions
description: FastAPI best practices and conventions. Use when writing or refactoring FastAPI endpoints, routers, dependencies, or Pydantic models.
allowed-tools: Read, Grep, Glob, Edit, Write
---

# FastAPI Conventions

Official FastAPI best practices sourced from the bundled Agent Skill at:
`backend/.venv/lib/python3.13/site-packages/fastapi/.agents/skills/fastapi/SKILL.md`

When this skill is invoked, read that file for the full canonical reference.
Below is a project-specific summary tailored to this codebase.

## Annotated Style (Required)

Always use `Annotated` for parameter and dependency declarations:

```python
from typing import Annotated
from fastapi import Depends, Query

# Parameters
async def read_item(q: Annotated[str | None, Query(max_length=50)] = None): ...

# Dependencies — create reusable type aliases
CurrentUserDep = Annotated[dict, Depends(get_current_user)]
```

Never use the default-value style (`q: str = Query(...)`) — it breaks function signatures in other contexts.

## Return Types and Response Models

Every endpoint MUST have a typed return. This project already enforces this (all 15 endpoints have `response_model`).

- Use return type annotation when the return type matches the response shape
- Use `response_model=` on the decorator when filtering fields (e.g., hiding internal data)
- This enables Pydantic Rust serialization (2x+ JSON performance since FastAPI 0.130.0)

```python
# Preferred — return type annotation
@app.get("/items/")
async def get_item() -> Item: ...

# When filtering — response_model on decorator
@app.get("/items/", response_model=PublicItem)
async def get_item() -> Any: ...
```

## Router Configuration

Set prefix, tags, and dependencies on `APIRouter()`, not in `include_router()`:

```python
router = APIRouter(prefix="/items", tags=["items"], dependencies=[Depends(auth)])
app.include_router(router)  # Clean — no config duplication
```

## Async vs Sync

- Use `async def` only when all internal calls are `await`-compatible
- Use plain `def` when calling blocking code or when in doubt (runs in threadpool)
- Never run blocking code inside `async def` — damages event loop performance

This project's endpoints are `async def` because they call async Neo4j driver and LangChain operations.

## Deprecated Patterns to Avoid

- `ORJSONResponse` / `UJSONResponse` — deprecated since 0.131.0, use standard `JSONResponse`
- Ellipsis (`...`) as default for required fields — just omit the default
- `RootModel` — use `Annotated` with `Field` instead
- `api_route()` with multiple methods — one HTTP operation per function
- Class dependencies with `Depends()` — prefer function dependencies returning instances

## Strict Content-Type (Since 0.132.0)

FastAPI now validates `Content-Type: application/json` on JSON requests by default.
Disable per-app with `FastAPI(strict_content_type=False)` if needed.

## Tooling

This project already follows the recommended stack: `uv`, `ruff`, `ty`.
