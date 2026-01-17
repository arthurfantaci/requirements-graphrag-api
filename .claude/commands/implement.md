# Implement Command

Implement the feature: $ARGUMENTS

## Instructions

1. Follow Test-Driven Development (TDD):
   - Write failing tests first
   - Implement minimum code to pass
   - Refactor while keeping tests green

2. Follow project conventions from CLAUDE.md

3. Apply relevant skills:
   - neo4j-patterns for database code
   - fastmcp-patterns for MCP tools
   - graphrag-patterns for RAG workflows

4. After each file, verify:
   - `uv run ruff check [file]` passes
   - `uv run pytest [test_file]` passes

## Workflow

```
1. Write test → 2. Run test (should fail) → 3. Implement → 4. Run test (should pass) → 5. Refactor
```

## Verification Checklist

Before marking complete:
- [ ] All new code has type hints
- [ ] All public functions have docstrings
- [ ] Tests cover happy path and edge cases
- [ ] `uv run ruff check src/` passes
- [ ] `uv run pytest` passes
- [ ] No credentials or secrets in code

## Output

After implementation, provide:
1. Summary of changes made
2. Test results
3. Any follow-up items needed
