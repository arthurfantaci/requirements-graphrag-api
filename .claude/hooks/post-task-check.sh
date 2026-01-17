#!/bin/bash
# Stop hook: Run quality checks after task completion

echo "Running post-task quality checks..."

# Check if we're in a Python project
if [[ -f "pyproject.toml" ]]; then
    # Run linting
    if command -v uv &> /dev/null; then
        echo "Lint check:"
        uv run ruff check src/ 2>/dev/null || echo "  (no src/ directory or ruff not configured)"
    fi
fi

# Check for uncommitted changes
if command -v git &> /dev/null && git rev-parse --git-dir &> /dev/null; then
    CHANGES=$(git status --porcelain 2>/dev/null | wc -l)
    if [[ $CHANGES -gt 0 ]]; then
        echo "Uncommitted changes: $CHANGES files"
    fi
fi

exit 0  # Always succeed (this is informational)
