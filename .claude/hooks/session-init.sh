#!/bin/bash
# SessionStart hook: Load context at session start

echo "=== Session Context ==="

# Show git status if in a repo
if command -v git &> /dev/null && git rev-parse --git-dir &> /dev/null; then
    BRANCH=$(git branch --show-current 2>/dev/null)
    echo "Git branch: $BRANCH"
    
    CHANGES=$(git status --porcelain 2>/dev/null | wc -l)
    echo "Uncommitted changes: $CHANGES files"
fi

# Show any TODOs in recent commits
if command -v git &> /dev/null && git rev-parse --git-dir &> /dev/null; then
    TODOS=$(git diff HEAD~5 --unified=0 2>/dev/null | grep -c "TODO\|FIXME\|XXX" || echo "0")
    if [[ $TODOS -gt 0 ]]; then
        echo "Recent TODOs added: $TODOS"
    fi
fi

# Check for failing tests (quick check)
if [[ -f "pyproject.toml" ]] && command -v uv &> /dev/null; then
    echo "Running quick test check..."
    uv run pytest --collect-only -q 2>/dev/null | tail -1 || echo "  (tests not configured)"
fi

echo "======================="

exit 0
