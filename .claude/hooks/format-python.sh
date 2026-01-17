#!/bin/bash
# PostToolUse hook: Auto-format Python files after edits

TOOL_INPUT="$1"

# Extract file path from tool input (assumes JSON with "path" field)
FILE_PATH=$(echo "$TOOL_INPUT" | grep -oP '"path"\s*:\s*"\K[^"]+' 2>/dev/null)

# Only process Python files
if [[ "$FILE_PATH" == *.py ]]; then
    if command -v uv &> /dev/null; then
        uv run ruff format "$FILE_PATH" 2>/dev/null
        uv run ruff check --fix "$FILE_PATH" 2>/dev/null
    elif command -v ruff &> /dev/null; then
        ruff format "$FILE_PATH" 2>/dev/null
        ruff check --fix "$FILE_PATH" 2>/dev/null
    fi
fi

exit 0  # Always succeed (formatting is best-effort)
