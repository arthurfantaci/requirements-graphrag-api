#!/bin/bash
# PreToolUse hook: Validate bash commands for safety

TOOL_INPUT="$1"

# Block dangerous patterns
DANGEROUS_PATTERNS=(
    "rm -rf /"
    "rm -rf /*"
    "sudo rm"
    "chmod 777"
    ":(){:|:&};:"
    "mkfs"
    "dd if="
    "> /dev/sd"
)

for pattern in "${DANGEROUS_PATTERNS[@]}"; do
    if echo "$TOOL_INPUT" | grep -qF "$pattern"; then
        echo "BLOCKED: Dangerous command pattern detected: $pattern" >&2
        exit 2  # Exit code 2 blocks the tool
    fi
done

exit 0  # Allow the command
