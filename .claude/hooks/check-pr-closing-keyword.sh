#!/bin/bash
# PreToolUse hook: Ensure `gh pr create` commands include a closing keyword
# (Closes #N, Fixes #N, or Resolves #N) in the PR body.
#
# Exit 0 = allow, Exit 2 = block with message

TOOL_INPUT="$1"

# Only check gh pr create commands
if ! echo "$TOOL_INPUT" | grep -q "gh pr create"; then
    exit 0
fi

# Check for closing keywords in the command body
if echo "$TOOL_INPUT" | grep -qiE '(closes?|fix(es|ed)?|resolves?)\s+#[0-9]+'; then
    exit 0
fi

# Allow explicit "no associated issue" bypass
if echo "$TOOL_INPUT" | grep -qi "no associated issue"; then
    exit 0
fi

# Missing closing keyword — block and explain
cat >&2 <<'MSG'
BLOCKED: PR body is missing a closing keyword (e.g., "Closes #42").

Without this, the associated GitHub Issue will NOT auto-close when the PR merges.

Add one of these to the --body:
  Closes #N     Fixes #N     Resolves #N

If this PR genuinely has no associated issue, add "No associated issue" to the body
and re-run.
MSG
exit 2
