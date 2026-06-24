---
name: memory-hygiene
description: >-
  Use when a project's Claude memory has gone stale and needs auditing — dead
  file references, drifted architecture notes, an overgrown working-state log,
  blown line budgets, invalid patterns/gotchas, or "pending / unexercised / TODO"
  residuals that should become tracked issues. Operates on the CURRENT project's
  memory surfaces (the per-project notes under ~/.claude/projects/<slug>/, plus
  any in-repo CLAUDE.md and MEMORY.md), resolved from the working directory — not
  hardcoded to any one repo. Triggers after merging a PR, deleting files or
  modules, finishing a phase or major initiative, before context compaction, or
  whenever memory feels stale or Claude gives outdated suggestions ("clean up my
  CLAUDE.md", "my project memory is stale", "audit my memory", "monthly memory
  check").
allowed-tools: Read, Edit, Write, Glob, Grep, Bash(gh issue *), Bash(grep *), Bash(wc *), Bash(ls *), Bash(find *), Bash(sed *), Bash(printf *)
---

# Memory Hygiene

Audit and refresh the current project's Claude memory so it never drifts from
reality. Memory that lies is worse than no memory — every step below reconciles a
stored claim against the live repo, the tracker, or the live tool surface. This
is a global tool: project-specific conventions belong in that project's
`CLAUDE.md`, never hardcoded here.

## Step 1 — Resolve the target (run first)

Derive the memory surfaces from the working directory; do not hardcode them.

```bash
SLUG="$(printf '%s' "$(pwd)" | sed 's#/#-#g')"
AUTO_DIR="$HOME/.claude/projects/$SLUG"
AUTO_MEMORY="$AUTO_DIR/memory/MEMORY.md"
AUTO_CLAUDE="$AUTO_DIR/CLAUDE.md"
ls -la "$AUTO_DIR" 2>/dev/null || echo "no auto-memory dir for this project at $AUTO_DIR"
ls -la ./CLAUDE.md ./MEMORY.md 2>/dev/null || echo "no in-repo CLAUDE.md / MEMORY.md"
```

Audit whichever surfaces exist: the auto-memory `MEMORY.md` (and its sibling
note files), the per-project `AUTO_CLAUDE`, and any repo-root `./CLAUDE.md` /
`./MEMORY.md` working-state index. Echo the resolved paths so the user can
confirm the right project was targeted.

## Audit checklist

### 2. Remove dead file references

Extract file paths mentioned in the memory files and confirm each still exists.
Use the `Grep` tool (broad extensions), then resolve each hit with `Glob`/`Read`:

```bash
grep -noE '[A-Za-z0-9_./-]+\.(py|ts|tsx|js|jsx|go|rs|java|rb|md|yaml|yml|toml|json|sh|sql)' "$AUTO_MEMORY" ./CLAUDE.md ./MEMORY.md 2>/dev/null | sort -u
```

For each path, try to resolve it relative to the project root. Distinguish
genuine repo files from intentional external pointers (other repos, URLs). Remove
or correct references to files that no longer exist.

### 3. Reconcile the architecture / structure notes

Read any "Architecture" / "Structure" section in the memory files, then compare
it to the project's ACTUAL layout — `Glob` the real top-level source directories
(do not assume a fixed package name or `backend/src/...` tree). Update notes that
no longer match.

### 4. Trim the working-state log

Working-state entries are time-ordered (a `| PR | Summary |` table, a "Recent
decisions" list, or a changelog — schemas vary by project). Keep only the most
recent ~5–10; remove older entries whose detail is now captured in code or git
history.

### 5. Enforce line budgets (soft caps)

Default guidance: `CLAUDE.md` ~150 lines, `MEMORY.md` ~100 lines. Treat as a
prompt to consolidate, not a hard failure — adjust per project.

```bash
wc -l "$AUTO_CLAUDE" "$AUTO_MEMORY" ./CLAUDE.md ./MEMORY.md 2>/dev/null
```

If over budget, consolidate or archive the oldest content.

### 6. Re-validate patterns and gotchas

Review "Patterns" / "Gotchas" / "Data Flow" notes. If a pattern was since fixed,
superseded, or disproven, update or remove it — a stale gotcha sends future
sessions down dead ends.

### 7. Promote dangling residuals → tracked Issues

A known gap left only as prose rots silently — no owner, no acceptance criteria,
gone the moment the row is trimmed. Scan working-state rows for gap-language:

```bash
grep -niE 'unexercised|untested|pending|TODO|present-but-|outstanding|not yet|still need' "$AUTO_MEMORY" ./MEMORY.md 2>/dev/null
```

For each hit that represents **real deferred implementation / code work** (NOT a
memory-only or CLAUDE.md-only edit — those never get their own issue/PR):
1. Open a GitHub Issue capturing full context + acceptance criteria
   (`gh issue create`). If `gh` is unavailable or unauthenticated, instead list
   the would-be issues in the run summary for the user to file.
2. Replace the memory prose with a one-line pointer: `… → tracked as Issue #N`.

**Always report every Issue created this way** (number + URL) in the summary.

### 8. Close stale Issues

```bash
gh issue list --state open --json number,title,body --limit 20 2>/dev/null
```

If an open Issue's work is clearly done (e.g. the merge missed a `Closes #N`),
close it with an explaining comment. **Always ask the user before closing** any
Issue whose acceptance criteria are ambiguous or only partially met.

### 9. Verify the phase pointer

For multi-phase projects, confirm `MEMORY.md` accurately reflects: the active
phase, the current working branch, and the next planned task/PR. If the phase
just transitioned, run the Phase Handoff Protocol from the user's global
`~/.claude/CLAUDE.md`.

### 10. Check protocol / tool-name drift

Memory often documents tools, MCP servers, or protocols by name. Compare that
prose against the LIVE tool surface — if the memory names a tool, server, file,
or flag, verify it still exists before leaving the claim in place. Flag or
correct references to anything renamed, retired, or moved. (Generalize: compare
stored protocol prose against the current environment; do not assume any specific
tool names.)

## When to Run This Skill

- After merging a PR that changes architecture, deletes files, or closes a phase
- After completing a major initiative or milestone
- Before context compaction (persist + prune first)
- When context feels stale or Claude makes outdated suggestions
- Monthly hygiene check

## Output

After running, report — in this order:
1. **Resolved target** memory dir + files audited (so the user confirms the right project)
2. Files checked
3. Stale references removed / corrected
4. Sections updated (architecture, working-state, patterns)
5. Current line counts vs budgets
6. **GitHub Issues opened** (from residuals) and **closed** (stale) — number + URL for each
7. Anything flagged for the user's decision
