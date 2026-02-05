---
name: memory-hygiene
description: Audit and update Claude memory files after PRs or architecture changes. Use after merging PRs, deleting files, or changing architecture.
allowed-tools: Read, Edit, Write, Glob, Grep
---

# Memory Hygiene

Audit and update Claude's private memory files to prevent stale context.

## Memory File Locations

```
~/.claude/projects/-Users-arthurfantaci-Projects-requirements-graphrag-api/
├── CLAUDE.md        # Instructions (~150 lines max)
└── memory/
    └── MEMORY.md    # Learnings (~100 lines max)
```

## Audit Checklist

### 1. Check for Deleted File References

```bash
# Extract file paths mentioned in memory files
grep -oE '[a-zA-Z_/]+\.(py|ts|md)' ~/.claude/projects/-Users-arthurfantaci-Projects-requirements-graphrag-api/CLAUDE.md
```

For each path found, verify the file still exists. Remove references to deleted files.

### 2. Verify Architecture Section

Read `CLAUDE.md` Architecture section. Compare against actual directory structure:

```bash
ls -la backend/src/requirements_graphrag_api/core/
ls -la backend/src/requirements_graphrag_api/core/agentic/
```

Update if structure has changed.

### 3. Update Recent PRs Table

In `MEMORY.md`, keep only the last 5-10 PRs. Remove older entries. Format:

```markdown
| PR | Summary |
|----|---------|
| #123 | Brief description |
```

### 4. Check Line Counts

- CLAUDE.md should be ≤150 lines
- MEMORY.md should be ≤100 lines

```bash
wc -l ~/.claude/projects/-Users-arthurfantaci-Projects-requirements-graphrap-api/CLAUDE.md
wc -l ~/.claude/projects/-Users-arthurfantaci-Projects-requirements-graphrap-api/memory/MEMORY.md
```

If over limit, consolidate or archive old content.

### 5. Verify Patterns Are Still Valid

Review "Data Flow Gotchas" and "Patterns" sections. If a pattern was fixed or changed, update or remove it.

## When to Run This Skill

- After merging a PR that changes architecture
- After deleting files or modules
- After completing a major initiative
- When context feels stale or Claude makes outdated suggestions
- Monthly hygiene check

## Output

After running, report:
1. Files checked
2. Stale references removed
3. Sections updated
4. Current line counts
