---
name: session-handoff
description: Prepare a clean handoff when deferring work from the current Claude Code session to a fresh one — e.g., at a phase or task boundary, when controller context is bloated, or when the user signals "let's continue this in a new session." Produces the four required artifacts (handoff commit, MEMORY.md row, copy-paste-ready prompt, tracked GitHub Issue with the prompt as a comment). Use proactively whenever the user says "let's pick this up in a new session", "fresh context", "hand this off", "I'll resume tomorrow", or whenever you recognize a session-deferral moment.
allowed-tools: Read, Edit, Write, Bash, Glob, Grep
---

# Session Handoff

Prepare the artifacts a fresh Claude Code session needs to resume in-flight work with zero conversation context.

## Overview

Session Handoff is the deferring-session counterpart to the **Context Recovery Protocol** in `~/.claude/CLAUDE.md`. Context Recovery is what the receiving session runs at startup; Session Handoff is what the *current* session prepares so Context Recovery is efficient.

The protocol mandates four artifacts. Skipping any one of them leaves the next session guessing — which defeats the purpose of deferring in the first place.

## When to invoke

Trigger this skill when any of the following hold:

- The user signals they are stopping work but expect to resume later ("pick this up tomorrow", "fresh session", "let's hand this off").
- The controller context is approaching auto-compaction and the work will not finish in this session.
- A phase or task boundary is reached and the next phase should start with clean context (this is the controller-context-discipline case in `~/.claude/CLAUDE.md`).
- The current session has produced a vetted plan or design artifact and the implementation should run in a separate session.
- You are about to end your turn after pausing in-flight work for any reason.

If you are uncertain whether a handoff is warranted, ask the user. Never end a deferred-work session without the prompt — that is a hard rule from `~/.claude/CLAUDE.md`.

## Required outputs

Produce all four artifacts, in order, before ending the session.

### 1. Handoff commit on the working branch

Capture in-flight state so nothing is lost between sessions. Typical commit types:

- `docs(memory):` — updates to `MEMORY.md` and any auto-memory files.
- `docs(plan):` — updates to plans under `docs/superpowers/plans/` or `docs/internal/`.
- `chore:` — only when the in-flight state genuinely is neither memory nor plan.

The commit lives on the active working branch. Do **not** push to `origin` unless the user has confirmed they want it pushed — the handoff commit is sometimes intentionally local until the receiving session verifies state.

### 2. MEMORY.md row in "Recent decisions"

Add a single row pointing the fresh session at the next concrete artifact:

- Plan path (e.g., `docs/superpowers/plans/2026-05-11-feature-x.md`).
- Active branch and any open PR/issue number.
- The named next-step skill or action (e.g., `/executing-plans`, `/implement`, "resume Phase 2.3").

Trim older rows if the section is over its ~100-line cap.

### 3. Copy-paste-ready prompt as the final user-facing message

The prompt is delivered to the user as the last message before the session ends. The user pastes it verbatim into the new session. It MUST be self-contained — the receiving session has zero conversation context.

Required content (per `~/.claude/CLAUDE.md`):

- **Project pointer** — absolute path or repo name.
- **Current branch + open PR/issue** — number, URL, and state.
- **Concrete state** — test counts, CI status, what is pushed vs local, line counts vs caps, lock-file state.
- **Named next-step skill or action** — be specific. `/executing-plans` and the plan path beat "continue the work".
- **Session-specific protocols** — project conventions, files NOT to pre-emptively edit, anti-patterns to avoid, any HITL gates the receiving session must respect.

## Prompt template

Use this scaffold as the final assistant message. Replace the placeholders, then deliver it verbatim.

```text
Resuming work on <project-name> (<absolute-path>).

Active branch: <branch-name>
Open PR/issue: #<number> — <title> — <state>
Pushed to origin: <yes/no, last pushed commit SHA>

State as of handoff:
- Tests: <N passing / M total>, `<test-command>` green/red
- CI: <green/red/pending> on commit <SHA>
- Lint/typecheck: <green/red>
- MEMORY.md: <line-count>/<cap> lines
- CLAUDE.md: <line-count>/<cap> lines

Next step: <named skill or action, e.g. `/executing-plans` with `docs/.../plan.md`>.

Session protocols to honor:
- <project convention 1, e.g. "Issue → Branch → PR for all phases after Phase 0">
- <files NOT to pre-emptively edit, e.g. "Do not edit `uv.lock` directly">
- <anti-pattern to avoid, e.g. "Do not bundle docs-only changes into a separate PR">
- <HITL gates, e.g. "create_comment requires AskUserQuestion upstream">

Start by running the Context Recovery Protocol from `~/.claude/CLAUDE.md`:
1. Re-read CLAUDE.md and MEMORY.md.
2. `search_memories` for "<project-name>" and "<current-task>".
3. Read the plan at <plan-path> if referenced above.
```

Keep the scaffold tight; if a field is genuinely not applicable, omit it rather than padding with "N/A".

### 4. Tracked GitHub Issue with the prompt as a comment

A chat message evaporates; a tracked Issue survives. File the handoff automatically — do **not** wait to be asked.

1. Derive the repo from the working directory (`gh repo view --json nameWithOwner -q .nameWithOwner`).
2. Create the Issue: title `Session handoff: <next concrete action>`; body = 3-5 lines of context (what just landed, the open PR/issue, the priority deferred action) plus the line "The self-contained copy-paste resume prompt is in the first comment."
3. Post the **full copy-paste prompt from artifact #3 verbatim** as the first comment, inside a ```text fenced block so it copies cleanly.
4. Report the Issue number + URL and the comment URL.

Write the comment body to a temp file and post with `gh issue comment <N> --body-file <path>` — passing a fenced multi-line prompt via `--body` inline mangles the formatting.

Fallback: if `gh` is unavailable or unauthenticated, say so and leave the prompt in the final chat message (artifact #3) as the only delivery — never silently skip.

Note: this Issue tracks the *handoff itself*. It is distinct from any code-residual Issues that `memory-hygiene` step 7 files; both can exist for one deferral without overlap.

## Common pitfalls

- **Don't push to origin without confirming.** The handoff commit is local by default. Ask before `git push`.
- **Don't paraphrase concrete state.** "Tests mostly passing" is useless; "47/48 passing, one xfail in `test_retry_backoff`" is actionable.
- **Don't reference the current conversation.** Phrases like "as we discussed" or "the bug we found earlier" presume context the receiving session does not have. Inline the facts.
- **Don't omit the prompt because the user "knows the project."** The user is not the receiver — a fresh Claude Code session is. The prompt is written for that session, not for the human.
- **Don't write the prompt to a file unless asked.** It belongs in the final assistant message so the user can copy-paste directly. A file adds a step.
- **Don't conflate this with `/phase-handoff`.** Phase Handoff merges the PR and resets memory across a *completed* phase. Session Handoff defers in-flight work across a session boundary; the work is not finished.
- **Don't skip the MEMORY.md row even if the prompt is rich.** MEMORY.md is the durable index; the prompt is single-use. Future audits depend on the MEMORY.md trail.
- **Don't leave the handoff only in chat.** Artifact #4 (the tracked Issue + comment) is automatic and required — file it without being asked. A handoff that lives only in the conversation is gone the moment the session ends.

## Example

A worked example of a good handoff at a mid-phase boundary on the `jama-mcp-server` repo:

1. **Handoff commit** on `feature/phase-3-search-tools`:
   `docs(memory): pause phase-3 search-tool work at integration-test scaffolding`

2. **MEMORY.md "Recent decisions" row added** (in the table format the project uses):
   `| 2026-05-11 | Paused phase-3 mid-task; resume via /executing-plans with docs/superpowers/plans/2026-05-09-search-tools.md on branch feature/phase-3-search-tools (PR #42, draft). | Controller context approaching auto-compaction; clean restart preserves the in-flight integration-test scaffolding work. |`

3. **Final assistant message** (the copy-paste prompt):

   > Resuming work on jama-mcp-server (/Users/arthurfantaci/jama-mcp-server).
   >
   > Active branch: feature/phase-3-search-tools
   > Open PR/issue: #42 — "feat: search tools for phase 3" — draft
   > Pushed to origin: yes, commit a1b2c3d
   >
   > State as of handoff:
   > - Tests: 84/85 passing, `uv run pytest -m "not integration"` green except `test_search_filter_pagination` xfail (intentional, awaiting fixture)
   > - CI: green on a1b2c3d
   > - Lint/typecheck: green
   > - MEMORY.md: 78/100 lines
   > - CLAUDE.md: 142/150 lines
   >
   > Next step: `/executing-plans` with `docs/superpowers/plans/2026-05-09-search-tools.md`, resuming at Step 4 (integration-test scaffolding).
   >
   > Session protocols to honor:
   > - Issue → Branch → PR for all phases after Phase 0; PR #42 already exists, do not open a new one.
   > - Do not edit `uv.lock` directly; use `uv lock` if dependencies change.
   > - Docs-only changes bundle into the phase PR; do not open a separate docs PR.
   > - Write tools require AskUserQuestion HITL upstream, never inside the MCP server.
   >
   > Start by running the Context Recovery Protocol from `~/.claude/CLAUDE.md`:
   > 1. Re-read CLAUDE.md and MEMORY.md.
   > 2. `search_memories` for "jama-mcp-server" and "phase-3 search tools".
   > 3. Read the plan at docs/superpowers/plans/2026-05-09-search-tools.md.

## Related

- `~/.claude/CLAUDE.md` — Session Handoff Protocol section (source of truth).
- `~/.claude/CLAUDE.md` — Context Recovery Protocol (what the receiving session runs).
- `.claude/commands/phase-handoff.md` — sibling for *completed-phase* transitions (different use case).
- `.claude/commands/pre-compact.md` — sibling for memory-persistence before auto-compaction.
- `.claude/skills/memory-hygiene/SKILL.md` — full hygiene checklist.
