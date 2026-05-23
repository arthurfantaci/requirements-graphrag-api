# Unified Source Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace three-way source-attribution numbering inconsistency with a single canonical scheme — inline `[Source N]` chips in the body, matched 1:1 by numbered `[N]` chips in the accordion below, with hover tooltips on accordion rows surfacing the retrieved chunk text.

**Architecture:** Three coordinated changes — one backend deletion (~6 lines), two frontend additions (~70 lines) — in a single bundled PR on branch `feat/unified-source-attribution` (already created, with the spec doc already committed). Backend removes the redundant `**Sources:**` markdown footer that synthesis post-processing currently appends. Frontend numbers accordion rows with the same emerald-pill styling already used for inline citation chips, and wraps row titles in the existing shared `<Tooltip>` widget so hovering surfaces a smart-truncated excerpt of the retrieved chunk text. No payload-schema changes; all needed fields already flow through the SSE stream.

**Tech Stack:** Backend — Python 3.13, FastAPI, LangGraph, structlog, pytest (run via `uv`); Frontend — React 19, Vite, Tailwind CSS v4, ReactMarkdown.

**Spec:** `docs/superpowers/specs/2026-05-23-unified-source-attribution-design.md` (read this first for the design rationale, non-goals, error-handling matrix, and HITL gate sequence).

**Tracking issue:** [#357](https://github.com/arthurfantaci/requirements-graphrag-api/issues/357)

**Branch:** `feat/unified-source-attribution` (verify with `git branch --show-current` before starting; the spec doc commits are already on this branch).

---

## Task 1: Backend — Remove the redundant Sources footer (TDD)

**Files:**

- Modify: `backend/src/requirements_graphrag_api/core/agentic/subgraphs/synthesis.py:281-287`
- Test: `backend/tests/test_core/test_agentic/test_subgraphs.py` (add one test method to `TestSynthesisSubgraph` class, around line 268)

**Why source-inspection over runtime assertion:** the spec's intended runtime assertion (`final_answer` does not contain `**Sources:**`) would require setting up LLM mocking infrastructure that does not currently exist for synthesis tests (only `test_synthesis_no_context` exercises the graph, and it uses the no-context early-exit to avoid the LLM). Source-inspection is a stronger invariant — if the source string is absent, the runtime output cannot produce it — and avoids introducing a new test-infrastructure dependency for a one-shot regression guard.

- [ ] **Step 1: Verify you are on the right branch with the spec already present**

Run:

```bash
cd /Users/arthurfantaci/Projects/requirements-graphrag-api
git branch --show-current
ls docs/superpowers/specs/2026-05-23-unified-source-attribution-design.md
```

Expected: branch is `feat/unified-source-attribution`; spec file exists. If not on the branch, stop and fix before continuing.

- [ ] **Step 2: Read the existing test patterns for context**

Open `backend/tests/test_core/test_agentic/test_subgraphs.py` and skim the `TestSynthesisSubgraph` class (lines ~188-267). Note that no existing test invokes `format_output` directly or asserts on the `**Sources:**` rendering — we are adding one new test, not modifying any existing test.

- [ ] **Step 3: Write the failing regression test**

Open `backend/tests/test_core/test_agentic/test_subgraphs.py`. Add the following test method as the LAST method inside the `TestSynthesisSubgraph` class (after `test_needs_revision_insufficient_completeness`, before the `# STATE TYPE TESTS` section comment around line 270):

```python
    def test_synthesis_module_has_no_sources_footer(self):
        """Regression guard: synthesis must not emit a '**Sources:**' citation footer.

        Issue #357 / docs/superpowers/specs/2026-05-23-unified-source-attribution-design.md.
        The accordion at the bottom of the response is the canonical numbered source list;
        a duplicated markdown footer creates a divergent third numbering scheme (LLM-judged
        citations) that conflicts with the inline [Source N] markers (retrieval order).

        This is a source-inspection regression rather than a runtime assertion because
        format_output is a closure inside create_synthesis_subgraph and is not reachable
        through a stable public API without elaborate LLM mocking through the full graph.
        Source-inspection is a strictly stronger invariant: if the offending string is
        absent from the module source, the runtime output cannot produce it.
        """
        import inspect

        from requirements_graphrag_api.core.agentic.subgraphs import synthesis

        source = inspect.getsource(synthesis)
        assert "**Sources:**" not in source, (
            "synthesis.py must not contain a '**Sources:**' footer marker; "
            "see docs/superpowers/specs/2026-05-23-unified-source-attribution-design.md"
        )
        assert "citation_text" not in source, (
            "The citation_text variable was part of the now-removed '**Sources:**' "
            "footer logic; see the spec above for the rationale."
        )
```

- [ ] **Step 4: Run the new test to verify it FAILS (TDD red)**

Run:

```bash
cd backend
uv run pytest tests/test_core/test_agentic/test_subgraphs.py::TestSynthesisSubgraph::test_synthesis_module_has_no_sources_footer -v
```

Expected: 1 failed. The failure message should be the first assertion (`'**Sources:**' not in source`). This is correct — the footer code is still present at this stage.

If the test PASSES at this point, stop. Something is wrong (either the footer is already removed, or the test imports the wrong module). Investigate before continuing.

- [ ] **Step 5: Delete the footer code in synthesis.py**

Open `backend/src/requirements_graphrag_api/core/agentic/subgraphs/synthesis.py`. Find the `format_output` function (around line 269). The current code at lines 281-287:

```python
        logger.info("Formatting final output")

        # Build final answer with citation footer
        final = draft
        if citations:
            citation_text = "\n\n**Sources:**\n"
            for i, source in enumerate(citations, 1):
                citation_text += f"- [{i}] {source}\n"
            final += citation_text

        # Add confidence indicator if low
```

Delete line 281 (the section comment, now misleading) and lines 283-287 (the `if citations:` block and its body). Line 282 (`final = draft`) MUST be retained — it initializes the variable that the confidence-indicator block at lines 290-294 still appends to.

Result after edit (lines 279-289 in the new file):

```python
        logger.info("Formatting final output")

        final = draft

        # Add confidence indicator if low
        if critique and critique.confidence < CONFIDENCE_THRESHOLD:
            final += (
                "\n\n*Note: This answer is based on limited context. "
                "Consider asking follow-up questions for more detail.*"
            )
```

The `citations = state.get("citations", [])` extraction at line 276 stays unchanged — only the rendering append is removed, so `citations` remains in `SynthesisState` for downstream consumers (e.g., LangSmith telemetry).

- [ ] **Step 6: Run the new test to verify it PASSES (TDD green)**

Run:

```bash
cd backend
uv run pytest tests/test_core/test_agentic/test_subgraphs.py::TestSynthesisSubgraph::test_synthesis_module_has_no_sources_footer -v
```

Expected: 1 passed.

- [ ] **Step 7: Run the full backend suite to confirm no regressions**

Run:

```bash
cd backend
uv run pytest --tb=short
```

Expected: 853 passed (was 852; one net new test added). If any pre-existing tests fail, the deletion broke something — most likely a test in `test_orchestrator.py` or `test_integration.py` that asserts on a substring of `final_answer`. Investigate and either fix the test (if its assertion was checking the footer pattern) or revert and reassess.

- [ ] **Step 8: Run lint and format checks**

Run:

```bash
cd backend
uv run ruff check .
uv run ruff format --check .
```

Expected: both pass (no output, exit 0). If ruff format flags changes, run `uv run ruff format .` and re-stage.

- [ ] **Step 9: Commit**

Run:

```bash
cd /Users/arthurfantaci/Projects/requirements-graphrag-api
git add backend/src/requirements_graphrag_api/core/agentic/subgraphs/synthesis.py \
        backend/tests/test_core/test_agentic/test_subgraphs.py
git commit -m "$(cat <<'EOF'
refactor(synthesis): remove redundant Sources footer from final answer

The format_output node was appending a markdown '**Sources:**' footer
to every explanatory answer, built from the LLM-judged `citations` field.
This created a third numbering scheme (LLM-judged) that conflicted with
the inline [Source N] markers (retrieval-order) and the unnumbered
accordion (retrieval-order). The accordion is the canonical source list;
the footer was redundant and divergently numbered.

The `citations` field is retained in SynthesisState since downstream
consumers (LangSmith telemetry) may read it; only the rendering append
is removed.

Adds a source-inspection regression test to prevent the footer pattern
from being reintroduced.

Refs: #357
See docs/superpowers/specs/2026-05-23-unified-source-attribution-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: pre-commit hooks pass; commit lands on `feat/unified-source-attribution`. If ruff format hook re-formats files, follow the project convention — re-stage the formatted files and re-commit.

---

## Task 2: Frontend — Number accordion rows with citation-style chips

**Files:**

- Modify: `frontend/src/components/metadata/SourcesPanel.jsx` (modify `SourceItem` signature + render structure; modify `SourcesPanel` map call)

No tests — frontend has no test framework per project posture. Verification is local dev-server inspection at this stage; full Vercel Preview HITL comes in Task 4.

- [ ] **Step 1: Modify `SourceItem` to accept `sourceNumber` and render the chip**

Open `frontend/src/components/metadata/SourcesPanel.jsx`. Find the `SourceItem` function (lines 76-98). Replace the entire function with:

```jsx
function SourceItem({ source, sourceNumber }) {
  const { title, url, relevance_score } = source

  return (
    <div className="flex items-start justify-between gap-2 py-2 border-b border-black/5 last:border-b-0">
      <div className="flex-1 min-w-0 flex items-start gap-2">
        <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 flex-shrink-0 mt-0.5">
          {sourceNumber}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-charcoal-light truncate">{title || 'Untitled Source'}</p>
          <RelevanceBar score={relevance_score} />
        </div>
      </div>
      {url && (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex-shrink-0 p-1 text-charcoal-muted hover:text-emerald-600 transition-colors"
          title="Open source"
        >
          <LinkIcon />
        </a>
      )}
    </div>
  )
}
```

What changed:

- Function now accepts a `sourceNumber` prop.
- The `<div className="flex-1 min-w-0">` that previously held title + relevance bar is now nested inside a new `<div className="flex-1 min-w-0 flex items-start gap-2">` that hosts the chip + the title/relevance block.
- New `<span>` chip is added BEFORE the title/relevance block; its class string matches `SourceCitationRenderer.jsx:30` exactly except for: `mr-2` is omitted (parent gap handles spacing via `gap-2`); `flex-shrink-0` is added so the chip never compresses when the title is long; `mt-0.5` is added so the chip vertically aligns with the title text baseline.
- Inner title/relevance block is unchanged.
- External-link icon block is unchanged.

- [ ] **Step 2: Update the map call in `SourcesPanel` to pass `sourceNumber`**

In the same file, find lines 133-137 (inside the `SourcesPanel` function's expandable content block). Change:

```jsx
          {sources.map((source, index) => (
            <SourceItem key={`source-${index}`} source={source} />
          ))}
```

To:

```jsx
          {sources.map((source, index) => (
            <SourceItem key={`source-${index}`} source={source} sourceNumber={index + 1} />
          ))}
```

- [ ] **Step 3: Run the dev server and verify chips render**

Run:

```bash
cd frontend
npm run dev
```

Expected: server starts on the configured local port (typically 5173). Open the browser. Submit any explanatory query that returns multiple sources (e.g., "What is requirements traceability?"). Wait for the response.

Verify:

- The Sources accordion header still shows `Sources (N)` with the correct count.
- Clicking the header expands the accordion.
- Each row now has an emerald `[N]` chip prefix (1, 2, 3, …) styled identically to the inline `[Source N]` chips in the response body — same emerald background, same border, same text color.
- Hovering an inline `[Source N]` chip in the body and visually scanning to the matching accordion row N — they should clearly be the same source (same title).
- External-link icon on each row still works (opens source URL in new tab).
- Relevance bar still renders correctly below each title.

If any of these fail, stop and debug before committing. Common causes: missing `sourceNumber` prop (chip will show "undefined"), broken flex layout (chip and title misaligned), or the class string typo.

Stop the dev server with Ctrl-C when verification is complete.

- [ ] **Step 4: Commit**

Run:

```bash
cd /Users/arthurfantaci/Projects/requirements-graphrag-api
git add frontend/src/components/metadata/SourcesPanel.jsx
git commit -m "$(cat <<'EOF'
feat(ui): number Sources accordion rows with citation-style chips

Prefix each accordion row with an emerald [N] chip matching the
inline [Source N] citation chip styling (same Tailwind class string
as SourceCitationRenderer.jsx:30, minus the inline-context margins).

Row N now visibly corresponds 1:1 to inline citation N — the
alignment was already correct in the data layer (both derive from
the same retrieval-order sources[] array in core/context.py), but
was invisible to users until the rows were numbered.

Refs: #357
See docs/superpowers/specs/2026-05-23-unified-source-attribution-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit lands on `feat/unified-source-attribution`. No pre-commit hooks should fire for `.jsx` files (the project's hooks are Python-focused).

---

## Task 3: Frontend — Add chunk-excerpt tooltip on row title hover

**Files:**

- Modify: `frontend/src/components/metadata/SourcesPanel.jsx` (add module-top constants + `buildExcerpt` helper; wrap title `<p>` in `<Tooltip>`)

No tests — same posture as Task 2.

- [ ] **Step 1: Add the truncation helper at the top of `SourcesPanel.jsx`**

Open `frontend/src/components/metadata/SourcesPanel.jsx`. Find the import block at the top (lines 1-2):

```jsx
import { useState } from 'react'
import { Tooltip } from '../ui/Tooltip'
```

Immediately AFTER the imports (with one blank line between the last import and the new code), add:

```jsx

const EXCERPT_MAX_CHARS = 280
const BOUNDARY_FLOOR = 100

function buildExcerpt(content) {
  if (!content) return null
  const normalized = content.trim().replace(/\s+/g, ' ')
  if (!normalized) return null
  if (normalized.length <= EXCERPT_MAX_CHARS) return normalized

  const window = normalized.slice(0, EXCERPT_MAX_CHARS)

  // Prefer the last sentence boundary within the window
  const sentenceMatches = [...window.matchAll(/[.?!]\s/g)]
  const lastSentence = sentenceMatches[sentenceMatches.length - 1]
  if (lastSentence && lastSentence.index >= BOUNDARY_FLOOR) {
    return normalized.slice(0, lastSentence.index + 1) + ' …'
  }

  // Fall back to the last word boundary
  const wordBoundary = window.lastIndexOf(' ')
  if (wordBoundary >= BOUNDARY_FLOOR) {
    return normalized.slice(0, wordBoundary) + ' …'
  }

  // Pathological: no whitespace in first 280 chars — hard cut
  return window + '…'
}
```

The `…` characters MUST be the single Unicode codepoint U+2026 (`…`), NOT three ASCII periods.

- [ ] **Step 2: Wrap the title `<p>` in `<Tooltip>` and destructure `content` from source**

Find the `SourceItem` function (now starting around line 27 after Task 2's changes, but verify by searching for `function SourceItem`). Replace the whole function with:

```jsx
function SourceItem({ source, sourceNumber }) {
  const { title, url, content, relevance_score } = source
  const excerpt = buildExcerpt(content)

  return (
    <div className="flex items-start justify-between gap-2 py-2 border-b border-black/5 last:border-b-0">
      <div className="flex-1 min-w-0 flex items-start gap-2">
        <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 flex-shrink-0 mt-0.5">
          {sourceNumber}
        </span>
        <div className="flex-1 min-w-0">
          <Tooltip
            title={title || `Source ${sourceNumber}`}
            description={excerpt || 'No excerpt available.'}
            color="emerald"
            position="top"
          >
            <p className="text-sm text-charcoal-light truncate">
              {title || 'Untitled Source'}
            </p>
          </Tooltip>
          <RelevanceBar score={relevance_score} />
        </div>
      </div>
      {url && (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex-shrink-0 p-1 text-charcoal-muted hover:text-emerald-600 transition-colors"
          title="Open source"
        >
          <LinkIcon />
        </a>
      )}
    </div>
  )
}
```

What changed from Task 2's version:

- Destructure now includes `content` (previously omitted; backend already returns it).
- Compute `excerpt = buildExcerpt(content)` at the top of the function body.
- Wrap the title `<p>` in `<Tooltip>` with `title`, `description`, `color="emerald"`, and `position="top"` props. The `Tooltip` import is already present from Task 2 / the original file.
- Everything else is identical to Task 2's version.

- [ ] **Step 3: Run the dev server and verify tooltips render**

Run:

```bash
cd frontend
npm run dev
```

Submit an explanatory query that returns multiple sources. Expand the accordion. Verify:

- Hover the title text of row 1. A dark tooltip appears with the source title (bold) and a body paragraph showing the chunk text excerpt.
- The excerpt is readable (clean text, no awkward gaps from preserved newlines).
- If the row's `source.content` is more than ~280 characters, the excerpt ends with `…` and the cut is at a clean sentence or word boundary.
- If the row's `source.content` is shorter than ~280 characters, the full text shows with no ellipsis.
- Mouse off; the tooltip disappears.
- Hover another row — same behavior, different content.
- The chip prefix from Task 2 is unaffected; the external-link icon still works.

Edge cases to spot-check:

- Hover a row near the bottom of the viewport — the tooltip should flip to render below the row if there's not enough space above.
- Hover a row near the very top of the viewport — tooltip should render below the row (existing widget flip logic).
- On a mobile-width browser (resize to ~400px), the tooltip should remain within the viewport bounds, not overflow horizontally.

If you cannot find a row whose `content` is empty (which would trigger the "No excerpt available." fallback), this is fine to leave unverified at the dev-server stage — the fallback is exercised by the code path regardless and will be verified during Vercel Preview HITL.

Stop the dev server with Ctrl-C when verification is complete.

- [ ] **Step 4: Commit**

Run:

```bash
cd /Users/arthurfantaci/Projects/requirements-graphrag-api
git add frontend/src/components/metadata/SourcesPanel.jsx
git commit -m "$(cat <<'EOF'
feat(ui): show retrieved chunk excerpt on Sources row hover

Wrap each accordion row title in the shared <Tooltip> widget. The
tooltip surfaces the retrieved chunk text (source.content, already
returned by the backend on the SSE stream) as a smart-truncated
excerpt — preferring sentence boundaries up to ~280 chars, falling
back to word boundaries, then to a hard cut at the limit.

Whitespace runs (including embedded newlines from source HTML) are
collapsed so the rendered excerpt flows cleanly inside the tooltip's
single <p> body. Empty or missing content shows "No excerpt available."
for a consistent affordance across all rows.

No XSS surface introduced — React text rendering escapes by default,
and the excerpt is passed as a plain string prop to TooltipContent.

Refs: #357
See docs/superpowers/specs/2026-05-23-unified-source-attribution-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Push branch + open PR + Vercel Preview HITL gate

**Files:** none (orchestration only).

- [ ] **Step 1: Verify the branch state before push**

Run:

```bash
cd /Users/arthurfantaci/Projects/requirements-graphrag-api
git status
git log main..HEAD --oneline
```

Expected: working tree clean. The branch should have SEVEN commits ahead of `main`:

1. `docs(spec): unified source attribution design`
2. `docs(spec): fix markdownlint diagnostics on unified source attribution spec`
3. `docs(plan): unified source attribution implementation plan`
4. `docs(plan): fix expected commit count in Task 4 Step 1`
5. `refactor(synthesis): remove redundant Sources footer from final answer`
6. `feat(ui): number Sources accordion rows with citation-style chips`
7. `feat(ui): show retrieved chunk excerpt on Sources row hover`

If the commit list looks wrong, stop and reconcile before pushing.

- [ ] **Step 2: Push the branch with upstream tracking**

Run:

```bash
git push -u origin feat/unified-source-attribution
```

Expected: branch is published to origin and upstream tracking is set.

- [ ] **Step 3: Open the PR via `gh pr create`**

Run:

```bash
gh pr create --base main --head feat/unified-source-attribution --title "feat: unify source attribution — accordion numbering + hover excerpts" --body "$(cat <<'EOF'
## Summary

- Remove redundant `**Sources:**` footer from synthesis output (`refactor(synthesis)`).
- Number Sources accordion rows with citation-style emerald chips matching the inline `[Source N]` chip styling (`feat(ui)`).
- Show retrieved chunk text excerpt on Sources row title hover (`feat(ui)`).

Resolves the three-way numbering inconsistency described in #357. Inline `[Source N]` chips and accordion rows now visibly correspond 1:1 (alignment was already correct in the data layer; this PR makes it visible). The third footer-numbering scheme is removed.

## Spec

`docs/superpowers/specs/2026-05-23-unified-source-attribution-design.md`

## Test plan

- [ ] Backend: full pytest suite goes 852 → 853 passing (one net new regression test added — `test_synthesis_module_has_no_sources_footer`).
- [ ] Frontend: no test framework per project posture; HITL manual verification on the Vercel Preview deployment.

### Vercel Preview HITL checklist (gate before merge)

- [ ] LLM response body shows no `**Sources:**` block at the bottom.
- [ ] Accordion header still shows `Sources (N)`.
- [ ] Expanding the accordion reveals `[1]`, `[2]`, … emerald chips at row start, matching inline-chip styling.
- [ ] Hovering an accordion row title shows a dark tooltip with the chunk text excerpt.
- [ ] Long chunks truncate with `…`; short chunks display fully; empty `content` shows "No excerpt available."
- [ ] External-link icon on each row still opens the source URL in a new tab.
- [ ] All pre-existing accordion behavior preserved (expand/collapse, info icon tooltip, relevance bar).

### Post-merge production HITL (after merge to main)

- [ ] After Railway auto-deploy completes, run an explanatory query against production and confirm: (a) `**Sources:**` block gone, (b) accordion numbering visible, (c) hover tooltip working.

Closes #357

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL is returned. Capture it.

- [ ] **Step 4: Monitor CI and Vercel Preview status**

Run (in a loop or via Monitor):

```bash
gh pr checks $(gh pr view --json number --jq .number) --watch
```

Wait for:

- Lint & Format Check: pass
- Test: pass
- Vercel — Preview: ready (preview URL listed in PR comments)

If any check fails, investigate the failure, push a fix on the branch, and re-monitor. Do not proceed to Step 5 until all checks are green AND the Vercel Preview is Ready.

- [ ] **Step 5: Surface the Vercel Preview URL to the user for manual HITL**

Per `feedback_hitl-before-production.md`, STOP before proposing merge. Post a message to the user that contains:

1. The Vercel Preview URL (from the PR's `vercel[bot]` comment or `gh pr view --comments`).
2. The seven-check HITL checklist verbatim (see PR body above).
3. An explicit "please exercise the preview manually and approve merge when ready" prompt.

Do NOT merge. Wait for the user's explicit approval. The act-on-authorized-tasks memory does NOT authorize self-merge of product-surface PRs.

---

## Task 5: Squash-merge after user approval + Railway production HITL

**Files:** none (orchestration only).

Pre-condition: the user has explicitly approved the merge after exercising the Vercel Preview per Task 4 Step 5.

- [ ] **Step 1: Squash-merge with branch deletion**

Run:

```bash
gh pr merge $(gh pr view --json number --jq .number) --squash --delete-branch
```

Expected: PR is squash-merged into `main`, the remote feature branch is deleted, the local branch is detached.

- [ ] **Step 2: Sync local main**

Run:

```bash
cd /Users/arthurfantaci/Projects/requirements-graphrag-api
git checkout main
git pull --ff-only origin main
git log -1 --format='%h %s'
```

Expected: local `main` is at the squash-merge commit.

- [ ] **Step 3: Wait for Railway auto-deploy to complete**

Railway watches `main` and auto-deploys when backend files change. Use the Monitor tool with a Railway API or status check, or wait ~3-5 minutes for typical deploy duration. Then probe production:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://api-production-f1cf.up.railway.app/health
```

Expected: `200`.

If Railway shows a failed deploy, halt and investigate before proceeding to production HITL.

- [ ] **Step 4: Surface production HITL prompt to the user**

Per `feedback_hitl-before-production.md` post-merge clause, prompt the user to run an explanatory query against production at https://graphrag.norfolkaibi.com/ and confirm in one sentence:

- `**Sources:**` block is gone from the response body.
- Accordion rows show `[N]` chips matching inline citation chip styling.
- Hovering a row title shows the chunk excerpt tooltip.

Wait for the user's confirmation. If they report any regression, halt and discuss rollback options.

- [ ] **Step 5: Close out memory + tracking**

After user confirms production is healthy:

- Update `~/.claude/projects/-Users-arthurfantaci-Projects-requirements-graphrag-api/memory/MEMORY.md` "Recent decisions" with a one-row entry describing the PR shipped, the commit hash, the test count change, and any production-verification notes.
- Verify issue #357 was auto-closed by the `Closes #357` trailer in the PR body. If not, close manually with a brief outcome comment.

This step is bundled into the implementation branch's outcome per the global "no separate workflow for CLAUDE.md / memory" rule — memory updates do not need their own commit or PR.

---

## Self-Review

After writing this plan, I checked it against the spec with fresh eyes:

**Spec coverage check** — every section in `2026-05-23-unified-source-attribution-design.md` maps to a task:

- Problem → context for the implementer (header summary + spec reference).
- Goals → goal statement at the top.
- Non-goals → not separately tasked; implementer reads spec for context. The "out of scope" items (Tooltip widget tuning, SourceBadge extraction, etc.) are not touched in any task.
- Approach — Backend change → Task 1.
- Approach — Frontend changes (row numbering + tooltip + map pass-through) → Tasks 2 and 3.
- Approach — Excerpt truncation helper → Task 3 Step 1.
- Data flow → no task needed (no schema changes).
- Error handling → covered by Task 3's `'No excerpt available.'` fallback and the existing `Tooltip` widget's viewport-clamp behavior.
- Testing — backend → Task 1.
- Testing — frontend → Task 2 Step 3, Task 3 Step 3, Task 4 Step 5 (HITL).
- HITL gate sequence → Task 4 Step 5 (pre-merge), Task 5 Steps 3-4 (post-merge).
- Branch / PR shape → Task 4 Step 3 (PR creation), Task 5 Step 1 (squash-merge with delete-branch).

**Placeholder scan** — no "TBD", "TODO", "implement later", "appropriate error handling", or similar vague language. Every step has either concrete code blocks or concrete shell commands.

**Type consistency check** — `sourceNumber` prop name is used identically in Task 2 (introduced) and Task 3 (extended). `buildExcerpt` function name is defined and called consistently. The class string for the chip is identical across the spec and Tasks 2 & 3.

**Spec deviation noted** — Task 1 uses source-inspection rather than the spec's runtime assertion for the regression test. Rationale is documented in the task header. This is a stricter invariant than the spec's intent, so it satisfies the spec's goal.

No remediation needed; plan is internally consistent.
