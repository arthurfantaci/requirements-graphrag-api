# Unified Source Attribution — Design Spec

**Date**: 2026-05-23
**Status**: Approved (brainstorming complete; awaiting implementation plan)
**Branch**: `feat/unified-source-attribution`
**Tracking issue**: [#357](https://github.com/arthurfantaci/requirements-graphrag-api/issues/357)
**Produced via**: `superpowers:brainstorming` skill workflow

## Problem

Source attribution in explanatory LLM responses currently uses three different and incompatible numbering schemes that the user cannot reconcile:

1. **Inline citation chips** `[Source N]` in the response body. Numbers derive from retrieval order, built in `backend/src/requirements_graphrag_api/core/context.py:291` (`[Source {i}: {title}]` markers fed into the synthesis prompt).
2. **Static `**Sources:**` footer** appended to every explanatory answer in `backend/src/requirements_graphrag_api/core/agentic/subgraphs/synthesis.py:283-287`. Numbers derive from the LLM-judged `citations` field parsed out of the synthesis JSON output (`synthesis.py:107, 222`). LLM-judged order can and does diverge from retrieval order.
3. **Sources accordion** at the bottom of the response (`frontend/src/components/metadata/SourcesPanel.jsx`). Renders the SSE `sources[]` array in retrieval order — the same order as (1) — but accordion rows have no visible numbers, so users cannot visually link inline chips to accordion rows.

End-users see all three at once. The divergence between (1) and (2) makes the existing inline chips appear "wrong" relative to the prominent static list. PR #354 (LangSmith Issues Agent) attempted to reconcile by forcing (1) to follow (2)'s numbering — which inverted the data direction (LLM-judged → deterministic retrieval) and made the symptom worse. That PR was rolled back as `fe3c6b7`.

## Goals

- Eliminate the divergent third numbering scheme by removing the static `**Sources:**` footer.
- Make the existing alignment between inline `[Source N]` chips and accordion rows visually unmistakable by numbering accordion rows with chip styling identical to inline citations.
- Surface the retrieved chunk text on demand via row-title hover tooltip, so users can audit *why* a row was cited without leaving the response.

## Non-goals

- **Backfilling historical responses.** Forward-only.
- **Tuning the shared Tooltip widget's `tooltipHeight = 80` position-flip constant** (`frontend/src/components/ui/Tooltip.jsx:20`). Pre-existing limitation; tuning it touches a widget shared across components and is a separate concern.
- **Extracting a shared `SourceBadge` component.** The emerald chip class string is one line of Tailwind. The two usages — inline chip (anchor + tooltip + click-to-open) vs. accordion-row prefix (plain span, non-interactive) — have meaningfully different surrounding structure. An abstraction would have to parameterize both, and the parameterization cost exceeds the duplication cost.
- **Removing the upstream `citations` extraction** at `synthesis.py:107, 222`. The field stays in `SynthesisState` in case downstream consumers (e.g., LangSmith telemetry) read it. Only the *rendering append* in `format_output` is removed.
- **Changing the inline `[Source N]` chip tooltip** in `SourceCitationRenderer.jsx:36-48`. It stays lean (title + relevance + click hint). The accordion-row tooltip provides progressive disclosure — chip identifies, row investigates.
- **Cypher / structured-response surfaces.** This PR touches the explanatory (RAG) response only.

## Approach

Three coordinated changes; two files of production code plus one test file; ~80-line net diff; single bundled PR.

### Backend change (1 file)

`backend/src/requirements_graphrag_api/core/agentic/subgraphs/synthesis.py`, function `format_output`:

Delete line 281 (the now-misleading section comment) and lines 283-287 (the `if citations:` block and its body):

```python
# Build final answer with citation footer         ← line 281, delete
final = draft                                       ← line 282, KEEP
if citations:                                       ← line 283, delete
    citation_text = "\n\n**Sources:**\n"           ← line 284, delete
    for i, source in enumerate(citations, 1):      ← line 285, delete
        citation_text += f"- [{i}] {source}\n"     ← line 286, delete
    final += citation_text                          ← line 287, delete
```

Line 282 (`final = draft`) MUST be retained — it is the initialization that the confidence-indicator block at lines 290-294 still appends to. The `citations = state.get("citations", [])` extraction at line 276 also stays — only the rendering append is removed, so `citations` remains available in `SynthesisState` for downstream consumers (e.g., LangSmith telemetry).

### Frontend changes (1 file)

`frontend/src/components/metadata/SourcesPanel.jsx`:

**1. Row numbering chip.** Prefix each `SourceItem` with a `[N]` span using the exact class string from `SourceCitationRenderer.jsx:30`, adjusted for the row-prefix context:

```jsx
<span className="inline-flex items-center px-1.5 py-0.5 mr-2 rounded text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 flex-shrink-0">
  {sourceNumber}
</span>
```

`mr-2` replaces the original `mx-0.5` — the row-prefix context wants right-margin spacing only, not symmetric. `flex-shrink-0` is added so the chip never shrinks when the title is long. The chip is non-interactive (no anchor wrapper, no hover state) because the row already exposes click affordance via the existing external-link icon.

**2. Row-title hover tooltip.** Wrap the title `<p>` in the shared `<Tooltip>` wrapper from `frontend/src/components/ui/Tooltip.jsx:159-185`:

```jsx
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
```

The `<Tooltip>` wrapper already supplies `cursor-help` on its trigger span. `excerpt` is computed via the local helper described below. The underlying `<p>` keeps its existing `truncate` class for visual handling of long titles within the row.

**3. Pass row number through.** At `SourcesPanel.jsx:134`, change the map call to pass the 1-indexed number:

```jsx
{sources.map((source, index) => (
  <SourceItem key={`source-${index}`} source={source} sourceNumber={index + 1} />
))}
```

### Excerpt truncation helper (frontend)

Module-private function at the top of `SourcesPanel.jsx`. Inline rather than a separate file — small, single-use, no test framework on the frontend to share with.

```javascript
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

Key choices:

- **`EXCERPT_MAX_CHARS = 280`**: ~3-4 lines at the tooltip's `text-xs leading-relaxed` styling. Readable without dominating the viewport.
- **`BOUNDARY_FLOOR = 100`**: prevents pathological cases like "Hi. <very long sentence>" from truncating to "Hi." — if the only sentence terminator is too close to the start, fall through to word boundary.
- **Whitespace normalization** (`replace(/\s+/g, ' ')`): chunks carry embedded `\n` from source HTML; `<p>` does not honor newlines as breaks, so normalizing prevents awkward gap rendering.
- **`…` (U+2026)** not `...`: single codepoint, correct typography.
- Returns `null` when there is no usable content — caller renders the fallback string `'No excerpt available.'`.

## Data flow

No payload-schema changes. The backend SSE stream already publishes `sources: list[dict]` where each entry includes `{title, content, url, chunk_id, relevance_score}` (`core/context.py:297-304`). The frontend `SourcesPanel` already iterates this array in retrieval order — the *same* order used to build `[Source i: {title}]` markers in `core/context.py:291`. The design surfaces this existing alignment; it does not change the underlying data.

## Error handling

| Condition | Behavior |
|---|---|
| `source.content` missing / empty / whitespace-only | Tooltip renders with `description="No excerpt available."` — consistent affordance across all rows, no conditional hiding. |
| `source.content` very long (>2000 chars) | Same truncation path; ellipsis at the boundary. No special case. |
| `source.content` contains HTML or markdown syntax | React's text rendering escapes by default. The excerpt renders inside `<p>{description}</p>` in `TooltipContent` (`Tooltip.jsx:135`) — pure React text rendering, no raw-HTML injection paths involved. **No XSS surface introduced.** |
| Tooltip near viewport top edge | Existing widget's flip-to-bottom logic (`Tooltip.jsx:33-51`) applies. May be off by ~40px due to `tooltipHeight = 80` constant (actual excerpt height ~120-160px); viewport-clamp prevents off-screen render. Acceptable cosmetic. |
| Mobile viewport | Widget clamps to `w-64 max-w-xs` (256-320px) and adjusts horizontally within bounds (`Tooltip.jsx:24-27`). No new behavior. |
| `sources` empty array | Existing early-return at `SourcesPanel.jsx:108` (`if (!sources || sources.length === 0) return null`) handles. No new code. |

## Testing

### Backend

Run from `backend/`:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --tb=short
```

Tests live in `backend/tests/test_core/test_agentic/test_subgraphs.py`, class `TestSynthesisSubgraph`. The class currently has seven tests; none of them invoke `format_output` directly or assert on the `**Sources:**` rendering — so there is nothing existing to UPDATE. We are ADDING one new regression test:

**New test — `test_format_output_no_sources_footer`** (or similar name): construct a `SynthesisState` with `draft_answer` and `citations` populated, invoke the compiled subgraph (or call the `format_output` node directly via the same pattern as the existing tests), and assert all of:
- `final_answer` does NOT contain `"**Sources:**"`.
- `final_answer` does NOT contain `"\n- ["` (the citation-list marker pattern).
- The returned state (or the input state, depending on test scope) still carries `citations` so downstream consumers (LangSmith telemetry) are not broken by the rendering removal.

Expected suite count: 852 → 853 (one net new test added; no existing tests changed).

### Frontend

No test framework per project posture. Verification path:

1. **Local**: `npm run dev` from `frontend/`. Exercise with an explanatory query that returns 3+ sources (e.g., "What is requirements traceability?"). Verify the seven checks listed below.
2. **Vercel Preview**: after push, the Preview deployment URL gets the same seven checks, exercised by the user per `feedback_hitl-before-production.md`.

Seven checks (these form the HITL prompt the user will receive when the preview is ready):

1. LLM response body shows no `**Sources:**` block at the bottom.
2. Accordion header still shows `Sources (N)`.
3. Expanding the accordion reveals `[1]`, `[2]`, … emerald chips at row start, matching inline-chip styling.
4. Hovering an accordion row title shows a dark tooltip with the chunk text excerpt.
5. Long chunks truncate with `…`; short chunks display fully; rows where `content` is empty show "No excerpt available."
6. External-link icon on each row still opens the source URL in a new tab.
7. All pre-existing accordion behavior preserved (expand/collapse, info icon tooltip, relevance bar).

## HITL gate sequence

Per `~/.claude/projects/-Users-arthurfantaci-Projects-requirements-graphrag-api/memory/feedback_hitl-before-production.md`:

1. **Pre-merge** — once CI green AND Vercel Preview Ready, surface the preview URL with the seven-check list above. User exercises manually. User approves merge.
2. **Post-merge** — backend change ships to Railway via auto-deploy (no preview environment for backend in this stack; Vercel preview frontend points at production Railway via `VITE_API_URL`). Prompt the user to run an explanatory query against production after Railway deploy completes, with one-line confirmation that (a) `**Sources:**` block is gone, (b) accordion numbering is visible, (c) hover tooltip is working.

## Branch / PR shape

- **Tracking issue**: [#357](https://github.com/arthurfantaci/requirements-graphrag-api/issues/357).
- **Branch**: `feat/unified-source-attribution`.
- **Commits** (three logical, squashed at merge):
  1. `refactor(synthesis): remove redundant Sources footer from final answer`
  2. `feat(ui): number Sources accordion rows with citation-style chips`
  3. `feat(ui): show retrieved chunk excerpt on Sources row hover`
- **Spec doc** (this file): commits with the implementation branch per project CLAUDE.md (this spec is code-adjacent and drives the implementation plan — NOT a docs-only branch/PR).
- **PR**: squash-merge with `--delete-branch` per standard workflow. PR description references this spec by path.

## Open questions

None remaining after Gates 1 and 2 of the brainstorming workflow. All design choices captured above.

## Next step

Invoke `superpowers:writing-plans` skill to produce the implementation plan that breaks this spec into ordered, executable tasks. The plan will be a separate file (also on this branch) that the implementation session works from.
