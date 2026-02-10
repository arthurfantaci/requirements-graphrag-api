# Verification Runbook

**Last updated**: 2026-02-10
**Test baseline**: 837 tests passing (Phase 3.1a)

---

## When to Use Each Tier

| Tier | When | Cost | Time |
|------|------|------|------|
| **1. Automated** | Every branch, before every PR | Free | ~30s |
| **2. Studio** | When changes touch core pipeline | LLM tokens | ~5 min |
| **3. Post-Deploy** | After merge to main | Free | ~15s |

---

## Tier 1 — Automated (Pre-PR)

Run locally before pushing. CI runs the same checks, but catching issues locally is faster.

```bash
# One-liner
./scripts/verify.sh pre-pr

# Or manually:
cd backend
uv run ruff check .                                            # Lint
uv run ruff format --check .                                   # Format
uv run pytest -q                                               # Tests
uv run pytest --cov=src/requirements_graphrag_api --cov-report=term-missing  # Coverage (optional)
```

### Checklist

- [ ] Ruff lint passes (zero errors)
- [ ] Ruff format passes (zero reformats needed)
- [ ] All tests pass (check count matches baseline: 837+)
- [ ] No new test files without corresponding source changes
- [ ] Coverage >= 60% (enforced by `pyproject.toml`)

### Pre-commit hook

The Ruff format hook auto-fixes on first commit attempt. If commit fails:
1. Review the auto-fixed formatting
2. `git add` the reformatted files
3. Commit again (new commit, not `--amend`)

---

## Tier 2 — Manual Studio Testing

### When to Run

Only when changes affect these modules:
- `core/agentic/` (orchestrator, subgraphs, state)
- `core/retrieval.py`, `core/context.py`
- `synthesis.py`, `research.py`, `rag.py`
- `routing.py`, `text2cypher.py`
- `prompts/definitions.py` (prompt text changes)

### Start Dev Server

```bash
cd backend && langgraph dev --allow-blocking
```

Studio opens at: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

If the Studio tab connection fails (Chrome PNA issue):
1. Click the lock/tune icon in Chrome's address bar
2. Set "Local network access" to "Allow"
3. Reload the page

### Test Matrix

Run these 5 queries in Studio. Each covers a different intent and routing path.

---

#### Query 1: Simple Definition (EXPLANATORY)

```
What is requirements traceability?
```

**Expected path:** `initialize` -> `run_rag` -> `route_after_rag` -> `run_synthesis` -> END

**Verify:**
- [ ] `ranked_results` count > 0 after `run_rag`
- [ ] `route_after_rag` routes to `run_synthesis` (not fallback)
- [ ] `final_answer` is substantive (>100 chars)
- [ ] `final_answer` mentions traceability concepts
- [ ] `current_phase` = `"complete"` at END
- [ ] No `error` field set

**Golden reference:** `def-001` in `tests/benchmark/golden_dataset.py`

---

#### Query 2: Comparison (EXPLANATORY + Research)

```
Compare ISO 26262 and DO-178C traceability requirements
```

**Expected path:** `initialize` -> `run_rag` -> `route_after_rag` -> `run_research` -> `run_synthesis` -> END

**Verify:**
- [ ] `route_after_rag` routes to `run_research` (comparison keyword detected)
- [ ] `entity_contexts` populated after research (>0 entities)
- [ ] `entities_str` is non-empty going into synthesis
- [ ] `final_answer` mentions both ISO 26262 AND DO-178C
- [ ] Answer covers differences between the standards

**Why this matters:** Comparison queries are the most complex path through the graph. They exercise RAG + Research + Synthesis all in sequence.

---

#### Query 3: Structured Query (STRUCTURED)

```
Show all tools related to Jama Connect
```

**Expected path:** Routed to STRUCTURED intent -> Text2Cypher -> direct results

**Verify:**
- [ ] Intent classified as STRUCTURED
- [ ] Cypher query generated (check Text2Cypher output)
- [ ] Results returned as structured data (not prose)
- [ ] No synthesis step needed

**Note:** This query bypasses the orchestrator graph entirely — it's handled by the STRUCTURED path in the route handler. In Studio, you may need to send this via the `/chat` API endpoint instead.

---

#### Query 4: Conversational

```
Hello, how are you?
```

**Expected path:** Routed to CONVERSATIONAL intent -> direct LLM response

**Verify:**
- [ ] Intent classified as CONVERSATIONAL
- [ ] Does NOT enter RAG pipeline
- [ ] Response is friendly and conversational
- [ ] No sources/citations in response

---

#### Query 5: Domain-Qualified Edge Case

```
Tell me about the webinar on traceability
```

**Expected path:** `initialize` -> `run_rag` -> `route_after_rag` -> `run_synthesis` -> END

**Verify:**
- [ ] `ranked_results` > 0 (the old LLM grader rejected ALL docs for this query)
- [ ] Route goes to synthesis, NOT fallback
- [ ] `final_answer` references webinar content
- [ ] No TypeError in entity formatting (Phase 3.1a fix)

**Why this matters:** This was the regression case from Phase 3.1a. The LLM grader only saw `doc.content[:500]` and never saw article titles like "Webinar: Traceability", rejecting everything. The pass-through fix resolved this.

---

### Studio Debugging Tips

- **Click any node** to inspect input/output state
- **Check `current_phase`** at each step — should progress: `rag` -> `research`/`synthesis` -> `complete`
- **Check `error` field** — should be `None` throughout
- **Check `ranked_results`** after `run_rag` — this is the single most important diagnostic
- **Dev server hot-reloads** — save a Python file and it picks up changes automatically

---

## Tier 3 — Post-Deploy Verification

Run after merging to main, once Railway and Vercel have finished deploying.

```bash
# One-liner (set your URLs first)
export BACKEND_URL="https://your-app.railway.app"
export FRONTEND_URL="https://your-app.vercel.app"
./scripts/verify.sh post-deploy
```

### Backend Checks (Railway)

```bash
# 1. Health check — must return "healthy" + "connected"
curl -s $BACKEND_URL/health | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d['status'] == 'healthy', f'Status: {d[\"status\"]}'
assert d['neo4j'] == 'connected', f'Neo4j: {d[\"neo4j\"]}'
print('Health: OK')
"

# 2. Schema endpoint — must return node labels
curl -s $BACKEND_URL/schema | python3 -c "
import sys, json
d = json.load(sys.stdin)
labels = d.get('node_labels', [])
assert len(labels) > 0, 'No node labels'
print(f'Schema: {len(labels)} labels')
"

# 3. Vector search — must return results
curl -s -X POST $BACKEND_URL/search/vector \
  -H 'Content-Type: application/json' \
  -d '{"query": "traceability", "top_k": 3}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
results = d.get('results', [])
assert len(results) > 0, 'No search results'
print(f'Search: {len(results)} results')
"

# 4. Definitions endpoint — must return terms
curl -s $BACKEND_URL/definitions | python3 -c "
import sys, json
d = json.load(sys.stdin)
terms = d.get('terms', [])
assert len(terms) > 0, 'No terms'
print(f'Definitions: {len(terms)} terms')
"
```

### Frontend Checks (Vercel)

- [ ] App loads at `$FRONTEND_URL` (HTTP 200)
- [ ] Welcome screen renders (no blank page)
- [ ] Type "What is traceability?" -> SSE stream starts -> answer renders
- [ ] Sources panel shows at least 1 citation
- [ ] Intent badge shows "EXPLANATORY"
- [ ] No errors in browser DevTools console

### Checklist

- [ ] Backend `/health` returns `healthy` + `connected`
- [ ] Backend `/schema` returns node labels
- [ ] Backend `/search/vector` returns results for "traceability"
- [ ] Backend `/definitions` returns terms
- [ ] Frontend loads and renders
- [ ] End-to-end chat works (type question -> get answer with sources)

---

## Quick Reference

### Before Creating a PR

```bash
./scripts/verify.sh pre-pr
# If changes touch core pipeline:
cd backend && langgraph dev --allow-blocking
# Run Tier 2 queries in Studio
```

### After Merging to Main

```bash
# Wait for Railway + Vercel deploys to complete (~2-3 min)
export BACKEND_URL="https://your-app.railway.app"
export FRONTEND_URL="https://your-app.vercel.app"
./scripts/verify.sh post-deploy
# Open frontend and test one chat query manually
```

### Test Baseline History

| Date | Tests | Phase | Notes |
|------|-------|-------|-------|
| 2026-02-10 | 837 | Phase 3.1a | Baseline after LLM grader disable |
| 2026-02-09 | 839 | Phase 3.1 | CostTracker wiring |
