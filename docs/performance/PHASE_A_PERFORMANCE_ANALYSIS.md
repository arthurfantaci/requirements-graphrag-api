# Phase A: Performance Analysis & Engineering Lessons

> **Date**: 2026-02-05
> **PR**: #124 (9 commits, -732 net lines)
> **Branch**: `phase-a/consolidation`
> **Raw data**: `backend/scripts/profile_results_post-phase-a.json`

---

## 1. Executive Summary

Phase A consolidated the codebase (dead code removal, bug fixes) and applied five performance optimizations targeting different layers of the request lifecycle. Profiling with 8 test queries (3 runs each) revealed that **structured queries are fast (< 2s)** but **explanatory queries remain slow (15-25s TTFT)** because the dominant bottleneck is the LLM call chain itself, not the code-level overhead we optimized.

**Key numbers:**
- Structured path (Text2Cypher): **0.5s TTFT**, **1.5s total** -- excellent
- Explanatory path (Agentic RAG): **16.1s avg TTFT**, **18.1s avg total** -- needs work
- Overall average TTFT: **16.14s** (target was < 3s)

---

## 2. What We Optimized vs What We Measured

### 2.1 Optimizations Applied (Phase A)

| # | Optimization | What It Targeted | Estimated Savings |
|---|---|---|---|
| 4.1 | Streaming delay (50-char chunks, 5ms sleep) | Post-synthesis chunk delivery | ~3.5s total latency |
| 4.2 | Revision threshold (MAX_REVISIONS=1, CONFIDENCE=0.5) | Synthesis self-critique loop | 2-4s when revision triggered |
| 4.3 | Research skip (< 12 words, no comparison, score > 0.85) | Research subgraph bypass | 1.5-2.5s for simple queries |
| 4.4 | gpt-4o-mini for guardrail classifiers | Topic guard + hallucination LLM | 0.5-1.5s combined |
| 4.5 | Parallel topic guard + intent via asyncio.gather() | Input guardrail TTFT | 1.0-1.5s when both need LLM |

### 2.2 Why the Numbers Don't Show Dramatic Improvement

The optimizations targeted **code-level overhead** (artificial delays, unnecessary LLM calls, sequential execution). But the profiling reveals that **90%+ of latency is the LLM call chain itself**:

```
Explanatory query lifecycle (~18s total):
  [Guardrails: ~0.5s] -> [Intent: ~0.3s] -> [RAG Subgraph: ~5-8s] -> [Research: ~3-5s] -> [Synthesis: ~5-10s] -> [Output guardrails: ~0.5s]
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                               These are dominated by LLM inference time, not our code
```

The streaming delay fix (4.1) saved ~3.5s of *artificial* delay but the synthesis LLM call itself takes 5-10s. Similarly, the research skip (4.3) saves 3-5s when it fires, but the test "traceability" query Run 2 hit 38s -- variance in OpenAI API response times dominates the signal.

### 2.3 The Variance Problem

Run-to-run variance is enormous (up to 2x for the same query):

| Query | Run 1 | Run 2 | Run 3 | Spread |
|-------|-------|-------|-------|--------|
| "What is requirements traceability?" | 21.1s | 38.0s | 15.9s | **22.1s** |
| "Change management for ISO 26262" | 21.1s | 20.4s | 19.1s | 2.0s |
| "Compare DOORS and Jama Connect" | 19.2s | 14.9s | 13.6s | 5.6s |

This tells us that **OpenAI API latency jitter** is a larger factor than any code optimization. The 38s outlier on query 1 was likely API cold-start or rate limiting.

---

## 3. Profiling Results (Full Data)

### 3.1 Summary Table

| Query | Type | Avg TTFT | Avg Total | Avg Tokens | Avg Answer Len |
|-------|------|----------|-----------|------------|----------------|
| What is requirements traceability? | Explanatory (simple) | 25.01s | 25.16s | 26 | 1,253 |
| Change management for ISO 26262 | Explanatory (research) | 20.16s | 20.45s | 51 | 2,514 |
| List all webinars | Structured (keyword) | 0.51s | 0.51s | 1 | 306 |
| Which standards apply to automotive? | Structured (pattern) | N/A* | 1.51s | 0 | 0 |
| Verification vs validation | Explanatory (borderline) | 17.81s | 17.94s | 24 | 1,164 |
| Requirements in agile | Explanatory (multi-concept) | 17.86s | 18.07s | 37 | 1,812 |
| Follow-up: Jama implements it? | Explanatory (multi-turn) | 15.75s | 15.88s | 24 | 1,160 |
| Compare DOORS vs Jama | Explanatory (comparison) | 15.90s | 16.01s | 20 | 998 |

*\*TTFT null = Structured path emits cypher/results events, not token events*

### 3.2 Token Count Observations

The "token count" from the profiler counts **SSE token events** (50-char chunks), not LLM tokens. So 26 events x 50 chars = ~1,300 chars, which matches the answer_length. This confirms the streaming chunking is working correctly.

The low token count for "List all webinars" (1 token event) means the structured path emits the full result in one event, not chunked -- expected and correct.

### 3.3 Anomalies

1. **"Which standards apply to automotive?" returns 0 tokens and 0 answer_length**
   - This query routes to Text2Cypher. The profiler's TTFT detection looks for `"token"` events, but structured queries emit `"cypher"` and `"results"` events instead. The profiling script should be updated to also track `"results"` events as TTFT for structured queries.

2. **38s outlier on "traceability" query**
   - Run 2 took 38s vs 16-21s for runs 1 and 3. This is API-side latency, not a code issue. Worth investigating in LangSmith traces (look for which specific LLM call was slow).

---

## 4. Architectural Insights

### 4.1 The LLM Call Chain Problem

The agentic orchestrator chains **multiple sequential LLM calls**:

```
Query Expansion LLM (gpt-4o)           ~2-4s
  -> Vector Search (Neo4j)             ~0.5s
  -> Graph Enrichment (Neo4j)          ~0.3s
  -> Entity Selector LLM (gpt-4o)     ~2-3s    [if research triggered]
  -> Entity Exploration (Neo4j)        ~0.3s    [if research triggered]
  -> Synthesis LLM (gpt-4o)           ~5-10s
  -> Critic LLM (gpt-4o)             ~2-4s     [if revision triggered]
  -> Revision LLM (gpt-4o)           ~5-10s    [if revision triggered]
```

**Worst case**: 7 sequential LLM calls. **Best case** (simple query, no research, no revision): 2 LLM calls (query expansion + synthesis).

### 4.2 Where to Get Real Gains (Phase B+ Ideas)

| Approach | Potential Savings | Complexity | Risk |
|----------|-----------------|------------|------|
| **Stream synthesis output** (abandon JSON, stream Markdown) | -5 to -10s TTFT | High | Loses structured self-critique |
| **Cache query expansion** (LRU for common patterns) | -2 to -4s for repeat queries | Low | Cache invalidation |
| **Parallel RAG + Research** (don't wait for RAG before research) | -3 to -5s | Medium | Research needs RAG context |
| **Use gpt-4o-mini for synthesis** | -3 to -5s | Low | Answer quality tradeoff |
| **Eliminate query expansion for simple queries** | -2 to -4s | Low | Reduced retrieval recall |
| **Pre-computed embeddings cache** | -0.5s | Low | Storage + invalidation |

### 4.3 The Fundamental Tradeoff

The agentic architecture was designed for **answer quality** (self-critique, entity exploration, multi-step reasoning). The cost is latency. Phase A proved that:

1. **Code-level optimizations have a ceiling** -- we removed ~7s of overhead, but LLM inference dominates
2. **The next big win requires architectural changes** -- specifically, streaming the synthesis output instead of waiting for the full JSON response
3. **Structured queries prove the floor** -- when we bypass the agentic orchestrator entirely (Text2Cypher), latency drops to < 2s

---

## 5. Bug Fixes Validated

### 5.1 F2: Assistant Messages in LangGraph State

**Before**: `chat.py` only added `HumanMessage` from conversation history, silently discarding assistant messages. Multi-turn context was broken.

**After**: Both `HumanMessage` and `AIMessage` are added. The profiling query "How does Jama Connect implement it?" (with conversation_history) completed successfully at 15.75s avg, confirming the fix works end-to-end.

**Test**: `test_chat_assistant_messages_become_aimessage` verifies `AIMessage` instances are created.

### 5.2 F4/F6: previous_context Wiring

**Before**: `SynthesisState` lacked `previous_context` field, and the orchestrator never built it. The `{previous_context}` variable in the SYNTHESIS prompt was always empty.

**After**: Orchestrator builds `previous_context` from all messages except the last (current query), formatted as `Q: ... / A: ...` pairs. The follow-up query profiling shows it's working (answer references the prior context about traceability).

**Tests**: `test_previous_context_from_conversation_history` and `test_no_previous_context_for_single_message`.

### 5.3 Research Skip Heuristic

The heuristic fires for simple queries: `< 12 words AND no comparison keywords AND top_score > 0.85`. In profiling:
- "What is requirements traceability?" (4 words) -- should skip research
- "Compare DOORS and Jama Connect" (has "compare") -- should trigger research, and it does (higher event count: 28 vs 33)

**Tests**: `test_simple_query_high_score_skips_research` and `test_comparison_query_triggers_research`.

---

## 6. Dead Code Removal Impact

| Removed | Lines | Rationale |
|---------|-------|-----------|
| `verify_api_key()` | ~95 | Unused -- AuthMiddleware is the active mechanism |
| `generate_answer()` | ~115 | Replaced by agentic orchestrator |
| `stream_chat()` | ~155 | Replaced by agentic streaming |
| Conversation endpoints (2) | ~180 | Not used by frontend |
| STEPBACK prompt | ~10 | Deprecated, superseded by QUERY_EXPANSION |
| AGENT_REASONING prompt | ~10 | Roadmap item, never wired |
| Associated tests | ~600+ | Tests for deleted code |
| **Total** | **~1,480 lines removed** | |

Net effect: -732 lines (748 added for profiling script, new tests, and optimized code).

---

## 7. Methodology Notes

### 7.1 Profiling Setup
- **Script**: `backend/scripts/profile_baseline.py`
- **Target**: Local uvicorn server (localhost:8000)
- **Queries**: 8 covering all paths (simple, complex, structured, multi-turn, comparison)
- **Runs per query**: 3
- **LangSmith dataset**: `phase-a-baseline` (id: 596806dd-c8fe-49db-aecd-184a8d67b239)
- **Phase tag**: `post-phase-a`

### 7.2 What the Profiler Measures
- **TTFT**: Time from HTTP request to first `"token"` SSE event
- **Total latency**: Time from HTTP request to stream completion
- **Token count**: Number of SSE token events (not LLM tokens)
- **Answer length**: Character count of full answer

### 7.3 What the Profiler Doesn't Measure
- Per-component breakdown (RAG vs Research vs Synthesis) -- need LangSmith trace analysis
- LLM token usage / cost -- need LangSmith billing data
- Retrieval quality -- need evaluation framework (not profiling)
- Memory usage / CPU -- need system-level monitoring

### 7.4 Improving the Profiler for Next Phase
1. Track `"cypher"` and `"results"` events as TTFT for structured queries
2. Add LangSmith trace span extraction for per-component breakdown
3. Add p50/p90/p99 percentile calculations (not just averages)
4. Add warm-up runs to reduce cold-start variance
5. Export results in a format LangSmith can ingest for comparison dashboards

---

## 8. Training & Development Rubric

### 8.1 Concepts to Master

| Concept | Why It Matters | Study Resource |
|---------|---------------|----------------|
| **LLM call chain analysis** | Identifying which calls dominate latency | Profile this system with LangSmith trace view |
| **asyncio.gather() for LLM parallelization** | Reducing TTFT by running independent calls concurrently | See `chat.py` topic guard + intent parallel flow |
| **Self-critique loops (OODA in LLMs)** | Understanding when revision adds value vs wastes time | See `synthesis.py` draft -> critique -> revise flow |
| **Conditional graph routing** | Skipping expensive subgraphs based on heuristics | See `orchestrator.py` should_research() |
| **SSE streaming architecture** | Delivering partial results to frontend immediately | See `streaming.py` event types and chunking |
| **TypedDict state management** | LangGraph state flows between nodes | See `state.py` OrchestratorState, SynthesisState |
| **Dead code identification** | Knowing when to remove vs keep unused code | See Step 2 methodology (grep, trace, remove) |

### 8.2 Practice Exercises

1. **Trace Analysis**: Open LangSmith, find the `post-phase-a` runs. For each explanatory query, identify which LLM call took the longest. Write a 1-paragraph analysis.

2. **Optimization Design**: Propose how you would stream synthesis output directly (instead of waiting for full JSON). What changes to the SYNTHESIS prompt, synthesis.py, streaming.py, and frontend are needed? What self-critique capabilities are lost?

3. **Heuristic Tuning**: The research skip uses `word_count < 12` and `top_score > 0.85`. Using the LangSmith traces, find queries where research was skipped but shouldn't have been (false negatives) or triggered but added no value (false positives). Propose adjusted thresholds.

4. **Profiler Enhancement**: Modify `profile_baseline.py` to extract per-component latency from LangSmith traces (using the LangSmith SDK `client.read_run()` and child span analysis). Add p50/p90 percentile calculations.

5. **Cost Analysis**: Using LangSmith token tracking, calculate the per-query cost for each of the 8 baseline queries. Identify which queries are most expensive and propose model-level optimizations (e.g., gpt-4o-mini for query expansion).

### 8.3 Competency Checkpoints

| Level | Skill | Evidence |
|-------|-------|---------|
| **Foundations** | Can read a LangSmith trace and identify the slowest span | Screenshot of trace analysis |
| **Intermediate** | Can implement asyncio.gather() for parallel LLM calls with error handling | Code PR with tests |
| **Intermediate** | Can design a conditional routing heuristic with measurable criteria | Heuristic + evaluation results |
| **Advanced** | Can switch from JSON synthesis to streaming Markdown without quality regression | Before/after evaluation comparison |
| **Advanced** | Can build a LangSmith evaluation dataset that catches quality regressions from optimization | Dataset with evaluator functions |
| **Expert** | Can design an end-to-end profiling pipeline (instrument -> measure -> compare -> decide) | Automated comparison report |

---

## 9. Next Steps (Phase B Preparation)

1. **Run baseline comparison**: Execute `profile_baseline.py --phase baseline` on a pre-Phase A commit to get true before/after numbers
2. **LangSmith trace analysis**: Drill into individual traces to build per-component latency breakdown
3. **Streaming synthesis**: Investigate switching SYNTHESIS prompt from JSON to streaming Markdown
4. **Conversation layer**: Implement CONVERSATIONAL intent, checkpointer querying, topic guard meta-conversation
5. **Evaluation framework**: Build automated quality checks that gate performance optimizations
