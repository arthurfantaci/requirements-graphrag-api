# LangSmith User Manual

> Detailed workflows with concrete examples for operating the GraphRAG
> LangSmith integration. For initial setup, see
> [langsmith-ui-setup.md](langsmith-ui-setup.md).

---

## Table of Contents

1. [Use Case 1: Reviewing User Feedback in Annotation Queues](#use-case-1-reviewing-user-feedback-in-annotation-queues)
2. [Use Case 2: Investigating Quality Drops with Dashboards](#use-case-2-investigating-quality-drops-with-dashboards)
3. [Use Case 3: Running Offline Evaluations Per-Vector](#use-case-3-running-offline-evaluations-per-vector)
4. [Use Case 4: Running the Online Cypher Evaluator](#use-case-4-running-the-online-cypher-evaluator)
5. [Use Case 5: Responding to Alerts](#use-case-5-responding-to-alerts)
6. [Use Case 6: Promoting Prompt Changes via Hub](#use-case-6-promoting-prompt-changes-via-hub)
7. [Use Case 7: Adding Examples to Golden Datasets](#use-case-7-adding-examples-to-golden-datasets)
8. [Use Case 8: Interpreting Rubric Feedback from the Frontend](#use-case-8-interpreting-rubric-feedback-from-the-frontend)
9. [Appendix: Feedback Key Reference](#appendix-feedback-key-reference)

---

## Use Case 1: Reviewing User Feedback in Annotation Queues

### When to use

When the annotation queues have accumulated traces for review, typically from:
- Users clicking thumbs-down in the chatbot (routed by `feedback.py`)
- Low online evaluator scores triggering automation rules

### Step-by-step

1. **Open the queue**: LangSmith → Annotation Queues → select a queue (e.g., `review-explanatory`)

2. **Review the trace**: For each item in the queue:
   - Read the **user question** (input)
   - Read the **AI response** (output)
   - Check **feedback scores** on the trace:
     - `user-feedback` — the overall thumbs up/down (0.0 or 1.0)
     - `user-accuracy`, `user-completeness`, `user-citation_quality` — rubric dimension scores (0.0–1.0 in 0.2 increments, from 1-5 stars)
     - `hallucination`, `coherence` — online evaluator scores (if present)
   - Check the **comment** field for user-provided details and metadata

3. **Annotate**: Use LangSmith's annotation controls to:
   - **Score the response** using the queue's rubric
   - **Add notes** explaining why the response was good/bad

4. **Add to dataset** (when appropriate): If the trace represents a good test case:
   - Click **Add to Dataset**
   - Select the appropriate dataset (e.g., `graphrag-eval-explanatory`)
   - Edit the expected output if the AI response was wrong
   - This grows the golden dataset for future evaluations

5. **Move to next**: Mark as reviewed and proceed to the next item

### Example: Reviewing an explanatory trace

```
Queue: review-explanatory
Reason: hallucination score = 0.4 (below 0.7 threshold)

Trace details:
  Input:  "What are the key phases of requirements engineering?"
  Output: "Requirements engineering has 7 phases: elicitation, analysis,
           specification, validation, management, tracing, and compliance."

Feedback on trace:
  hallucination: 0.4   ← triggered the queue routing
  user-feedback: (none) ← not from user, from online evaluator

Review:
  The response claims 7 phases and includes "compliance" which is not
  in the knowledge graph. Actual phases per the graph: elicitation,
  analysis, specification, validation, management. "Tracing" is a
  sub-activity, not a phase.

Action:
  - Score: 0.3 (inaccurate)
  - Note: "Hallucinated 'compliance' phase and miscounted"
  - Add to dataset: YES — with corrected expected output
```

### Queue routing logic

Understanding which queue a trace lands in:

| Source | Condition | Queue |
|--------|-----------|-------|
| User thumbs-down (explanatory intent) | `score < 0.5` AND `intent = "explanatory"` | `review-explanatory` |
| User thumbs-down (structured intent) | `score < 0.5` AND `intent = "structured"` | `review-structured` |
| User thumbs-down (conversational intent) | `score < 0.5` AND `intent = "conversational"` | `review-conversational` |
| User thumbs-down (no intent) | `score < 0.5` AND `intent = null` | `user-reported-issues` |
| Online evaluator: hallucination | `hallucination < 0.7` | `review-explanatory` |
| Online evaluator: coherence | `coherence < 0.6` | `review-conversational` |
| Online evaluator: cypher parse | `online_cypher_parse = 0` | `review-structured` |
| Batch script: cypher execution | `online_cypher_execution = 0` | `review-structured` |

---

## Use Case 2: Investigating Quality Drops with Dashboards

### When to use

When you receive an alert, notice a quality trend, or want to do a routine
quality review.

### Step-by-step

1. **Start with Production Overview**: Check the high-level health metrics:
   - Is trace volume normal? A sudden spike or drop may indicate deployment issues.
   - Is error rate elevated? Check by intent to isolate which vector is affected.
   - Is latency trending up? Compare by intent — structured (Text2Cypher) traces
     are typically slower due to Neo4j round-trips.

2. **Drill into Evaluation Health**: Check online evaluator trends:
   - Is the `hallucination` score trending down? This means explanatory responses
     are becoming less grounded — possible cause: context retrieval degradation or
     prompt drift.
   - Is `coherence` dropping? Conversational responses may have coreference
     resolution issues.
   - Is `online_cypher_parse` dropping? The Text2Cypher prompt may be generating
     malformed Cypher.

3. **Use Intent Deep-Dive** for per-intent analysis:
   - Compare token usage across intents — a spike may indicate prompt bloat.
   - Check per-intent latency — if structured is slow, Neo4j may be under load.

4. **Cross-reference with quality_check.py**:
   ```bash
   uv run python scripts/quality_check.py --project jama-mcp-graphrag --dry-run
   ```
   This compares the last 7 days against a 30-day baseline and shows degradation
   across 6 monitored keys.

### Example: Investigating a hallucination alert

```
Alert: hallucination-score-drop
Details: Average hallucination score < 0.65 over 24h

Investigation steps:
1. Open Evaluation Health dashboard
   → hallucination score dropped from 0.82 to 0.58 starting 6 hours ago

2. Open review-explanatory queue
   → 12 new items, all from the last 6 hours
   → Common pattern: responses cite "Section 4.3" but the source
     documents don't have section numbers

3. Check recent deployments
   → A prompt change was pushed to Hub 7 hours ago
   → The SYNTHESIS prompt was updated — new version asks the LLM to
     "cite specific sections" but the source chunks don't preserve
     section numbering

4. Resolution:
   → Revert the SYNTHESIS prompt to previous :production tag
   → Update the prompt to say "cite source articles" instead of sections
   → Re-run evaluation: uv run python scripts/run_vector_evaluation.py --vector explanatory
   → Verify hallucination scores recover before re-promoting
```

---

## Use Case 3: Running Offline Evaluations Per-Vector

### When to use

- Before promoting a prompt change to `:production`
- After modifying retrieval logic, handlers, or chain configuration
- As part of a release (CI Tier 3 runs this automatically)

### Step-by-step

1. **Choose your vector** based on what changed:
   - Changed SYNTHESIS or retrieval prompts → run `explanatory`
   - Changed TEXT2CYPHER or Neo4j prompts → run `structured`
   - Changed coreference resolver or conversation logic → run `conversational`
   - Changed intent classifier → run `intent`
   - Not sure → run `all`

2. **Run the evaluation**:
   ```bash
   # Single vector (fast — ~2 minutes for intent, ~5 min for explanatory)
   uv run python scripts/run_vector_evaluation.py --vector explanatory

   # All vectors (~15 minutes, costs ~$0.65)
   uv run python scripts/run_vector_evaluation.py --vector all
   ```

3. **Review results in LangSmith**:
   - Go to **Experiments** tab in your project
   - Find the experiment by name: `graphrag-{vector}-production-{timestamp}`
   - Check per-evaluator scores
   - Compare against previous experiments to detect regressions

4. **Run regression comparison** (if you have a baseline):
   ```bash
   uv run python scripts/compare_experiments.py \
       --current graphrag-explanatory-production-20260213-140000 \
       --baseline graphrag-explanatory-production-20260210-090000
   ```

### Evaluator reference by vector

**Explanatory** (8 evaluators):
| Evaluator | Type | What it measures |
|-----------|------|------------------|
| faithfulness | LLM-as-judge | Are claims supported by context? |
| answer_relevancy | LLM-as-judge | Does the answer address the question? |
| context_precision | LLM-as-judge | Are retrieved contexts relevant? |
| context_recall | LLM-as-judge | Was enough context retrieved? |
| answer_correctness | LLM-as-judge | Is the answer factually correct? |
| context_entity_recall | Deterministic | Are key entities from the reference present? |
| groundedness | LLM-as-judge | Is the answer grounded in provided sources? |
| hallucination | LLM-as-judge | Does the answer contain unsupported claims? |

**Structured** (6 evaluators):
| Evaluator | Type | What it measures |
|-----------|------|------------------|
| cypher_parse_valid | Deterministic | Does the Cypher parse without errors? |
| cypher_schema_adherence | Deterministic | Are all labels/relationships valid? |
| cypher_execution_success | Deterministic | Does the query execute against Neo4j? |
| cypher_safety | Deterministic | No write/admin operations? |
| result_shape_accuracy | Deterministic | Does the result shape match expected? |
| result_correctness | LLM-as-judge | Are results semantically correct? |

**Conversational** (4 evaluators):
| Evaluator | Type | What it measures |
|-----------|------|------------------|
| conv_coherence | LLM-as-judge | Logically follows conversation context? |
| conv_context_retention | LLM-as-judge | Retains information from prior turns? |
| coreference_accuracy | Deterministic | Resolves pronouns/references correctly? |
| conv_hallucination | LLM-as-judge | Stays grounded across conversation? |

**Intent** (1 evaluator):
| Evaluator | Type | What it measures |
|-----------|------|------------------|
| intent_accuracy | Deterministic | Correctly classifies user intent? |

---

## Use Case 4: Running the Online Cypher Evaluator

### When to use

The online Cypher evaluator is a batch script that must be run periodically
because it needs Neo4j access (which LangSmith's UI-based evaluators don't
have). It evaluates recent structured traces by actually executing the
generated Cypher against Neo4j.

### Step-by-step

1. **Run manually**:
   ```bash
   # Default: last 24h, up to 50 traces
   uv run python scripts/online_eval_cypher.py --project jama-mcp-graphrag

   # Preview without posting feedback
   uv run python scripts/online_eval_cypher.py --project jama-mcp-graphrag --dry-run

   # Broader window
   uv run python scripts/online_eval_cypher.py --project jama-mcp-graphrag \
       --hours-back 72 --limit 200
   ```

2. **Read the output**: The script logs per-trace results:
   ```
   [1/15] Evaluating run abc-123
     online_cypher_parse: PASS (1.0) — Valid Cypher structure
     online_cypher_safety: PASS (1.0) — Safe read-only query
     online_cypher_schema: PASS (1.0) — All labels/relationships valid
     online_cypher_execution: PASS (1.0) — Executed OK, 42 rows
   [2/15] Evaluating run def-456
     online_cypher_parse: PASS (1.0) — Valid Cypher structure
     online_cypher_safety: PASS (1.0) — Safe read-only query
     online_cypher_schema: FAIL (0.0) — Unknown label: Entity
     online_cypher_execution: FAIL (0.0) — Skipped — Unknown label: Entity
   ```

3. **Set up scheduled runs** (recommended):
   ```bash
   # Example crontab entry: run daily at 2 AM
   0 2 * * * cd /path/to/backend && uv run python scripts/online_eval_cypher.py \
       --project jama-mcp-graphrag >> /var/log/online_eval_cypher.log 2>&1
   ```

4. **Check results in LangSmith**: After running, structured traces will have
   feedback keys: `online_cypher_parse`, `online_cypher_safety`,
   `online_cypher_schema`, `online_cypher_execution`.

### The 4 metrics explained

| Metric | Score 1 | Score 0 |
|--------|---------|---------|
| `online_cypher_parse` | Cypher has MATCH + RETURN | Missing required clauses |
| `online_cypher_safety` | Read-only query | Contains write/admin keywords (Tier 1-3) |
| `online_cypher_schema` | All labels & relationships exist in schema | References unknown labels/relationships |
| `online_cypher_execution` | Executes against Neo4j (any row count) | Execution error or skipped due to prior failure |

---

## Use Case 5: Responding to Alerts

### Alert: High Error Rate (> 5%)

**Severity**: Critical

**Investigation steps**:
1. Open **Production Overview** dashboard → check error rate chart
2. Filter by `metadata.intent` to identify which vector is failing
3. Open **Traces** → filter by `is_error = true` → examine recent failures
4. Common causes:
   - Neo4j connection issues → check `NEO4J_URI` and database status
   - LLM API errors → check OpenAI/Anthropic status page
   - Timeout issues → check `middleware/timeout.py` TIMEOUTS config
5. Check deployment history — was a new version recently deployed?

**Resolution pattern**:
```
If Neo4j → check connection pool, restart if needed
If LLM API → wait for provider recovery, consider fallback model
If timeout → increase timeout or optimize the slow chain
If code bug → roll back deployment, fix, re-deploy
```

### Alert: High Latency (> 8s average)

**Severity**: Warning

**Investigation steps**:
1. Open **Intent Deep-Dive** dashboard → check per-intent latency
2. Identify which intent is slow:
   - **Structured** slow → Neo4j query execution time (check query complexity)
   - **Explanatory** slow → Retrieval + synthesis chain (check embedding service)
   - **Conversational** slow → Coreference resolution + retrieval chain
3. Open slow traces in LangSmith → check child run durations to find the bottleneck

**Resolution pattern**:
```
If retrieval slow → check vector DB / embedding service
If LLM slow → check model provider latency
If Neo4j slow → optimize Cypher queries, check indexes
If all intents slow → infrastructure issue (check Railway/hosting)
```

### Alert: Hallucination Score Drop (< 0.65)

**Severity**: Warning

**Investigation steps**:
1. Open **Evaluation Health** dashboard → check `hallucination` trend
2. Identify when the drop started
3. Cross-reference with:
   - Prompt changes (check Hub history)
   - Dataset changes (check if retrieval quality changed)
   - Model changes (was a different model deployed?)
4. Open `review-explanatory` queue → review flagged traces for patterns

**Resolution pattern**:
```
If prompt drift → revert to previous :production prompt tag
If retrieval degraded → check embedding model, re-index if needed
If model change → evaluate both models, pick the better performer
Run: uv run python scripts/run_vector_evaluation.py --vector explanatory
```

---

## Use Case 6: Promoting Prompt Changes via Hub

### When to use

When modifying any of the 13 managed prompts (SYNTHESIS, TEXT2CYPHER,
coreference resolver, intent classifier, evaluation judges, etc.).

### Step-by-step

1. **Edit locally** in `prompts/definitions.py`

2. **Test with staging tag**:
   ```bash
   # Push to Hub with staging tag
   uv run python scripts/sync_prompts_to_hub.py

   # Set environment to staging for local testing
   export PROMPT_ENVIRONMENT=staging

   # Run evaluation against the changed vector
   uv run python scripts/run_vector_evaluation.py --vector explanatory
   ```

3. **Compare against production baseline**:
   ```bash
   uv run python scripts/compare_experiments.py \
       --current graphrag-explanatory-staging-20260213-140000 \
       --baseline graphrag-explanatory-production-20260210-090000
   ```

4. **Promote if scores are equal or better**:
   - In LangSmith Hub, tag the new version as `:production`
   - Remove `:production` from the old version
   - Or use the API: `client.update_prompt(prompt_name, tags=["production"])`

5. **Verify** in production:
   ```bash
   uv run python scripts/check_prompt_sync.py
   ```

### Rollback

If the new prompt causes issues:
1. In LangSmith Hub, move the `:production` tag back to the previous version
2. The app will pick up the old version on next request (prompts are fetched
   from Hub at runtime, with caching)

---

## Use Case 7: Adding Examples to Golden Datasets

### When to use

- After reviewing traces in annotation queues (found good test cases)
- When improving coverage for a specific intent or edge case
- Before a major release to strengthen regression gates

### From the annotation queue (preferred)

1. Open a reviewed trace in the annotation queue
2. Click **Add to Dataset**
3. Select the target dataset:
   - `graphrag-eval-explanatory` for explanatory traces
   - `graphrag-eval-structured` for structured traces
   - `graphrag-eval-conversational` for conversational traces
   - `graphrag-eval-intent` for intent classification examples
4. Review and edit the fields:
   - **Input**: The user question (and conversation history for conversational)
   - **Expected output**: The correct/ideal response
   - **Metadata**: Intent, difficulty, category
5. Save

### From the script (bulk)

```bash
# Add examples from a JSON file
uv run python scripts/create_golden_dataset.py \
    --dataset graphrag-eval-explanatory \
    --file new_examples.json
```

JSON format:
```json
[
  {
    "input": "What is requirements traceability?",
    "expected_output": "Requirements traceability is the ability to...",
    "metadata": {
      "intent": "explanatory",
      "difficulty": "medium",
      "category": "concept"
    }
  }
]
```

### Best practices

- **Balance the dataset**: Don't over-represent one topic or difficulty level
- **Include edge cases**: Questions that are ambiguous, multi-intent, or domain-boundary
- **Keep expected outputs realistic**: Match what your system should produce, not a perfect textbook answer
- **Update regularly**: Add 2-5 examples per week from annotation queue reviews
- **Document provenance**: Note in metadata where the example came from (user report, evaluator flag, etc.)

---

## Use Case 8: Interpreting Rubric Feedback from the Frontend

### How the feedback flow works

When a user submits feedback through the chatbot:

1. **User clicks thumbs up/down** → `ResponseActions.jsx` opens `FeedbackModal.jsx`
2. **FeedbackModal shows intent-specific rubric** (star ratings):
   - **Explanatory**: Accuracy (1-5), Completeness (1-5), Citation Quality (1-5)
   - **Structured**: Query Quality (1-5), Result Correctness (1-5)
   - **Conversational**: Coherence (1-5), Helpfulness (1-5)
3. **User rates + optionally comments** → POST to `/feedback`
4. **Backend creates feedback**:
   - `user-feedback` key with score 0.0 or 1.0 (thumbs down/up)
   - `user-accuracy` key with score 0.0–1.0 (from 1-5 stars, mapped as star/5)
   - `user-completeness` key, `user-citation_quality` key, etc.
5. **If negative** (score < 0.5): trace routed to intent-specific queue

### Reading rubric scores in LangSmith

On a trace, you'll see feedback entries like:

| Key | Score | Meaning |
|-----|-------|---------|
| `user-feedback` | 0.0 | User gave thumbs-down |
| `user-accuracy` | 0.6 | 3 of 5 stars for accuracy |
| `user-completeness` | 0.4 | 2 of 5 stars for completeness |
| `user-citation_quality` | 0.8 | 4 of 5 stars for citations |

### Star-to-score mapping

| Stars | Score |
|-------|-------|
| 1 | 0.2 |
| 2 | 0.4 |
| 3 | 0.6 |
| 4 | 0.8 |
| 5 | 1.0 |

### Using rubric data for analysis

Filter feedback in LangSmith by key to analyze specific dimensions:
- Low `user-accuracy` across many traces → retrieval or factuality problem
- Low `user-completeness` → synthesis prompt may need to be more comprehensive
- Low `user-citation_quality` → source attribution needs improvement
- Low `user-cypher_quality` → Text2Cypher prompt generating poor queries
- Low `user-coherence` → coreference resolution or context threading issue

---

## Appendix: Feedback Key Reference

### User feedback keys (from frontend)

| Key | Source | Range | Meaning |
|-----|--------|-------|---------|
| `user-feedback` | All intents | 0.0 or 1.0 | Thumbs down / up |
| `user-accuracy` | Explanatory | 0.2–1.0 | Factual accuracy (5-star) |
| `user-completeness` | Explanatory | 0.2–1.0 | Topic coverage (5-star) |
| `user-citation_quality` | Explanatory | 0.2–1.0 | Source quality (5-star) |
| `user-cypher_quality` | Structured | 0.2–1.0 | Query quality (5-star) |
| `user-result_correctness` | Structured | 0.2–1.0 | Result accuracy (5-star) |
| `user-coherence` | Conversational | 0.2–1.0 | Logical consistency (5-star) |
| `user-helpfulness` | Conversational | 0.2–1.0 | Usefulness (5-star) |

### Online evaluator keys (from LangSmith UI evaluators)

| Key | Source | Range | Sampling |
|-----|--------|-------|----------|
| `hallucination` | Online evaluator (LLM) | 0.0–1.0 | 10% of explanatory |
| `coherence` | Online evaluator (LLM) | 0.0–1.0 | 15% of conversational |
| `answer_relevancy` | Online evaluator (LLM) | 0.0–1.0 | 5% of all |
| `online_cypher_parse` | Online evaluator (code) | 0 or 1 | 100% of structured |

### Batch script keys (from `online_eval_cypher.py`)

| Key | Range | Meaning |
|-----|-------|---------|
| `online_cypher_parse` | 0 or 1 | Has MATCH + RETURN |
| `online_cypher_safety` | 0 or 1 | Read-only (no write/admin) |
| `online_cypher_schema` | 0 or 1 | Valid labels/relationships |
| `online_cypher_execution` | 0 or 1 | Executes against Neo4j |

### Offline evaluator keys (from `run_vector_evaluation.py`)

| Key | Vector | Type |
|-----|--------|------|
| `faithfulness` | Explanatory | LLM-as-judge |
| `answer_relevancy` | Explanatory | LLM-as-judge |
| `context_precision` | Explanatory | LLM-as-judge |
| `context_recall` | Explanatory | LLM-as-judge |
| `answer_correctness` | Explanatory | LLM-as-judge |
| `context_entity_recall` | Explanatory | Deterministic |
| `groundedness` | Explanatory | LLM-as-judge |
| `hallucination` | Explanatory | LLM-as-judge |
| `cypher_parse_valid` | Structured | Deterministic |
| `cypher_schema_adherence` | Structured | Deterministic |
| `cypher_execution_success` | Structured | Deterministic |
| `cypher_safety` | Structured | Deterministic |
| `result_shape_accuracy` | Structured | Deterministic |
| `result_correctness` | Structured | LLM-as-judge |
| `conv_coherence` | Conversational | LLM-as-judge |
| `conv_context_retention` | Conversational | LLM-as-judge |
| `coreference_accuracy` | Conversational | Deterministic |
| `conv_hallucination` | Conversational | LLM-as-judge |
| `intent_accuracy` | Intent | Deterministic |

### Monitored by quality_check.py

These 6 keys are checked for degradation (7d vs 30d baseline):

1. `user-feedback`
2. `hallucination`
3. `coherence`
4. `answer_relevancy`
5. `online_cypher_parse`
6. `online_cypher_execution`
