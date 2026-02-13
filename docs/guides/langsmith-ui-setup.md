# LangSmith UI Configuration Guide

> Step-by-step instructions for configuring LangSmith UI components that the
> GraphRAG codebase depends on. Complete these after merging the Platform-First
> Architecture (PR #187).
>
> **Time estimate**: ~45 minutes
> **Prerequisites**: LangSmith admin access
> **Organization**: Norfolk AI|BI
> **Workspace**: graphrag-api
> **Project**: graphrag-api-prod

---

## Table of Contents

1. [Section A: Annotation Queues](#section-a-annotation-queues-4-queues)
2. [Section B: Online Evaluators](#section-b-online-evaluators-4-evaluators)
3. [Section C: Automation Rules](#section-c-automation-rules-5-rules)
4. [Section D: Dashboards](#section-d-dashboards-3-dashboards)
5. [Section E: Alerts](#section-e-alerts-3-alerts)
6. [Section F: Verification Checklist](#section-f-verification-checklist)

---

## Existing Resources (pre-configuration)

Before starting, note what already exists in the **graphrag-api** workspace:

| Component | Existing | Notes |
|-----------|----------|-------|
| Tracing Projects | `graphrag-api-prod`, `graphrag-api-dev`, `evaluators` | Production, dev, and eval runs |
| Datasets | `graphrag-rag-golden` (39 ex), `graphrag-agentic-eval` (16 ex), `graphrag-text2cypher-eval` (11 ex), `graphrag-critic-eval` (10 ex), `graphrag-intent-classifier-eval` (18 ex) | Legacy datasets; new per-vector datasets created by migration script |
| Annotation Queues | `qa-random-sample`, `failed-routing-review`, `low-confidence-review` | Legacy queues from prior setup — keep or archive as desired |
| Prompts | 19 prompts with `production`/`staging` tags | Phase 1 complete |
| Custom Dashboards | `graphrag-api-prod` (1 chart) | Will be expanded in Section D |

---

## Section A: Annotation Queues (4 queues)

Annotation queues are a **workspace-level** resource (shared across all projects).
The backend routes negative user feedback to queues via `routes/feedback.py` →
`_resolve_queue_id()`, which resolves queue **names** to UUIDs. The names below
must match `evaluation/constants.py` exactly.

> **Note**: 3 legacy queues already exist (`qa-random-sample`, `failed-routing-review`,
> `low-confidence-review`). The 4 queues below are new — create them alongside the
> existing ones.

### How it works in the code

```
User clicks thumbs-down → FeedbackModal collects rubric scores
  → POST /feedback with {intent, rubric_scores, score: 0.0}
    → feedback.py checks body.score < 0.5
      → QUEUE_INTENT_MAP[body.intent] → queue name
        → _resolve_queue_id(client, queue_name) → UUID
          → client.add_runs_to_annotation_queue(queue_id, [run_id])
```

### Queue 1: `review-explanatory`

1. Go to **LangSmith** → workspace **graphrag-api** → **Annotation Queues** → **+ New Queue**
2. Configure:
   - **Name**: `review-explanatory`
   - **Description**: `Explanatory vector responses flagged by low hallucination scores or negative user feedback. Review for accuracy, completeness, and citation quality.`
   - **Default Dataset** (optional): Link to `graphrag-eval-explanatory` — this allows annotators to send reviewed traces directly into the golden dataset
3. Click **Create**

### Queue 2: `review-structured`

1. **+ New Queue**
2. Configure:
   - **Name**: `review-structured`
   - **Description**: `Structured (Text2Cypher) traces flagged by Cypher parse failures or negative user feedback. Review for query correctness and result accuracy.`
   - **Default Dataset** (optional): Link to `graphrag-eval-structured`
3. Click **Create**

### Queue 3: `review-conversational`

1. **+ New Queue**
2. Configure:
   - **Name**: `review-conversational`
   - **Description**: `Conversational traces flagged by low coherence scores or negative user feedback. Review for coherence and helpfulness.`
   - **Default Dataset** (optional): Link to `graphrag-eval-conversational`
3. Click **Create**

### Queue 4: `user-reported-issues`

1. **+ New Queue**
2. Configure:
   - **Name**: `user-reported-issues`
   - **Description**: `Catch-all queue for negative user feedback where intent is unknown or not provided. General quality review.`
   - **Default Dataset** (optional): Link to `graphrag-eval-explanatory` (most common intent)
3. Click **Create**

> **Verify**: After creating all 4, the feedback endpoint will resolve queue
> names on first use and cache the UUIDs. Test by submitting a thumbs-down
> from the frontend.

---

## Section B: Online Evaluators (4 evaluators)

Online evaluators are configured at the **project level** — set these up on
the `graphrag-api-prod` project (not `graphrag-api-dev`). They automatically
score a sample of production traces asynchronously (no latency impact on
users). Each evaluated trace is auto-upgraded to extended retention at ~10x
base cost — the sampling rates below are tuned to control this.

> **Note**: There is also a 5th evaluator (`online_eval_cypher.py`) that runs
> as a batch script outside LangSmith because it needs Neo4j access. See the
> [User Manual](langsmith-user-manual.md#use-case-4-running-the-online-cypher-evaluator) for scheduling instructions.

### Evaluator 1: Hallucination Detector

1. Go to **LangSmith** → **graphrag-api** workspace → project **graphrag-api-prod** → **Online Evaluation** tab → **+ Add Evaluator**
2. Configure:
   - **Name**: `hallucination`
   - **Evaluator Type**: LLM-as-judge
   - **Filter**: `has(metadata, "intent") and eq(metadata["intent"], "explanatory")`
   - **Sampling Rate**: `10%`
   - **Prompt**: Use the LangSmith default hallucination prompt, or paste:
     ```
     You are evaluating whether an AI response contains hallucinated information.

     Given the following context documents and AI response, determine if the
     response contains any claims not supported by the provided context.

     Context: {context}
     AI Response: {output}

     Score 1.0 if the response is fully grounded in the context.
     Score 0.0 if the response contains hallucinated claims.
     Score between 0 and 1 proportionally.
     ```
   - **Score Key**: `hallucination`
3. Click **Save**

### Evaluator 2: Coherence Evaluator

1. **+ Add Evaluator**
2. Configure:
   - **Name**: `coherence`
   - **Evaluator Type**: LLM-as-judge
   - **Filter**: `has(metadata, "intent") and eq(metadata["intent"], "conversational")`
   - **Sampling Rate**: `15%`
   - **Prompt**:
     ```
     You are evaluating the coherence of a conversational AI response.

     Given the conversation history and the AI response, evaluate whether:
     1. The response logically follows from the conversation context
     2. The response maintains consistency with prior turns
     3. The response properly resolves references from earlier messages

     Conversation History: {input}
     AI Response: {output}

     Score 1.0 for perfectly coherent responses.
     Score 0.0 for incoherent or contradictory responses.
     ```
   - **Score Key**: `coherence`
3. Click **Save**

### Evaluator 3: Answer Relevancy

1. **+ Add Evaluator**
2. Configure:
   - **Name**: `answer_relevancy`
   - **Evaluator Type**: LLM-as-judge
   - **Filter**: None (applies to all traces)
   - **Sampling Rate**: `5%`
   - **Prompt**:
     ```
     You are evaluating whether an AI response is relevant to the user's question.

     User Question: {input}
     AI Response: {output}

     Score 1.0 if the response directly addresses the question.
     Score 0.5 if partially relevant.
     Score 0.0 if the response is off-topic or unhelpful.
     ```
   - **Score Key**: `answer_relevancy`
3. Click **Save**

### Evaluator 4: Cypher Parse Validity

1. **+ Add Evaluator**
2. Configure:
   - **Name**: `online_cypher_parse`
   - **Evaluator Type**: Code (Python)
   - **Filter**: `has(metadata, "intent") and eq(metadata["intent"], "structured")`
   - **Sampling Rate**: `100%`
   - **Code**:
     ```python
     def evaluate(run, example=None):
         """Check if generated Cypher is syntactically valid."""
         outputs = run.outputs or {}
         cypher = outputs.get("cypher", "") or outputs.get("output", "")

         if not cypher or not cypher.strip():
             return {"key": "online_cypher_parse", "score": 0, "comment": "Empty Cypher"}

         # Basic parse validation: must have MATCH or RETURN
         cypher_upper = cypher.upper().strip()
         has_match = "MATCH" in cypher_upper
         has_return = "RETURN" in cypher_upper

         if has_match and has_return:
             return {"key": "online_cypher_parse", "score": 1, "comment": "Valid Cypher structure"}

         missing = []
         if not has_match:
             missing.append("MATCH")
         if not has_return:
             missing.append("RETURN")

         return {
             "key": "online_cypher_parse",
             "score": 0,
             "comment": f"Missing required clause(s): {', '.join(missing)}",
         }
     ```
   - **Score Key**: `online_cypher_parse`
3. Click **Save**

---

## Section C: Automation Rules (5 rules)

Automation rules connect online evaluator scores to annotation queues. This is
a **2-step process**: the evaluator scores the trace (producing feedback), then
the rule filters by that score and routes to a queue.

> **Important**: Create these AFTER both the queues (Section A) and evaluators
> (Section B) exist.

### Rule 1: Low Hallucination → review-explanatory

1. Go to **LangSmith** → **graphrag-api** workspace → project **graphrag-api-prod** → **Rules** tab → **+ Add Rule**
2. Configure:
   - **Name**: `route-low-hallucination`
   - **Trigger**: Feedback received
   - **Condition**: Feedback key = `hallucination` AND score < `0.7`
   - **Action**: Add to annotation queue → `review-explanatory`
3. Click **Save**

### Rule 2: Low Coherence → review-conversational

1. **+ Add Rule**
2. Configure:
   - **Name**: `route-low-coherence`
   - **Trigger**: Feedback received
   - **Condition**: Feedback key = `coherence` AND score < `0.6`
   - **Action**: Add to annotation queue → `review-conversational`
3. Click **Save**

### Rule 3: Cypher Parse Failure → review-structured

1. **+ Add Rule**
2. Configure:
   - **Name**: `route-cypher-failure`
   - **Trigger**: Feedback received
   - **Condition**: Feedback key = `online_cypher_parse` AND score = `0`
   - **Action**: Add to annotation queue → `review-structured`
3. Click **Save**

### Rule 4: Cypher Execution Failure → review-structured

1. **+ Add Rule**
2. Configure:
   - **Name**: `route-cypher-execution-failure`
   - **Trigger**: Feedback received
   - **Condition**: Feedback key = `online_cypher_execution` AND score = `0`
   - **Action**: Add to annotation queue → `review-structured`
3. Click **Save**

### Rule 5: Extended Retention Sampling

1. **+ Add Rule**
2. Configure:
   - **Name**: `extended-retention-sampling`
   - **Trigger**: Run completed
   - **Condition**: Sampling rate = `5%`
   - **Action**: Upgrade to extended retention
3. Click **Save**

> **Note**: Traces scored by online evaluators are already auto-upgraded to
> extended retention. This rule adds a small sample of *non-evaluated* traces
> to the extended pool for broader monitoring coverage.

---

## Section D: Dashboards (3 dashboards)

### Dashboard 1: Production Overview (expand existing `graphrag-api-prod`)

Your workspace already has a `graphrag-api-prod` dashboard with 1 chart.
Expand it to be the primary production overview:

1. Go to **LangSmith** → **graphrag-api** workspace → **Custom Dashboards** → open **graphrag-api-prod**
2. **Add filter group**: `metadata.intent` — this enables per-intent breakdown
3. **Customize charts**:
   - Ensure these charts exist (most are auto-generated):
     - Trace volume over time (grouped by `metadata.intent`)
     - Median latency over time (grouped by `metadata.intent`)
     - Error rate over time
     - Token usage over time
     - Cost over time
   - If any are missing, add them via **+ Add Chart**
4. **Rename** the dashboard to `Production Overview`
5. **Save**

### Dashboard 2: Evaluation Health (custom)

1. **+ New Dashboard** → name it `Evaluation Health`
2. Add these charts:
   - **Feedback score trending** (30 days):
     - Chart type: Line
     - Metric: Average feedback score
     - Group by: Feedback key
     - Filter: feedback key in (`hallucination`, `coherence`, `answer_relevancy`, `online_cypher_parse`, `online_cypher_execution`)
     - Time range: 30 days
   - **User feedback score trending** (30 days):
     - Chart type: Line
     - Metric: Average feedback score
     - Filter: feedback key = `user-feedback`
     - Time range: 30 days
   - **Evaluation volume**:
     - Chart type: Bar
     - Metric: Count of feedback items
     - Group by: Feedback key
     - Time range: 7 days
3. **Save**

### Dashboard 3: Intent Deep-Dive (custom)

1. **+ New Dashboard** → name it `Intent Deep-Dive`
2. Add these charts:
   - **Per-intent latency**:
     - Chart type: Line
     - Metric: Median latency
     - Group by: `metadata.intent`
     - Time range: 30 days
   - **Per-intent token usage**:
     - Chart type: Line
     - Metric: Total tokens
     - Group by: `metadata.intent`
     - Time range: 30 days
   - **Per-intent eval scores**:
     - Chart type: Line
     - Metric: Average feedback score
     - Group by: `metadata.intent`
     - Filter: feedback key in (`hallucination`, `coherence`, `online_cypher_parse`)
     - Time range: 30 days
   - **Intent distribution**:
     - Chart type: Pie
     - Metric: Trace count
     - Group by: `metadata.intent`
     - Time range: 7 days
3. **Save**

---

## Section E: Alerts (3 alerts)

### Alert 1: High Error Rate (Critical)

1. Go to **LangSmith** → **graphrag-api** workspace → project **graphrag-api-prod** → **Alerts** → **+ New Alert**
2. Configure:
   - **Name**: `high-error-rate`
   - **Metric**: Errored Runs (percentage)
   - **Condition**: > 5%
   - **Window**: 10 minutes
   - **Severity**: Critical
   - **Notification**: Configure email/Slack as needed
3. Click **Save**

### Alert 2: High Latency (Warning)

1. **+ New Alert**
2. Configure:
   - **Name**: `high-latency`
   - **Metric**: Latency (average)
   - **Condition**: > 8 seconds
   - **Window**: 5 minutes
   - **Severity**: Warning
   - **Notification**: Configure as needed

> **Note**: LangSmith does not support P95 latency alerts. Using average with
> a lower threshold (8s instead of 10s) as a proxy. Monitor P95 via the
> Intent Deep-Dive dashboard.

3. Click **Save**

### Alert 3: Hallucination Score Drop (Warning)

1. **+ New Alert**
2. Configure:
   - **Name**: `hallucination-score-drop`
   - **Metric**: Feedback Score (average) for key `hallucination`
   - **Condition**: < 0.65
   - **Window**: 24 hours
   - **Severity**: Warning
   - **Notification**: Configure as needed
3. Click **Save**

> **Not supported**: Cost-based alerts are not available in LangSmith. Monitor
> cost via the Production Overview dashboard or the LangSmith billing page.

---

## Section F: Verification Checklist

After completing all sections, verify the setup:

### Annotation Queues
- [ ] Navigate to **graphrag-api** workspace → **Annotation Queues** — all 4 queues are visible
- [ ] Each queue has the correct name (exact match to constants.py):
  - `review-explanatory`
  - `review-structured`
  - `review-conversational`
  - `user-reported-issues`
- [ ] Submit a thumbs-down from the frontend → check that the trace appears in the correct intent queue

### Online Evaluators
- [ ] Navigate to **graphrag-api-prod** → **Online Evaluation** — 4 evaluators listed
- [ ] Wait for production traffic → verify evaluator scores appear on traces
- [ ] Check that `hallucination` scores appear on explanatory traces
- [ ] Check that `coherence` scores appear on conversational traces
- [ ] Check that `online_cypher_parse` scores appear on structured traces
- [ ] Check that `answer_relevancy` scores appear on a sample of all traces

### Automation Rules
- [ ] Navigate to **graphrag-api-prod** → **Rules** — 5 rules listed
- [ ] Wait for a low hallucination score → confirm trace appears in `review-explanatory`
- [ ] Wait for a cypher parse failure → confirm trace appears in `review-structured`

### Dashboards
- [ ] **Production Overview** shows intent-grouped trace volume and latency
- [ ] **Evaluation Health** shows feedback score trending
- [ ] **Intent Deep-Dive** shows per-intent breakdowns

### Alerts
- [ ] Navigate to **Alerts** — 3 alerts listed and active
- [ ] (Optional) Trigger a test alert by temporarily lowering the threshold

### End-to-End Smoke Test
1. Open the GraphRAG chatbot
2. Ask an explanatory question → verify trace appears in LangSmith with `intent: explanatory`
3. Click thumbs-down → fill rubric → submit
4. Check LangSmith:
   - `user-feedback` score = 0.0 on the trace
   - `user-accuracy`, `user-completeness`, `user-citation_quality` rubric scores present
   - Trace routed to `review-explanatory` queue
5. Ask a structured question (e.g., "List all Challenges") → verify `intent: structured`
6. Wait for online evaluator → verify `online_cypher_parse` feedback appears

---

## Quick Reference: Names & Thresholds

| Component | Name | Key Config |
|-----------|------|------------|
| Queue | `review-explanatory` | Linked to `graphrag-eval-explanatory` dataset |
| Queue | `review-structured` | Linked to `graphrag-eval-structured` dataset |
| Queue | `review-conversational` | Linked to `graphrag-eval-conversational` dataset |
| Queue | `user-reported-issues` | Catch-all for unknown intent |
| Evaluator | `hallucination` | 10% of explanatory traces |
| Evaluator | `coherence` | 15% of conversational traces |
| Evaluator | `answer_relevancy` | 5% of all traces |
| Evaluator | `online_cypher_parse` | 100% of structured traces (code, no LLM) |
| Rule | `route-low-hallucination` | hallucination < 0.7 → review-explanatory |
| Rule | `route-low-coherence` | coherence < 0.6 → review-conversational |
| Rule | `route-cypher-failure` | online_cypher_parse = 0 → review-structured |
| Rule | `route-cypher-execution-failure` | online_cypher_execution = 0 → review-structured |
| Rule | `extended-retention-sampling` | 5% of all traces → extended retention |
| Alert | `high-error-rate` | > 5% errors for 10 min (Critical) |
| Alert | `high-latency` | > 8s avg for 5 min (Warning) |
| Alert | `hallucination-score-drop` | hallucination avg < 0.65 over 24h (Warning) |
