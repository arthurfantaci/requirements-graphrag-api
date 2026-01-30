# LangSmith Evaluation Guide

This guide explains how to use LangSmith for evaluating the GraphRAG pipeline using RAGAS metrics.

## Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. CREATE      │────▶│  2. RUN         │────▶│  3. COMPARE     │────▶│  4. ITERATE     │
│  DATASET        │     │  EVALUATION     │     │  EXPERIMENTS    │     │  AND IMPROVE    │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Prerequisites

### Environment Variables

```bash
# Required in .env
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
OPENAI_API_KEY=sk-...

# Optional
LANGSMITH_PROJECT=graphrag-api-dev
```

### Installation

```bash
cd backend
uv sync --extra dev
```

## Step 1: Create Golden Dataset

The golden dataset contains curated question-answer pairs for evaluating the RAG pipeline.

### Create the Dataset

```bash
# Preview what will be created
uv run python scripts/create_golden_dataset.py --dry-run

# Create the dataset in LangSmith
uv run python scripts/create_golden_dataset.py
```

### Dataset Structure

Each example contains:

```python
{
    "inputs": {
        "question": "What is requirements traceability?"
    },
    "outputs": {
        "expected_answer": "Requirements traceability is...",
        "expected_sources": ["Article 1", "Article 2"],
        "intent": "explanatory"
    },
    "metadata": {
        "difficulty": "easy",
        "domain": "core_concepts"
    }
}
```

### View in LangSmith UI

After creation, view the dataset at:
https://smith.langchain.com/datasets

## Step 2: Run RAGAS Evaluation

The evaluation script runs the RAG pipeline against the golden dataset and scores outputs using RAGAS evaluators.

### RAGAS Metrics

| Metric | Description | What It Measures |
|--------|-------------|------------------|
| **Faithfulness** | Is the answer grounded in context? | Prevents hallucination |
| **Answer Relevancy** | Does the answer address the question? | Response quality |
| **Context Precision** | Are retrieved contexts relevant? | Retrieval quality |
| **Context Recall** | Do contexts contain ground truth? | Retrieval completeness |

### Run Evaluation

```bash
# Basic evaluation
uv run python scripts/run_ragas_evaluation.py

# With custom experiment name
uv run python scripts/run_ragas_evaluation.py --experiment-name "ragas-baseline-v1"

# Filter to only explanatory queries (RECOMMENDED for RAGAS)
uv run python scripts/run_ragas_evaluation.py --filter-intent explanatory

# With different model
uv run python scripts/run_ragas_evaluation.py --model gpt-4o

# Dry run to preview
uv run python scripts/run_ragas_evaluation.py --dry-run
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--dataset, -d` | Dataset name (default: graphrag-rag-golden) |
| `--experiment-name, -e` | Custom experiment name |
| `--model, -m` | Model for generation (default: gpt-4o-mini) |
| `--filter-intent, -f` | Filter by intent: `explanatory` or `structured` |
| `--max-concurrency` | Parallel evaluations (default: 4) |
| `--verbose, -v` | Detailed output |
| `--list, -l` | List available datasets |

### Why Filter by Intent?

The golden dataset contains two types of queries:

| Intent | Purpose | RAGAS Evaluation |
|--------|---------|------------------|
| **explanatory** | Concept questions → RAG pipeline | Full RAGAS metrics meaningful |
| **structured** | List/count queries → Text2Cypher | Context recall not applicable |

**Recommendation:** Use `--filter-intent explanatory` for meaningful RAGAS scores, since structured queries have placeholder expected answers that don't work with context_recall evaluation.

### View Results

After running, view experiments at:
https://smith.langchain.com/experiments

## Step 3: Compare Experiments

Compare different experiments to make data-driven decisions about changes.

### List Experiments

```bash
uv run python scripts/compare_experiments.py --list
```

### Show Experiment Details

```bash
uv run python scripts/compare_experiments.py --show ragas-baseline-v1
```

### Compare Two Experiments

```bash
uv run python scripts/compare_experiments.py --compare ragas-baseline-v1 ragas-new-prompt-v1
```

### Interpreting Results

```
EXPERIMENT COMPARISON
====================================================================
Baseline: ragas-baseline-v1
Variant:  ragas-new-prompt-v1
====================================================================

Metric                    Baseline    Variant     Change
------------------------------------------------------------
answer_relevancy              0.850      0.920     +0.070
context_precision             0.780      0.810     +0.030
context_recall                0.720      0.750     +0.030
faithfulness                  0.890      0.910     +0.020
------------------------------------------------------------
AVERAGE                       0.810      0.848     +0.038

====================================================================
RECOMMENDATION: Variant is better (4 metrics improved)
====================================================================
```

## Step 4: Evaluation-Driven Development

Use evaluation results to guide improvements.

### Workflow

1. **Baseline**: Run initial evaluation to establish baseline metrics
2. **Hypothesize**: Identify potential improvements (prompts, retrieval, etc.)
3. **Experiment**: Make changes and run new evaluation
4. **Compare**: Use compare_experiments.py to analyze differences
5. **Iterate**: If improved, promote changes; if not, try different approach

### Common Improvements

| Low Metric | Likely Cause | Potential Fix |
|------------|--------------|---------------|
| Faithfulness | Hallucination | Improve prompt grounding instructions |
| Answer Relevancy | Off-topic answers | Refine RAG generation prompt |
| Context Precision | Irrelevant retrieval | Tune embedding model or search params |
| Context Recall | Missing information | Increase retrieval limit or add sources |

## Best Practices

### 1. Consistent Experiment Naming

Use descriptive, versioned names:

```
Good:  ragas-baseline-v1, ragas-new-prompt-v1, ragas-hybrid-search-v1
Bad:   test, experiment1, final
```

### 2. Document Changes

Include metadata when running experiments:

```python
results = evaluate(
    target=rag_pipeline,
    data="graphrag-rag-golden",
    evaluators=evaluators,
    experiment_prefix="ragas-new-prompt",
    metadata={
        "model": "gpt-4o",
        "change": "Updated RAG prompt with explicit citation instructions",
        "hypothesis": "Explicit instructions improve faithfulness",
    },
)
```

### 3. Run Multiple Iterations

For statistically significant comparisons, run multiple times:

```bash
# Run 3 times with different seeds
for i in 1 2 3; do
    uv run python scripts/run_ragas_evaluation.py \
        --experiment-name "ragas-test-run-$i"
done
```

### 4. Stratified Analysis

Use dataset metadata to identify weaknesses:

- Filter by `difficulty: hard` to find edge cases
- Filter by `domain: standards` to check specialized knowledge
- Filter by `intent: structured` to check Cypher generation

## Custom Evaluators

The RAGAS evaluators are in `evaluation/ragas_evaluators.py`. To create custom evaluators:

```python
async def my_custom_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, Any]:
    """Custom evaluator template."""
    inputs = run.inputs or {}
    outputs = run.outputs or {}

    # Your evaluation logic here
    score = calculate_my_metric(inputs, outputs)

    return {
        "key": "my_metric",
        "score": score,
        "comment": "Explanation of score",
    }
```

## Troubleshooting

### Dataset Not Found

```
ERROR - Dataset not found: graphrag-rag-golden
```

**Solution**: Run `create_golden_dataset.py` first.

### No Metrics in Comparison

```
No metrics found for this experiment.
```

**Solution**: The experiment may not have completed. Check LangSmith UI for errors.

### Low Scores Across All Metrics

Possible causes:
- Mock context in evaluation (expected for testing)
- Dataset examples too difficult
- Model mismatch between training and evaluation

## Links

- LangSmith Datasets: https://smith.langchain.com/datasets
- LangSmith Experiments: https://smith.langchain.com/experiments
- RAGAS Documentation: https://docs.ragas.io/
- LangSmith Evaluate Guide: https://docs.smith.langchain.com/evaluation

## Quick Reference

| Task | Command |
|------|---------|
| Create golden dataset | `uv run python scripts/create_golden_dataset.py` |
| Run RAGAS evaluation | `uv run python scripts/run_ragas_evaluation.py` |
| Run RAGAS (explanatory only) | `uv run python scripts/run_ragas_evaluation.py -f explanatory` |
| List experiments | `uv run python scripts/compare_experiments.py --list` |
| Compare experiments | `uv run python scripts/compare_experiments.py --compare exp1 exp2` |
| View experiment | `uv run python scripts/compare_experiments.py --show exp-name` |
