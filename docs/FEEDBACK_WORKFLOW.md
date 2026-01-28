# Human Feedback Loop Workflow

This document describes the human annotation workflow for continuous improvement of the GraphRAG evaluation system.

## Overview

The feedback loop enables systematic improvement of RAG quality through:

1. **Identification**: Automatically flag low-confidence responses for review
2. **Annotation**: Human reviewers evaluate and correct flagged responses
3. **Integration**: Verified feedback becomes new golden dataset examples
4. **Re-evaluation**: Measure improvement from feedback integration

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FEEDBACK LOOP WORKFLOW                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  RAG Query   │───▶│  Evaluation  │───▶│ Low Conf?    │          │
│  │  Execution   │    │   Metrics    │    │ < 0.7        │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
│                                                  │                  │
│                                          ┌──────▼───────┐          │
│                                          │   Export to  │          │
│                                          │  Annotation  │          │
│                                          │    Queue     │          │
│                                          └──────┬───────┘          │
│                                                  │                  │
│                                          ┌──────▼───────┐          │
│                                          │    Human     │          │
│                                          │  Annotation  │          │
│                                          └──────┬───────┘          │
│                                                  │                  │
│                                          ┌──────▼───────┐          │
│                                          │   Import     │          │
│                                          │  Feedback    │          │
│                                          └──────┬───────┘          │
│                                                  │                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐          │
│  │ Re-evaluate  │◀───│   Update     │◀───│  Generate    │          │
│  │  with New    │    │   Golden     │    │   Examples   │          │
│  │   Dataset    │    │   Dataset    │    │              │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Export Low-Confidence Runs

```bash
# Export runs with confidence below 0.7 from the last 7 days
python scripts/export_for_annotation.py \
    --project graphrag-api-dev \
    --threshold 0.7 \
    --days 7 \
    --output data/annotations/pending.json
```

### 2. Annotate (Manual Step)

Review the exported `pending.json` and add feedback:

```json
{
  "run_id": "abc123",
  "question": "What is requirements traceability?",
  "original_answer": "...",
  "is_correct": false,
  "corrected_answer": "Requirements traceability is the ability to...",
  "quality_score": 3,
  "feedback_notes": "Missing reference to ISO 26262 requirement",
  "missing_information": ["ISO 26262 compliance"],
  "factual_errors": [],
  "annotator_id": "reviewer@example.com"
}
```

### 3. Import Feedback

```bash
# Import annotated feedback and generate new examples
python scripts/import_feedback.py \
    --input data/annotations/completed.json \
    --output data/feedback/new_examples.json \
    --generate-examples
```

### 4. Update Golden Dataset

```bash
# Merge new examples into the golden dataset
python scripts/update_datasets.py \
    --examples data/feedback/new_examples.json \
    --output data/datasets/updated_golden_dataset.json \
    --min-confidence 0.9
```

### 5. Re-Run Evaluation

```bash
# Run evaluation with updated dataset
python scripts/run_evaluation.py --dataset data/datasets/updated_golden_dataset.json
```

## Scripts Reference

### export_for_annotation.py

Exports low-confidence evaluation runs for human review.

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `data/annotations/pending.json` | Output file path |
| `--project`, `-p` | `graphrag-api-dev` | LangSmith project name |
| `--threshold`, `-t` | `0.7` | Confidence threshold |
| `--max-runs`, `-m` | `100` | Maximum runs to export |
| `--days`, `-d` | `7` | Days to look back |
| `--queue` | `false` | Also add to LangSmith annotation queue |
| `--dry-run` | `false` | Preview without writing |

**Confidence Calculation:**
- Faithfulness: 30% weight
- Answer Relevancy: 30% weight
- Context Precision: 20% weight
- Context Recall: 20% weight

### import_feedback.py

Imports annotated feedback and generates evaluation examples.

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | Required | Input JSON file |
| `--from-langsmith` | `false` | Import from LangSmith instead |
| `--project`, `-p` | `graphrag-api-dev` | LangSmith project |
| `--output`, `-o` | `data/feedback/new_examples.json` | Output file |
| `--generate-examples` | `true` | Generate evaluation examples |
| `--validate-only` | `false` | Only validate, don't import |

**Output Statistics:**
- Total annotations processed
- Correct/incorrect answer counts
- Average quality score
- Model accuracy rate

### update_datasets.py

Merges verified feedback into the golden dataset.

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--examples`, `-e` | Required | New examples JSON file |
| `--output`, `-o` | `data/datasets/updated_golden_dataset.json` | Output file |
| `--python-output` | None | Also output as Python module |
| `--min-confidence` | `0.8` | Minimum confidence threshold |
| `--similarity-threshold` | `0.85` | Duplicate detection threshold |
| `--dry-run` | `false` | Preview without writing |

**Auto-Enrichment:**
- Infers query category from question patterns
- Infers difficulty from question/answer complexity
- Extracts standard references (ISO, IEC, DO-178C, etc.)
- Extracts domain entities

## Annotation Guidelines

### Quality Score Scale

| Score | Meaning | Action |
|-------|---------|--------|
| 5 | Excellent | Use as golden example |
| 4 | Good | Minor improvements possible |
| 3 | Acceptable | Needs some corrections |
| 2 | Poor | Significant corrections needed |
| 1 | Incorrect | Completely wrong, rewrite |

### What to Annotate

**Mark as CORRECT if:**
- Answer is factually accurate
- Covers key points of the question
- Citations/references are accurate
- Appropriate level of detail

**Mark as INCORRECT if:**
- Contains factual errors
- Missing critical information
- Cites wrong standards/references
- Significantly off-topic

### Providing Corrections

When marking as incorrect, provide:

1. **Corrected Answer**: Full corrected response
2. **Missing Information**: List of topics not covered
3. **Factual Errors**: Specific errors identified
4. **Feedback Notes**: General observations

### Example Annotation

```json
{
  "run_id": "run_12345",
  "question": "What are the ASIL levels in ISO 26262?",
  "original_answer": "ASIL levels are A, B, C, D used for safety classification.",
  "is_correct": false,
  "corrected_answer": "ISO 26262 defines four ASIL (Automotive Safety Integrity Level) classifications: ASIL A (lowest), ASIL B, ASIL C, and ASIL D (highest/most stringent). There is also QM (Quality Management) for non-safety-relevant functions. ASIL is determined based on severity, exposure probability, and controllability of potential hazards.",
  "quality_score": 2,
  "feedback_notes": "Answer was too brief and missing key details about QM and ASIL determination factors",
  "missing_information": [
    "QM (Quality Management) level",
    "ASIL determination factors (severity, exposure, controllability)",
    "Explanation of stringency levels"
  ],
  "factual_errors": [],
  "annotator_id": "safety_expert_1"
}
```

## LangSmith Integration

### Setting Up Annotation Queues

1. Enable LangSmith annotation queues in your project settings
2. Configure the `--queue` flag in export script
3. Annotators can review runs directly in LangSmith UI

### Importing from LangSmith

```bash
# Import feedback directly from LangSmith
python scripts/import_feedback.py \
    --from-langsmith \
    --project graphrag-api-dev \
    --days 14
```

## Data Directory Structure

```
data/
├── annotations/
│   ├── pending.json          # Runs awaiting annotation
│   └── completed.json        # Annotated runs (manual)
├── feedback/
│   ├── new_examples.json     # Generated examples from feedback
│   └── import_stats.json     # Import statistics
└── datasets/
    ├── updated_golden_dataset.json  # Updated dataset (JSON)
    └── updated_golden_dataset.py    # Updated dataset (Python)
```

## Best Practices

### Annotation Quality

1. **Consistency**: Use the quality score scale consistently
2. **Completeness**: Always provide corrected answers for incorrect responses
3. **Specificity**: List specific missing information and errors
4. **Domain Expertise**: Route technical questions to subject matter experts

### Feedback Loop Cadence

| Environment | Frequency | Scope |
|-------------|-----------|-------|
| Development | Daily | All low-confidence runs |
| Staging | Weekly | Sample of production traffic |
| Production | Bi-weekly | Flagged runs from user feedback |

### Measuring Improvement

Track these metrics over feedback cycles:

1. **Model Accuracy Rate**: Percentage of correct answers
2. **Average Quality Score**: Mean quality score from annotations
3. **Golden Dataset Growth**: Number of verified examples
4. **Re-evaluation Scores**: RAGAS metrics on updated dataset

## Troubleshooting

### No Runs Exported

- Verify LangSmith API key is set: `echo $LANGSMITH_API_KEY`
- Check project name matches: `--project graphrag-api-dev`
- Lower threshold: `--threshold 0.5`
- Extend date range: `--days 30`

### Import Validation Errors

- Ensure `is_correct: false` has `corrected_answer`
- Check `quality_score` is 1-5
- Verify JSON syntax

### Duplicate Detection

- Adjust similarity threshold: `--similarity-threshold 0.9`
- Review detected duplicates in verbose mode: `-v`

## Related Documentation

- [Evaluation Framework](../CLAUDE_CODE_HANDOFF.md#phase-2-custom-domain-metrics)
- [Platform Comparison](./PLATFORM_COMPARISON.md)
- [Production Monitoring](./PRODUCTION_MONITORING.md)
