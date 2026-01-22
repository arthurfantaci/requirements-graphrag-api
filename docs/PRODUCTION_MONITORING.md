# Production Monitoring Guide

This guide covers setting up production monitoring for the GraphRAG system using LangSmith.

## Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  TRACES         │────▶│  ALERTS         │────▶│  ANNOTATION     │
│  All LLM calls  │     │  Latency/Errors │     │  Human review   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 1. Production Project Setup

### Create Separate Projects

Create distinct projects for each environment:

| Project | Purpose | Environment Variable |
|---------|---------|---------------------|
| `requirements-graphrag-dev` | Development testing | `LANGSMITH_PROJECT=requirements-graphrag-dev` |
| `requirements-graphrag-staging` | Pre-production | `LANGSMITH_PROJECT=requirements-graphrag-staging` |
| `requirements-graphrag-prod` | Production | `LANGSMITH_PROJECT=requirements-graphrag-prod` |

### Environment Configuration

```bash
# Production .env
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=requirements-graphrag-prod
LANGSMITH_API_KEY=<org-scoped-key>
LANGSMITH_WORKSPACE_ID=33c08fbf-3b88-4b49-ae32-9677043ebed2
PROMPT_ENVIRONMENT=production
```

## 2. Alert Configuration

### Access Alerts

1. Go to: https://smith.langchain.com/o/33c08fbf-3b88-4b49-ae32-9677043ebed2/settings/alerts
2. Or: Settings → Alerts in your organization

### Recommended Alerts

| Alert | Condition | Threshold | Action |
|-------|-----------|-----------|--------|
| High Latency | P95 latency | > 5 seconds | Investigate slow queries |
| Error Rate | Error percentage | > 1% | Check for prompt/API issues |
| Token Spike | Token usage | > 2x baseline | Check for runaway queries |
| Low Success | Success rate | < 95% | Review failed runs |

### Creating an Alert (UI)

1. Click **"Create Alert"**
2. Select project: `requirements-graphrag-prod`
3. Configure condition:
   - Metric: `latency_p95`
   - Operator: `greater_than`
   - Value: `5000` (ms)
4. Set notification (email/Slack)
5. Save

## 3. Monitoring Dashboard

### Key Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| **Latency P50** | Median response time | < 2s |
| **Latency P95** | 95th percentile | < 5s |
| **Error Rate** | Failed runs / total | < 1% |
| **Token Usage** | Avg tokens per run | Monitor trends |
| **Success Rate** | Successful completions | > 99% |

### Viewing Metrics

1. Go to your project in LangSmith
2. Click **"Analytics"** tab
3. Set time range (last 24h, 7d, 30d)
4. View charts for latency, throughput, errors

### Custom Filters

Filter runs by:
- Run type: `chain`, `llm`, `retriever`
- Status: `success`, `error`
- Tags: `production`, `high-priority`
- Latency: `> 5000ms`

## 4. Annotation Queues

Annotation queues enable human review of LLM outputs.

### Use Cases

| Queue | Purpose | Filter Criteria |
|-------|---------|-----------------|
| Low Confidence | Review uncertain answers | Critic confidence < 0.5 |
| Failed Routing | Check routing decisions | Router errors |
| User Reported | Investigate complaints | Tagged `user-reported` |
| Random Sample | Quality assurance | Random 5% of runs |

### Creating an Annotation Queue (UI)

1. Go to: Annotation Queues in LangSmith
2. Click **"Create Queue"**
3. Configure:
   - Name: `low-confidence-review`
   - Project: `requirements-graphrag-prod`
   - Filter: `metadata.confidence < 0.5`
4. Assign reviewers
5. Save

### Adding Runs to Queue (Programmatic)

```python
from langsmith import Client

client = Client()

# Add specific runs to annotation queue
client.create_annotation_queue_run(
    queue_name="low-confidence-review",
    run_id="<run-id>",
)
```

## 5. Regression Testing

### Export Production Traces as Dataset

```python
from langsmith import Client

client = Client()

# Get successful production runs
runs = client.list_runs(
    project_name="requirements-graphrag-prod",
    filter='eq(status, "success")',
    limit=100,
)

# Create regression dataset
run_ids = [run.id for run in runs]
client.create_dataset_from_runs(
    run_ids=run_ids,
    dataset_name="prod-regression-2024-01",
)
```

### Automated Regression Testing

Run before each deployment:

```bash
# Test new prompt versions against production traces
uv run python scripts/run_prompt_comparison.py router \
    --variant v2 \
    --dataset prod-regression-2024-01
```

## 6. Quick Reference

### LangSmith URLs

| Resource | URL |
|----------|-----|
| Organization | https://smith.langchain.com/o/33c08fbf-3b88-4b49-ae32-9677043ebed2 |
| Projects | https://smith.langchain.com/o/33c08fbf-3b88-4b49-ae32-9677043ebed2/projects |
| Datasets | https://smith.langchain.com/o/33c08fbf-3b88-4b49-ae32-9677043ebed2/datasets |
| Prompts | https://smith.langchain.com/prompts?organizationId=33c08fbf-3b88-4b49-ae32-9677043ebed2 |

### Environment Variables

```bash
# Required for production
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your-org-api-key>
LANGSMITH_PROJECT=requirements-graphrag-prod
LANGSMITH_WORKSPACE_ID=33c08fbf-3b88-4b49-ae32-9677043ebed2
PROMPT_ENVIRONMENT=production
```

### CLI Commands

```bash
# View tracing status
uv run python -c "from requirements_graphrag_api.observability import get_tracing_status; print(get_tracing_status())"

# List prompts
uv run python -m requirements_graphrag_api.prompts.cli list

# Validate prompts
uv run python -m requirements_graphrag_api.prompts.cli validate
```

## 7. Troubleshooting

### No Traces Appearing

1. Check `LANGSMITH_TRACING=true`
2. Verify API key is valid
3. Confirm workspace ID is set
4. Check project name matches

### High Error Rates

1. Check LangSmith for error details
2. Review recent prompt changes
3. Verify API keys (OpenAI, Neo4j)
4. Check rate limits

### Performance Degradation

1. Review P95 latency trends
2. Check Neo4j query times
3. Monitor OpenAI API latency
4. Review recent code changes
