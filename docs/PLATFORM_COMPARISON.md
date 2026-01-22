# LangSmith vs MLflow: Platform Comparison

This document provides a comprehensive comparison of LangSmith and MLflow for LLM observability and experiment tracking, specifically in the context of GraphRAG evaluation.

## Executive Summary

| Aspect | LangSmith | MLflow |
|--------|-----------|--------|
| **Best For** | LangChain-native teams needing quick setup | Teams requiring self-hosting or open-source |
| **Setup Time** | 5 minutes | 30+ minutes |
| **Cost** | Usage-based (free tier available) | Free (infra costs for hosting) |
| **Self-Hosting** | ❌ No | ✅ Yes |
| **LangChain Integration** | ✅ Native auto-tracing | ⚠️ Manual instrumentation |

## 1. Setup Complexity

### LangSmith
```bash
# Total setup: ~5 minutes
export LANGSMITH_API_KEY=<your-key>
export LANGSMITH_PROJECT=requirements-graphrag
export LANGSMITH_TRACING=true
# Done! LangChain auto-traces everything.
```

**Pros:**
- Zero-config for LangChain applications
- Managed infrastructure
- No server maintenance

**Cons:**
- Cloud-only (no self-hosting)
- Requires internet connectivity
- Data leaves your network

### MLflow
```bash
# Total setup: ~30 minutes (basic)
pip install mlflow
mlflow server --host 0.0.0.0 --port 5000
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow experiments create -n requirements-graphrag
# Then: manual instrumentation in code
```

**Pros:**
- Full control over infrastructure
- Data stays on-premises
- No vendor lock-in

**Cons:**
- Requires server setup and maintenance
- Manual instrumentation required
- More operational overhead

## 2. Evaluation Features

### Feature Matrix

| Feature | LangSmith | MLflow |
|---------|-----------|--------|
| Auto-tracing (LangChain) | ✅ Native | ❌ Manual |
| Custom evaluators | ✅ Yes | ✅ Yes |
| Dataset management | ✅ Built-in | ✅ MLflow Datasets |
| A/B testing | ✅ Prompt comparison UI | ✅ Experiment comparison |
| Trace visualization | ✅ Rich tree view | ❌ Basic logs |
| Token counting | ✅ Automatic | ❌ Manual |
| Cost tracking | ✅ Built-in | ❌ Manual |

### LangSmith Evaluation Example
```python
from langsmith import evaluate

results = evaluate(
    lambda inputs: rag_chain.invoke(inputs),
    data="golden-dataset",
    evaluators=[faithfulness, relevancy],
)
```

### MLflow Evaluation Example
```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({"model": "gpt-4o", "k": 6})

    # Manual evaluation loop
    for sample in dataset:
        result = rag_chain.invoke(sample)
        metrics = compute_metrics(result, sample)
        mlflow.log_metrics(metrics)
```

## 3. Visualization Capabilities

### LangSmith
- **Trace Tree View**: Hierarchical visualization of chain execution
- **Token Breakdown**: Per-step token usage and costs
- **Latency Analysis**: Waterfall charts for performance
- **Feedback Integration**: Human annotation support
- **Comparison View**: Side-by-side prompt comparison

### MLflow
- **Experiment Dashboard**: Metric comparison across runs
- **Parameter Search**: Filter runs by parameters
- **Artifact Browser**: View logged files and models
- **Custom Charts**: Via MLflow UI plugins
- **Model Registry**: Version and stage models

## 4. Prompt Versioning

### LangSmith (LangChainHub)
```python
from langchain import hub

# Pull versioned prompts
prompt = hub.pull("jama/rag-qa:v2")

# Push new versions
hub.push("jama/rag-qa", prompt, new_repo_is_public=False)
```

**Features:**
- Centralized prompt repository
- Version history
- Environment-based deployment (dev/staging/prod)
- Playground testing with datasets

### MLflow
```python
import mlflow

# Log prompt as artifact
with mlflow.start_run():
    mlflow.log_text(prompt_template, "prompts/rag_qa.txt")
    mlflow.log_param("prompt_version", "v2")
```

**Features:**
- Store prompts as artifacts
- Version via run tracking
- No dedicated prompt management UI

## 5. Self-Hosting Options

### LangSmith
- **Status**: Cloud-only (as of 2024)
- **Data Residency**: US-based servers
- **Compliance**: SOC 2 Type II, GDPR
- **Enterprise**: Custom data handling available

### MLflow
- **Docker Deployment**:
```bash
docker run -p 5000:5000 \
  -v /data/mlflow:/mlflow \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server --host 0.0.0.0
```

- **Kubernetes Deployment**: Helm charts available
- **Database Backends**: PostgreSQL, MySQL, SQLite
- **Artifact Stores**: S3, GCS, Azure Blob, local filesystem

## Cost Analysis

### LangSmith Pricing (2024)

| Tier | Traces/Month | Price |
|------|--------------|-------|
| Free | 5,000 | $0 |
| Developer | 50,000 | $39/month |
| Team | 500,000 | $400/month |
| Enterprise | Custom | Contact sales |

### MLflow Self-Hosted Costs

| Scale | Infrastructure | Est. Monthly Cost |
|-------|----------------|-------------------|
| Dev | Single VM | $50-100 |
| Team | Small cluster | $200-500 |
| Enterprise | HA deployment | $1,000+ |

**Note**: MLflow software is free (Apache 2.0). Costs are for infrastructure only.

## Decision Framework

### Choose LangSmith if:
- ✅ Using LangChain/LangGraph natively
- ✅ Need quick setup (< 1 hour)
- ✅ Want managed infrastructure
- ✅ Need trace visualization
- ✅ Require prompt versioning
- ✅ Budget for SaaS tooling

### Choose MLflow if:
- ✅ Require self-hosting (compliance, security)
- ✅ Need data to stay on-premises
- ✅ Have DevOps capacity for hosting
- ✅ Want open-source solution
- ✅ Already using MLflow for ML experiments
- ✅ Budget-constrained (infrastructure only)

### Use Both if:
- ✅ Migrating from one to another
- ✅ Need comparison data for decision
- ✅ Different teams have different needs

## Usage in This Project

This project provides a unified interface for both platforms:

```python
from requirements_graphrag_api.observability_comparison import (
    UnifiedTracker,
    Platform,
    recommend_platform,
)

# Get recommendation based on your needs
recommendation = recommend_platform(
    needs_self_hosting=False,
    needs_trace_visualization=True,
    langchain_native=True,
)
print(f"Recommended: {recommendation.recommended}")

# Track to both platforms simultaneously
with UnifiedTracker([Platform.LANGSMITH, Platform.MLFLOW]) as tracker:
    tracker.log_metrics({"faithfulness": 0.85})
```

## Running the Comparison

```bash
# Feature comparison
uv run python scripts/compare_platforms.py --features

# Full comparison with recommendations
uv run python scripts/compare_platforms.py --full

# Get personalized recommendation
uv run python scripts/compare_platforms.py --recommend --self-hosting

# Generate JSON report
uv run python scripts/compare_platforms.py --report comparison_report.json
```

## Conclusion

Both platforms serve valid use cases:

- **LangSmith** excels at LangChain-native observability with minimal setup
- **MLflow** excels at self-hosted, open-source experiment tracking

For this GraphRAG project, **LangSmith** is the default choice due to native LangChain integration, but MLflow support is provided for teams with self-hosting requirements.
