# Claude Code Handoff: Prompt Catalog + LangSmith Integration

## Project Context
This is the `jama-mcp-server-graphrag` project - a GraphRAG MCP Server for Requirements Management. We're implementing a centralized Prompt Catalog system with **full LangSmith integration** for prompt versioning, evaluation, tracing, and production monitoring.

## Learning Objectives (From Project Instructions)
- **Prompt Engineering & Optimization**: Fine-tune prompts, iterate on designs, leverage LangSmith Hub for version control
- **Evaluation & Quality Assurance**: Build evaluation pipelines, benchmark agent behavior with custom metrics
- **Production Operations**: Implement comprehensive tracing, monitoring, and observability for LLM applications

## Current State

### ✅ Completed (Files Created)
```
src/jama_mcp_server_graphrag/prompts/
├── __init__.py          # Package exports
├── definitions.py       # 6 prompt templates with metadata
├── catalog.py           # LangSmith Hub integration + caching
├── cli.py               # CLI for push/pull/list/validate
└── evaluation.py        # Evaluators + A/B testing

tests/
└── test_prompts.py      # 35+ test cases
```

### ✅ Already Updated (Dependent Modules)
These modules already import from the new prompts catalog:
- `src/jama_mcp_server_graphrag/agentic/router.py`
- `src/jama_mcp_server_graphrag/agentic/critic.py`
- `src/jama_mcp_server_graphrag/agentic/stepback.py`
- `src/jama_mcp_server_graphrag/agentic/query_updater.py`
- `src/jama_mcp_server_graphrag/core/generation.py`
- `src/jama_mcp_server_graphrag/core/text2cypher.py`

---

## Remaining Tasks

### Phase 1: Verify Local Implementation

#### 1.1 Run and Fix Tests
```bash
uv run pytest tests/test_prompts.py -v --tb=short
```
- Fix any import errors or test failures
- Ensure all tests pass

#### 1.2 Linting and Formatting
```bash
uv run ruff check src/jama_mcp_server_graphrag/prompts/ --fix
uv run ruff format src/jama_mcp_server_graphrag/prompts/
```

#### 1.3 Validate CLI
```bash
uv run python -m jama_mcp_server_graphrag.prompts.cli list
uv run python -m jama_mcp_server_graphrag.prompts.cli validate
```

#### 1.4 Run Full Test Suite
```bash
uv run pytest -v --tb=short
```

---

### Phase 2: LangSmith Hub Setup (Prompt Versioning)

#### 2.1 Configure LangSmith Environment
Create or update `.env` with LangSmith credentials:
```bash
# LangSmith Configuration
LANGSMITH_API_KEY=<your-langsmith-api-key>
LANGSMITH_PROJECT=jama-graphrag
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_ORG=<your-organization-name>

# Prompt Catalog Settings
PROMPT_ENVIRONMENT=development
```

#### 2.2 Push All Prompts to LangSmith Hub
```bash
# Push all 6 prompts to Hub
uv run python -m jama_mcp_server_graphrag.prompts.cli push --all
```

This will create the following prompts in your Hub:
- `{org}/graphrag-router`
- `{org}/graphrag-critic`
- `{org}/graphrag-stepback`
- `{org}/graphrag-query-updater`
- `{org}/graphrag-rag-generation`
- `{org}/graphrag-text2cypher`

#### 2.3 Verify Prompts in LangSmith Dashboard
1. Navigate to https://smith.langchain.com
2. Go to Hub → Your Organization
3. Verify all 6 prompts are visible with:
   - Correct descriptions
   - Input variables
   - Tags (routing, agentic, rag, etc.)

#### 2.4 Create Environment Tags
In LangSmith Hub, tag prompts for different environments:
- `development` - Active development/testing
- `staging` - Pre-production validation
- `production` - Production-ready versions

---

### Phase 3: LangSmith Tracing Integration

#### 3.1 Verify Tracing Configuration
Check `src/jama_mcp_server_graphrag/observability.py` has LangSmith tracing enabled:
```python
from langsmith import traceable

# Ensure LANGSMITH_TRACING=true in environment
```

#### 3.2 Update Config for Tracing
Add to `src/jama_mcp_server_graphrag/config.py`:
```python
# LangSmith Settings
langsmith_api_key: str | None = None
langsmith_project: str = "jama-graphrag"
langsmith_tracing: bool = True

# Prompt Catalog Settings
langsmith_org: str = "jama-graphrag"
prompt_environment: str = "development"
prompt_cache_ttl: int = 300
```

#### 3.3 Add Prompt Version to Traces
Update the catalog to log prompt source/version in traces. In `catalog.py`, ensure traces include:
- Prompt name
- Prompt version
- Source (hub vs local)
- Environment

---

### Phase 4: Evaluation Datasets

#### 4.1 Create Evaluation Datasets in LangSmith
Create datasets for each prompt type with representative examples:

**Router Evaluation Dataset** (`router-eval-dataset`):
```json
{
  "inputs": {"tools": "...", "question": "What is requirements traceability?"},
  "outputs": {"expected_tools": ["graphrag_vector_search"]}
}
```

**Critic Evaluation Dataset** (`critic-eval-dataset`):
```json
{
  "inputs": {"context": "...", "question": "..."},
  "outputs": {"expected_answerable": true, "expected_completeness": "complete"}
}
```

**Text2Cypher Evaluation Dataset** (`text2cypher-eval-dataset`):
```json
{
  "inputs": {"schema": "...", "examples": "...", "question": "How many articles?"},
  "outputs": {"expected_cypher": "MATCH (a:Article) RETURN count(a)"}
}
```

#### 4.2 Script to Create Datasets
Create `scripts/create_eval_datasets.py`:
```python
"""Create evaluation datasets in LangSmith."""
from langsmith import Client

client = Client()

# Router dataset
router_examples = [
    {
        "inputs": {
            "tools": "- graphrag_vector_search: Basic semantic search...",
            "question": "What is requirements traceability?"
        },
        "outputs": {
            "expected_tools": ["graphrag_vector_search"],
            "expected_reasoning": "General concept lookup"
        }
    },
    # Add 10-20 more examples covering different routing scenarios
]

dataset = client.create_dataset("router-eval-dataset")
for example in router_examples:
    client.create_example(
        inputs=example["inputs"],
        outputs=example["outputs"],
        dataset_id=dataset.id
    )
```

#### 4.3 Run Evaluations
```bash
# Evaluate router prompt against dataset
uv run python -c "
import asyncio
from jama_mcp_server_graphrag.prompts.evaluation import evaluate_prompt
from jama_mcp_server_graphrag.prompts import PromptName

result = asyncio.run(evaluate_prompt(PromptName.ROUTER, 'router-eval-dataset'))
print(f'Scores: {result.scores}')
"
```

---

### Phase 5: Prompt Iteration Workflow

#### 5.1 Test Prompts in LangSmith Playground
1. Go to LangSmith Hub → Select a prompt
2. Click "Playground"
3. Test with different inputs
4. Compare model outputs (GPT-4, Claude, etc.)

#### 5.2 Create Prompt Variants
For A/B testing, create variants in Hub:
- `graphrag-router` (baseline v1.0.0)
- `graphrag-router-v2` (candidate with improvements)

#### 5.3 Run A/B Comparison
```python
from jama_mcp_server_graphrag.prompts.evaluation import compare_prompts
from jama_mcp_server_graphrag.prompts import PromptName

result = await compare_prompts(
    baseline=PromptName.ROUTER,
    candidate="your-org/graphrag-router-v2",  # Hub path
    dataset_name="router-eval-dataset"
)
print(f"Winner: {result.winner}")
print(f"Improvements: {result.improvements}")
```

#### 5.4 Promote Winning Prompts
When a variant wins:
1. Tag it as `production` in Hub
2. Update `PROMPT_ENVIRONMENT=production` in prod config
3. The catalog will automatically pull the production-tagged version

---

### Phase 6: Production Monitoring

#### 6.1 Set Up Monitoring Dashboard
In LangSmith:
1. Create a project for production traces
2. Set up alerts for:
   - High latency (> 5s)
   - Error rate (> 1%)
   - Token usage anomalies

#### 6.2 Export Traces for Regression Testing
```python
from langsmith import Client

client = Client()

# Export recent production traces as evaluation dataset
runs = client.list_runs(
    project_name="jama-graphrag-prod",
    filter='eq(status, "success")',
    limit=100
)

# Convert to dataset for regression testing
client.create_dataset_from_runs(
    runs=[r.id for r in runs],
    dataset_name="prod-regression-dataset"
)
```

#### 6.3 Annotation Queues for Human Feedback
Set up annotation queues for:
- Low-confidence answers (from critic)
- Failed routing decisions
- User-reported issues

---

## LangSmith Integration Checklist

| Task | Status | Notes |
|------|--------|-------|
| Push prompts to Hub | ✅ | All 6 prompts |
| Verify prompts in dashboard | ✅ | Check descriptions, variables |
| Create environment tags | ✅ | dev/staging/prod |
| Enable tracing in config | ✅ | LANGSMITH_TRACING=true |
| Create router eval dataset | ✅ | 14 examples |
| Create critic eval dataset | ✅ | 10 examples |
| Create text2cypher eval dataset | ✅ | 11 examples |
| Run baseline evaluations | ✅ | Record initial scores |
| Test Playground workflow | ✅ | Interactive prompt testing |
| Set up production project | ✅ | Separate from dev |
| Configure monitoring alerts | ✅ | Latency, errors |

---

## Key Files Reference

### For Prompt Hub Operations
- `prompts/cli.py` - Push/pull/list commands
- `prompts/catalog.py` - Hub integration logic

### For Evaluation
- `prompts/evaluation.py` - Evaluators and A/B testing
- `prompts/definitions.py` - Evaluation criteria per prompt

### For Tracing
- `observability.py` - @traceable decorator usage
- All agentic/*.py files - Already instrumented

---

## Success Criteria

### Phase 1 (Local)
- [ ] All `test_prompts.py` tests pass
- [ ] Full test suite passes
- [ ] CLI commands work

### Phase 2 (Hub)
- [ ] All 6 prompts visible in LangSmith Hub
- [ ] Prompts have correct metadata and tags
- [ ] Can pull prompts from Hub in code

### Phase 3 (Tracing)
- [ ] Traces appear in LangSmith dashboard
- [ ] Traces include prompt version/source

### Phase 4 (Evaluation)
- [ ] 3+ evaluation datasets created
- [ ] Baseline evaluation scores recorded

### Phase 5 (Iteration)
- [ ] Successfully ran A/B test between variants
- [ ] Documented prompt improvement workflow

### Phase 6 (Production)
- [ ] Production project configured
- [ ] Monitoring alerts set up
