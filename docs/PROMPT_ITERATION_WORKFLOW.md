# Prompt Iteration Workflow

This guide explains how to iterate on prompts using LangSmith for version control, testing, and promotion.

## Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. PLAYGROUND  │────▶│  2. CREATE      │────▶│  3. A/B TEST    │────▶│  4. PROMOTE     │
│  Test changes   │     │  VARIANT        │     │  Compare scores │     │  Tag as prod    │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Step 1: Test in LangSmith Playground

The Playground lets you interactively test prompt changes before creating a variant.

### Access the Playground

1. Go to https://smith.langchain.com/prompts
2. Click on a prompt (e.g., `graphrag-router`)
3. Click the **"Playground"** button

### Test Your Changes

1. **Modify the prompt template** in the editor
2. **Fill in test inputs** (e.g., `tools`, `question`)
3. **Click "Run"** to see the output
4. **Compare models** - test with GPT-4, Claude, etc.
5. **Iterate** until you're satisfied with the output

### Tips for Playground Testing

- Test edge cases (very short questions, multi-part questions)
- Try different phrasings of the same question
- Check that JSON/Cypher output is valid
- Verify the reasoning in the response makes sense

## Step 2: Create a Prompt Variant

Once you've refined a prompt in the Playground, save it as a variant.

### Option A: Save from Playground

1. After testing, click **"Save as"** in the Playground
2. Name it with a version suffix: `graphrag-router-v2`
3. Add a description of what changed

### Option B: Push via CLI

```bash
# Modify the prompt in definitions.py, then push with a new name
uv run python -c "
from langsmith import Client
from jama_mcp_server_graphrag.prompts.definitions import PROMPT_DEFINITIONS, PromptName

client = Client()
definition = PROMPT_DEFINITIONS[PromptName.ROUTER]

# Push as variant
client.push_prompt(
    'graphrag-router-v2',
    object=definition.template,
    description='Router v2: Improved tool selection for complex queries',
    tags=['variant', 'testing'],
)
"
```

### Naming Convention

| Pattern | Example | Use Case |
|---------|---------|----------|
| `{name}-v2` | `graphrag-router-v2` | Simple version increment |
| `{name}-{date}` | `graphrag-router-20240119` | Date-based versioning |
| `{name}-{experiment}` | `graphrag-router-detailed-reasoning` | Experiment name |

## Step 3: Run A/B Comparison

Use the comparison script to evaluate variants against the baseline.

### Basic Comparison

```bash
# Compare router baseline vs v2 variant
uv run python scripts/run_prompt_comparison.py router --variant v2
```

### Multiple Iterations (More Reliable)

```bash
# Run 3 iterations for statistical confidence
uv run python scripts/run_prompt_comparison.py router --variant v2 --iterations 3
```

### Save Report to File

```bash
uv run python scripts/run_prompt_comparison.py router --variant v2 -o comparison_report.json
```

### Interpreting Results

```
--- SCORES ---
Metric                    Baseline    Variant     Change
-------------------------------------------------------
json_valid                   0.857      0.929     +0.072
length_appropriate           1.000      1.000     =0.000

--- RESULT ---
Winner: VARIANT
Recommendation: PROMOTE: Variant shows improvement in 1 metrics
```

| Indicator | Meaning |
|-----------|---------|
| `+` | Variant is better |
| `-` | Baseline is better |
| `=` | No significant difference |

## Step 4: Promote Winning Prompts

When a variant wins, promote it to production.

### Option A: Tag in LangSmith UI

1. Go to the winning prompt in LangSmith Hub
2. Click **"Tags"** or **"Versions"**
3. Add tag: `production`
4. Optionally add: `staging` for pre-production testing

### Option B: Programmatic Promotion

```python
from langsmith import Client

client = Client()

# Pull the winning variant
winning_prompt = client.pull_prompt("graphrag-router-v2")

# Push as the new baseline with production tag
client.push_prompt(
    "graphrag-router:production",
    object=winning_prompt,
    description="Promoted from v2 after A/B testing",
    tags=["production", "promoted"],
)
```

### Update Environment

To use production prompts:

```bash
# In .env
PROMPT_ENVIRONMENT=production
```

The catalog will automatically pull `prompt-name:production` tagged versions.

## Best Practices

### 1. Always Test Before Promoting

- Run at least 3 iterations of A/B testing
- Check both aggregate scores AND individual examples
- Review any failures manually

### 2. Keep Baseline Stable

- Don't modify the baseline while testing variants
- Create a new variant for each change
- Document what changed in each variant

### 3. Use Descriptive Names

```
Good:  graphrag-router-detailed-reasoning-v1
Bad:   graphrag-router-test
```

### 4. Track Changes

Add descriptions when pushing:

```python
client.push_prompt(
    "graphrag-router-v2",
    object=template,
    description="""
    Changes from v1:
    - Added explicit reasoning step
    - Improved handling of multi-part questions
    - Fixed JSON formatting issues
    """,
)
```

### 5. Gradual Rollout

1. Test in `development` environment
2. Promote to `staging` for integration testing
3. Finally promote to `production`

## Troubleshooting

### Variant Not Found

```
ERROR - Variant prompt not found: graphrag-router-v2
```

**Solution**: Create the variant first using Playground or CLI push.

### No Evaluation Dataset

```
ERROR - No evaluation dataset for stepback
```

**Solution**: Create a dataset using `scripts/create_eval_datasets.py`.

### Low Scores

If both baseline and variant have low scores:
- Review the evaluation dataset examples
- Check that expected outputs are realistic
- Ensure the prompt has the right input variables

## Quick Reference

| Task | Command |
|------|---------|
| List available prompts | `uv run python scripts/run_prompt_comparison.py --list` |
| Compare prompts | `uv run python scripts/run_prompt_comparison.py router --variant v2` |
| Create datasets | `uv run python scripts/create_eval_datasets.py` |
| Push prompts | `uv run python -m jama_mcp_server_graphrag.prompts.cli push --all --personal` |
| List prompts | `uv run python -m jama_mcp_server_graphrag.prompts.cli list` |

## Links

- LangSmith Prompts: https://smith.langchain.com/prompts
- LangSmith Datasets: https://smith.langchain.com/datasets
- Organization: https://smith.langchain.com/o/33c08fbf-3b88-4b49-ae32-9677043ebed2
