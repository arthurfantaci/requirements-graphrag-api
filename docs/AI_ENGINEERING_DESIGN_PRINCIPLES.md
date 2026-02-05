# AI Engineering Design Principles

## The Guardrail Stack: Defense in Depth for AI Applications

### Principle: Guardrails are the entire system, not just code gates

When building AI-powered applications, safety and quality are not achieved by a
single layer of input validation. They emerge from the **composition of
deterministic code, AI classification, prompt engineering, and output filtering**
working together.

### The Four Layers

| Layer | Type | Latency | Catches |
|-------|------|---------|---------|
| **1. Deterministic** | Keywords, regex, input validation | ~1ms | Obvious abuse, known-bad patterns |
| **2. AI Classification** | Toxicity APIs, topic LLM, embedding similarity | 200-1000ms | Subtle violations, ambiguous inputs |
| **3. Prompt Engineering** | System prompts, few-shot examples | 0ms marginal | Edge cases, tone, intent, hostility |
| **4. Output Guardrails** | Hallucination checks, output filter | 200-500ms | Grounding, content safety post-generation |

### Key Insight: Prompts ARE Guardrails

A query like *"Your advice about requirements management sucks!"* will pass both
keyword-based toxicity checks (mild language) and topic guards (contains domain
keywords). But a well-crafted system prompt can handle this naturally:

> "If the user expresses frustration rather than asking a question, acknowledge
> their concern briefly and offer to help with a specific topic."

This produces a better user experience than blocking, costs zero additional
latency, and avoids false positives on legitimate queries like *"This
traceability process sucks, how do I improve it?"*

### Design Guidelines

1. **Don't over-index on code guardrails.** Keyword lists and regex are brittle.
   They catch the obvious cases but miss creative circumvention and nuance.

2. **Use AI classification for the middle ground.** LLM-based topic guards and
   moderation APIs handle ambiguous inputs that deterministic checks miss.

3. **Lean on prompt engineering for edge cases.** The LLM is already processing
   the query -- give it instructions for graceful handling of hostility,
   non-questions, and boundary cases.

4. **Always validate outputs.** Even with perfect input guardrails, LLMs can
   hallucinate or produce unsafe content. Post-generation checks are essential.

5. **Test across all layers.** Unit tests for deterministic checks. LangSmith
   evaluation datasets for AI classifiers and prompt behavior. Manual testing
   for adversarial edge cases.

### Testing Strategy

- **Unit tests**: Validate deterministic guardrail logic (keyword matching,
  regex patterns, threshold comparisons)
- **LangSmith evaluations**: Build adversarial/edge-case datasets and run
  experiments to measure prompt robustness
- **Integration tests**: Verify the full guardrail pipeline end-to-end
- **Prompt regression tests**: Track prompt changes and measure impact on
  safety metrics via LangSmith experiments

### Sprint Planning Considerations

When planning guardrail work, always consider all four layers:
- Which layer is the best fit for the specific threat?
- Are there gaps where one layer's weakness isn't covered by another?
- Can prompt engineering solve this more naturally than code?
- What evaluation dataset do we need to test this properly?
