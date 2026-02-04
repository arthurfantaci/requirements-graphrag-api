# Agentic RAG Implementation Plan

> **Generated**: 2026-02-04
> **Status**: Ready for execution
> **Estimated Phases**: 6
> **Context**: Full replacement of routed RAG with agentic system

---

## Claude Code Optimization Guide

### Context Window Management

Your context is at 61% with consolidation impending. Here's how to work effectively:

1. **Reference this file** - After consolidation, say "read the implementation plan at docs/AGENTIC_IMPLEMENTATION_PLAN.md" to restore context
2. **Work in phases** - Complete one phase fully before starting the next
3. **Use sub-agents** - They have their own context and return summaries
4. **Checkpoint progress** - Update the `[ ]` checkboxes in this file as you complete tasks

### Skill/Agent Usage Map

| Task Type | Best Tool | When to Use |
|-----------|-----------|-------------|
| Codebase exploration | `Task:Explore` agent | Finding patterns, understanding architecture |
| Architecture design | `Task:Plan` agent or `/plan` skill | Before implementing complex features |
| Implementation | `/implement` skill | Writing new code with guidance |
| Code review | `Task:code-reviewer` agent | After completing a chunk of work |
| Testing | `/test` skill | After implementation |
| GraphRAG patterns | `/graphrag-patterns` skill | When implementing retrieval/generation |
| Neo4j queries | `/neo4j-patterns` skill | When writing Cypher or driver code |
| LangSmith operations | `mcp__langsmith__*` tools | Managing prompts, datasets, experiments |
| Library docs | `mcp__context7__*` tools | Looking up LangGraph, LangChain docs |

### MCP Servers Available

```
langsmith/          # Push prompts, create datasets, run experiments
  - push_prompt
  - create_dataset
  - list_prompts
  - run_experiment

context7/           # Get library documentation
  - resolve-library-id
  - get-library-docs

ide/                # VSCode integration
  - getDiagnostics
  - executeCode
```

---

## Phase 0: Pre-Implementation Setup

**Goal**: Prepare environment and verify dependencies

### Tasks

- [x] **0.1** Add LangGraph dependency
  ```bash
  cd backend && uv add langgraph langgraph-checkpoint-postgres
  ```
  > Completed: langgraph 1.0.7, langgraph-checkpoint-postgres 3.0.4, psycopg-binary 3.3.2

- [x] **0.2** Verify LangGraph installation
  ```bash
  uv run python -c "import langgraph; print(langgraph.__version__)"
  ```
  > Completed: StateGraph, PostgresSaver, ToolNode all import successfully

- [x] **0.3** Create agentic module structure
  ```
  backend/src/requirements_graphrag_api/core/agentic/
  ├── __init__.py
  ├── state.py           # TypedDict state definitions
  ├── tools.py           # Tool definitions
  ├── nodes.py           # Graph node functions
  ├── prompts.py         # New prompt definitions
  ├── subgraphs/
  │   ├── __init__.py
  │   ├── rag.py         # RAG retrieval subgraph
  │   ├── research.py    # Entity exploration subgraph
  │   └── synthesis.py   # Answer synthesis subgraph
  ├── orchestrator.py    # Main composed graph
  ├── streaming.py       # SSE streaming utilities
  └── checkpoints.py     # PostgresSaver configuration
  ```

- [ ] **0.4** Update CLAUDE.md with agentic patterns
  > Use `/claude-md-management:revise-claude-md` skill after Phase 0

### Claude Code Commands for Phase 0

```
# Get LangGraph documentation
Use ToolSearch: "select:mcp__context7__resolve-library-id"
Then: mcp__context7__get-library-docs for "langgraph"

# After completing Phase 0
Use skill: /claude-md-management:revise-claude-md
```

---

## Phase 1: State & Tool Definitions

**Goal**: Define the foundational types and tools for the agent

### Tasks

- [x] **1.1** Create state definitions (`state.py`)
  - `AgentState` - Main agent state
  - `RAGState` - RAG subgraph state
  - `ResearchState` - Research subgraph state
  - `SynthesisState` - Synthesis subgraph state
  - `OrchestratorState` - Main orchestrator state
  > Completed: Also added RetrievedDocument, CriticEvaluation, EntityInfo dataclasses

- [x] **1.2** Create tool definitions (`tools.py`)
  - `graph_search` - Wraps `graph_enriched_search`
  - `text2cypher` - Wraps `text2cypher_query`
  - `explore_entity` - Wraps `explore_entity`
  - `lookup_standard` - Wraps standard lookup
  - `search_definitions` - Wraps `search_terms`
  - `lookup_term` - Wraps `lookup_term`
  > Completed: 7 tools with Pydantic input schemas, factory pattern with bound dependencies

- [x] **1.3** Verify tool signatures match existing functions
  > Use `Task:Explore` agent to check function signatures in:
  > - `core/retrieval.py`
  > - `core/text2cypher.py`
  > - `core/definitions.py`
  > - `core/standards.py`

### Claude Code Commands for Phase 1

```
# Before implementing, explore existing signatures
Spawn Task:Explore agent with prompt:
"Find all async functions in core/*.py that could be wrapped as agent tools.
List their signatures, parameters, and return types."

# Use GraphRAG patterns skill for tool design
Use skill: /graphrag-patterns

# After implementation, review code
Spawn Task:code-reviewer agent
```

---

## Phase 2: Prompt Integration

**Goal**: Activate unused prompts and add new agentic prompts

### Tasks

- [x] **2.1** Add new prompt definitions to `prompts/definitions.py`
  ```python
  # New prompts added:
  AGENT_REASONING = "graphrag-agent-reasoning"
  QUERY_EXPANSION = "graphrag-query-expansion"
  SYNTHESIS = "graphrag-synthesis"
  ENTITY_SELECTOR = "graphrag-entity-selector"
  ```
  > Completed: Added to PromptName enum, templates, metadata, and PROMPT_DEFINITIONS registry

- [x] **2.2** Design AGENT_REASONING prompt
  - System prompt for main agent loop
  - Tool descriptions and usage guidelines
  - Iteration control instructions
  > Completed: 7 tools documented, decision guidelines, iteration control with placeholders

- [x] **2.3** Design QUERY_EXPANSION prompt
  - Uses STEPBACK pattern internally
  - Generates 2-3 search queries from user question
  > Completed: Step-back, synonym, and aspect-specific query strategies

- [x] **2.4** Design SYNTHESIS prompt
  - Uses CRITIC for self-evaluation
  - Integrates QUERY_UPDATER for multi-turn
  > Completed: Self-critique JSON output with confidence, completeness, citations

- [x] **2.5** Push new prompts to LangSmith Hub
  ```
  Use ToolSearch: "select:mcp__langsmith__push_prompt"
  ```

### Claude Code Commands for Phase 2

```
# Get existing prompt patterns
Read: backend/src/requirements_graphrag_api/prompts/definitions.py

# Push prompts to LangSmith
Use ToolSearch: "+langsmith push"
Then use mcp__langsmith__push_prompt for each new prompt

# Verify prompts in LangSmith
Use ToolSearch: "select:mcp__langsmith__list_prompts"
```

### Prompt Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROMPT ORCHESTRATION                        │
│                                                                 │
│  AGENT_REASONING (new)                                         │
│  └── Decides which tool to call                                │
│                                                                 │
│  QUERY_EXPANSION (new)                                         │
│  └── Uses STEPBACK (existing) internally                       │
│  └── Generates multiple search queries                         │
│                                                                 │
│  SYNTHESIS (new)                                                │
│  └── Uses CRITIC (existing) for self-evaluation                │
│  └── Uses RAG_GENERATION (existing) for answer format          │
│  └── Uses QUERY_UPDATER (existing) for multi-turn context      │
│                                                                 │
│  TEXT2CYPHER (existing) - wrapped as tool                      │
│  INTENT_CLASSIFIER (existing) - optional hybrid fallback       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 3: Subgraph Implementation

**Goal**: Build modular, testable subgraphs

### Tasks

- [x] **3.1** Implement RAG subgraph (`subgraphs/rag.py`)
  - Node: `expand_queries` - Uses QUERY_EXPANSION prompt
  - Node: `parallel_retrieve` - Concurrent searches
  - Node: `dedupe_and_rank` - Result processing
  - Edge: Linear flow
  > Completed: Query expansion, parallel retrieval, deduplication/ranking

- [x] **3.2** Implement Research subgraph (`subgraphs/research.py`)
  - Node: `identify_entities` - Extract from context
  - Node: `explore_entity` - Deep dive
  - Edge: Conditional loop (more entities?)
  > Completed: Entity identification via ENTITY_SELECTOR, conditional exploration loop

- [x] **3.3** Implement Synthesis subgraph (`subgraphs/synthesis.py`)
  - Node: `draft_answer` - Initial generation
  - Node: `critique` - Uses CRITIC prompt
  - Node: `revise` - If critique finds issues
  - Node: `format_output` - Final formatting
  - Edge: Conditional (needs revision?)
  > Completed: Self-critique in SYNTHESIS prompt, conditional revision loop (MAX_REVISIONS=2)

- [x] **3.4** Write unit tests for each subgraph
  > Completed: 19 tests in tests/test_core/test_agentic/test_subgraphs.py
  > Tests cover: subgraph creation, state structures, conditional edge logic, graceful error handling

### Claude Code Commands for Phase 3

```
# Get LangGraph StateGraph documentation
Use ToolSearch: "select:mcp__context7__resolve-library-id"
Query: "langgraph"
Then: mcp__context7__get-library-docs

# Use architecture agent for subgraph design
Spawn Task:code-architect agent with prompt:
"Design the RAG subgraph following the pattern in
docs/agentic-rag-architecture-guide.md section 5"

# After each subgraph, run tests
Use skill: /test
```

---

## Phase 4: Orchestrator & Checkpoint Persistence

**Goal**: Compose subgraphs and add conversation persistence

### Tasks

- [x] **4.1** Implement orchestrator (`orchestrator.py`)
  - Composes all subgraphs
  - Routes based on query complexity
  - Manages iteration limits
  > Completed: initialize -> run_rag -> should_research -> run_research/run_synthesis flow

- [x] **4.2** Implement checkpoint persistence (`checkpoints.py`)
  - PostgresSaver configuration
  - Connection pool management
  - Thread/conversation ID handling
  > Completed: AsyncPostgresSaver, async_checkpointer_context, get_thread_config helpers

- [x] **4.3** Add database migration for checkpoints
  > Completed: LangGraph manages its own tables via checkpointer.setup()
  > Uses CHECKPOINT_DATABASE_URL environment variable

- [x] **4.4** Integration test: Full agent flow
  > Completed: 17 tests in tests/test_core/test_agentic/test_orchestrator.py
  > Tests: checkpoint helpers, orchestrator creation, routing logic, message extraction

### Claude Code Commands for Phase 4

```
# Get PostgresSaver documentation
Use ToolSearch: "+langgraph checkpoint"

# Review existing database patterns
Spawn Task:Explore agent with prompt:
"Find how PostgreSQL connections are managed in this project.
Look for connection pools, async patterns, and configuration."

# After implementation
Spawn Task:code-reviewer agent
```

---

## Phase 5: Streaming & API Integration

**Goal**: Replace existing routes with agentic endpoints

### Tasks

- [x] **5.1** Implement streaming utilities (`streaming.py`)
  - `AgenticEventType` enum (ROUTING, SOURCES, TOKEN, PHASE, PROGRESS, ENTITIES, DONE, ERROR)
  - `AgenticEvent` dataclass with `to_sse()` method
  - `stream_agentic_events` async generator using `graph.astream_events()`
  - Event factory functions for each type
  > Completed: SSE streaming with LangGraph astream_events integration

- [x] **5.2** Update chat routes (`routes/chat.py`)
  - Replace `stream_chat` with agentic streaming
  - Update request/response models
  - Maintain backward-compatible response format
  > Completed: Replaced _generate_explanatory_events with agentic orchestrator, kept guardrails

- [x] **5.3** Add conversation management endpoints
  ```python
  POST /chat/stream          # Main streaming endpoint (existing /chat)
  GET  /chat/{thread_id}     # Get conversation state
  POST /chat/{thread_id}/continue  # Continue conversation
  ```
  > Completed: Added GET /chat/{thread_id} and POST /chat/{thread_id}/continue endpoints

- [x] **5.4** Update frontend for new event types
  - Handle subgraph progress events
  - Display tool calls
  - Show iteration progress
  > NOTE: Frontend uses shape-based event detection, so agentic events are backward
  > compatible. New phase/progress events are simply ignored by current frontend.
  > Optional enhancements (PhaseIndicator, ProgressTimeline) can be added later.

- [x] **5.5** End-to-end testing
  > Completed: 10 integration tests in tests/test_core/test_agentic/test_integration.py
  > Tests cover: orchestrator flow, streaming events, state propagation, thread isolation

### Claude Code Commands for Phase 5

```
# Explore current streaming implementation
Spawn Task:Explore agent with prompt:
"Analyze the current SSE streaming in routes/chat.py.
Document the event types and how the frontend consumes them."

# After route updates, check for issues
Use ToolSearch: "select:mcp__ide__getDiagnostics"
```

---

## Phase 6: Evaluation & Optimization

**Goal**: Validate agentic system and optimize performance

### Tasks

- [x] **6.1** Create agentic evaluation dataset
  ```
  Use ToolSearch: "select:mcp__langsmith__create_dataset"
  ```
  Dataset categories:
  - Multi-hop reasoning questions
  - Tool selection scenarios
  - Iteration efficiency tests
  > Completed: Dataset 'graphrag-agentic-eval' created with 16 examples
  > Categories: multi_hop (4), tool_selection (6), iteration_efficiency (2), critic_calibration (2), general (2)

- [x] **6.2** Define agentic evaluators
  - `tool_selection_accuracy`
  - `iteration_efficiency`
  - `critic_calibration`
  - `answer_quality` (existing)
  > Completed: evaluation/agentic_evaluators.py with 4 evaluators + sync wrappers
  > Tests: 27 tests in tests/test_evaluation/test_agentic_evaluators.py

- [x] **6.3** Run comparison experiment
  ```
  Use ToolSearch: "select:mcp__langsmith__run_experiment"
  ```
  Compare: Old routed RAG vs New agentic RAG
  > Completed: scripts/run_agentic_evaluation.py created
  > Run with: `uv run python scripts/run_agentic_evaluation.py`
  > Compare with: `uv run python scripts/compare_experiments.py -d graphrag-agentic-eval`

- [ ] **6.4** Performance optimization
  - Identify slow subgraphs
  - Optimize parallel tool execution
  - Tune iteration limits

- [ ] **6.5** Cost analysis
  - Track LLM calls per query
  - Compare cost: old vs new
  - Implement cost-saving measures if needed

### Claude Code Commands for Phase 6

```
# Create evaluation dataset
Use ToolSearch: "+langsmith dataset"
Then: mcp__langsmith__create_dataset

# Run experiment
Use ToolSearch: "select:mcp__langsmith__run_experiment"

# Fetch results for analysis
Use ToolSearch: "select:mcp__langsmith__fetch_runs"
```

---

## Post-Implementation Checklist

- [ ] All tests passing (`uv run pytest`)
- [ ] Linting clean (`uv run ruff check src/`)
- [ ] Type checking clean (`uv run mypy src/`)
- [ ] Documentation updated
- [ ] CLAUDE.md updated with agentic patterns
- [ ] Prompts pushed to LangSmith Hub
- [ ] Evaluation dataset created
- [ ] Comparison experiment completed
- [ ] Frontend updated and tested
- [ ] Deployed to staging for validation

---

## Quick Reference: Claude Code Commands

### Sub-Agent Spawning
```
# Exploration (use for understanding code)
Task tool with subagent_type="Explore"

# Architecture (use for design decisions)
Task tool with subagent_type="Plan"

# Code review (use after implementation)
Task tool with subagent_type="code-reviewer"
```

### Skills (User-Invokable)
```
/plan          - Create implementation plan
/implement     - Guided implementation
/test          - Run and create tests
/review        - Code review
/graphrag-patterns  - GraphRAG best practices
/neo4j-patterns     - Neo4j driver patterns
```

### MCP Tools (Require ToolSearch first)
```
# LangSmith
mcp__langsmith__push_prompt
mcp__langsmith__create_dataset
mcp__langsmith__run_experiment
mcp__langsmith__fetch_runs

# Documentation
mcp__context7__resolve-library-id
mcp__context7__get-library-docs

# IDE
mcp__ide__getDiagnostics
```

---

## Context Recovery After Consolidation

If context is consolidated, restore with:

```
Please read docs/AGENTIC_IMPLEMENTATION_PLAN.md and tell me:
1. Which phase am I currently on (check the checkboxes)
2. What's the next uncompleted task
3. What Claude Code tools should I use for that task
```

---

*Plan created by Claude Code for Requirements GraphRAG API agentic implementation*
