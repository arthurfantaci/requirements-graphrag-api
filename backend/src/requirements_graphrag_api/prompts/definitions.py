"""Centralized prompt definitions for GraphRAG.

This module contains all prompt templates used throughout the application.
Local definitions serve as fallbacks when LangSmith Hub is unavailable.
Prompts use ChatPromptTemplate for multi-message support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class PromptName(StrEnum):
    """Unique identifiers for prompts in the catalog.

    These names correspond to LangSmith Hub prompt names in the format:
    {organization}/{prompt_name}

    For local development, the simple name is used as the key.
    """

    # Existing prompts
    INTENT_CLASSIFIER = "graphrag-intent-classifier"
    CRITIC = "graphrag-critic"
    STEPBACK = "graphrag-stepback"
    QUERY_UPDATER = "graphrag-query-updater"
    RAG_GENERATION = "graphrag-rag-generation"
    TEXT2CYPHER = "graphrag-text2cypher"

    # Agentic prompts (Phase 2)
    AGENT_REASONING = "graphrag-agent-reasoning"
    QUERY_EXPANSION = "graphrag-query-expansion"
    SYNTHESIS = "graphrag-synthesis"
    ENTITY_SELECTOR = "graphrag-entity-selector"


@dataclass(frozen=True)
class PromptMetadata:
    """Metadata for prompt versioning and evaluation.

    Attributes:
        version: Semantic version string (e.g., "1.0.0").
        description: Human-readable description of the prompt's purpose.
        input_variables: List of required input variable names.
        output_format: Expected output format (e.g., "json", "text", "cypher").
        evaluation_criteria: Criteria for evaluating prompt effectiveness.
        tags: Tags for categorization and filtering.
    """

    version: str
    description: str
    input_variables: list[str]
    output_format: str = "text"
    evaluation_criteria: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PromptDefinition:
    """Complete prompt definition with template and metadata.

    Attributes:
        name: Unique prompt identifier.
        template: LangChain ChatPromptTemplate.
        metadata: Associated metadata for versioning and evaluation.
    """

    name: PromptName
    template: ChatPromptTemplate
    metadata: PromptMetadata


# =============================================================================
# INTENT CLASSIFIER PROMPT
# Classifies queries as explanatory (RAG) or structured (Cypher)
# =============================================================================

INTENT_CLASSIFIER_SYSTEM: Final[
    str
] = """You classify user queries for a Requirements Management knowledge base.

Your task is to determine the user's intent to route their query appropriately.

## Intent Types

**EXPLANATORY** - User wants understanding, explanation, or synthesis
- Questions about concepts: "What is requirements traceability?"
- How-to questions: "How do I implement change management?"
- Best practices: "What are best practices for verification?"
- Why questions: "Why is traceability important?"
- Comparisons: "What's the difference between verification and validation?"

**STRUCTURED** - User wants enumeration, lists, counts, or specific data
- List requests: "List all webinars", "Show me all videos"
- Counts: "How many articles mention ISO 26262?"
- Enumeration: "What standards are in the knowledge base?"
- Table requests: "Provide a table of all tools"
- Specific lookups: "Which articles discuss MBSE?"

## Classification Rules

1. Keywords suggesting STRUCTURED intent:
   - "list all", "show all", "show me all"
   - "how many", "count", "total number"
   - "table of", "enumerate"
   - "which [noun]s" (plural entity request)

2. Keywords suggesting EXPLANATORY intent:
   - "what is", "what are" (when asking for definition/explanation)
   - "how do I", "how to", "how can I"
   - "explain", "describe", "help me understand"
   - "why", "best practices", "recommendations"

3. When ambiguous, prefer EXPLANATORY (provides richer context)

Respond with ONLY a JSON object:
{{"intent": "explanatory"}} or {{"intent": "structured"}}"""

INTENT_CLASSIFIER_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", INTENT_CLASSIFIER_SYSTEM),
        ("human", "Query: {question}"),
    ]
)

INTENT_CLASSIFIER_METADATA = PromptMetadata(
    version="1.0.0",
    description="Classifies queries as explanatory (RAG) or structured (Cypher) for routing",
    input_variables=["question"],
    output_format="json",
    evaluation_criteria=[
        "correct_classification",
        "handling_of_ambiguous_queries",
        "valid_json_output",
    ],
    tags=["routing", "classification", "intent"],
)


# =============================================================================
# CRITIC PROMPT
# Evaluates whether context is sufficient to answer the question
# =============================================================================

CRITIC_SYSTEM: Final[str] = """You are a quality evaluator for a Requirements Management RAG system.

Your task is to assess whether the retrieved context is sufficient to answer the user's question.

Evaluate:
1. **Relevance**: Does the context address the question?
2. **Completeness**: Are all aspects of the question covered?
3. **Confidence**: How confident are you that a good answer can be generated?

Context:
{context}

Respond with a JSON object:
{{
    "answerable": true/false,
    "confidence": 0.0-1.0,
    "completeness": "complete" | "partial" | "insufficient",
    "missing_aspects": ["list of missing information if any"],
    "followup_query": "suggested query to fill gaps (if needed)",
    "reasoning": "Brief explanation of your assessment"
}}"""

CRITIC_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", CRITIC_SYSTEM),
        ("human", "Question: {question}"),
    ]
)

CRITIC_METADATA = PromptMetadata(
    version="1.0.0",
    description="Evaluates retrieval quality and context sufficiency",
    input_variables=["context", "question"],
    output_format="json",
    evaluation_criteria=[
        "calibrated_confidence",
        "accurate_completeness",
        "useful_followup_queries",
    ],
    tags=["evaluation", "agentic", "quality"],
)


# =============================================================================
# STEPBACK PROMPT
# Generates broader queries for better context retrieval
# =============================================================================

STEPBACK_SYSTEM: Final[
    str
] = """You are a query refinement assistant for a Requirements Management knowledge base.

Your task is to transform specific questions into broader, more general queries.
This helps retrieve better context by finding foundational information first.

Guidelines:
- Remove specific details, names, or narrow constraints
- Focus on the underlying concept or principle
- Maintain the domain relevance (requirements management)
- Keep the query concise (1-2 sentences)

Examples:
- Specific: "What are the ASIL levels in ISO 26262?"
  Broader: "What is ISO 26262 and how does it classify safety?"

- Specific: "How does Jama Connect handle trace links?"
  Broader: "What is requirements traceability and how is it implemented?"

- Specific: "What's the difference between verification and validation in DO-178C?"
  Broader: "What are verification and validation in requirements management?"

Respond with ONLY the broader question, no explanation."""

STEPBACK_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", STEPBACK_SYSTEM),
        ("human", "Specific question: {question}\n\nBroader question:"),
    ]
)

STEPBACK_METADATA = PromptMetadata(
    version="1.0.0",
    description="Transforms specific questions into broader queries for step-back prompting",
    input_variables=["question"],
    output_format="text",
    evaluation_criteria=[
        "appropriate_generalization",
        "domain_relevance",
        "retrieval_improvement",
    ],
    tags=["query_refinement", "agentic", "stepback"],
)


# =============================================================================
# QUERY UPDATER PROMPT
# Updates remaining questions with context from answered parts
# =============================================================================

QUERY_UPDATER_SYSTEM: Final[str] = """You are a query refinement assistant for multi-part questions.

Your task is to update a question by incorporating relevant context from previously
answered questions.

This helps:
1. Resolve pronoun references ("it", "this", "that")
2. Incorporate established facts from previous answers
3. Make the question more specific and focused

Previous Q&A:
{previous_answers}

Guidelines:
- Keep the core intent of the original question
- Add relevant context from previous answers
- Remove redundant information already covered
- Ensure the question is self-contained

Respond with ONLY the updated question, no explanation."""

QUERY_UPDATER_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", QUERY_UPDATER_SYSTEM),
        ("human", "Original question: {question}\n\nUpdated question:"),
    ]
)

QUERY_UPDATER_METADATA = PromptMetadata(
    version="1.0.0",
    description="Refines multi-part questions with context from previous answers",
    input_variables=["previous_answers", "question"],
    output_format="text",
    evaluation_criteria=[
        "context_incorporation",
        "pronoun_resolution",
        "question_clarity",
    ],
    tags=["query_refinement", "agentic", "multi-turn"],
)


# =============================================================================
# RAG GENERATION PROMPT
# Generates answers with citations from retrieved context
# =============================================================================

RAG_GENERATION_SYSTEM: Final[
    str
] = """You are a Requirements Management expert answering questions based on retrieved context.

Your task is to provide accurate, helpful answers grounded in the provided context.

Context:
{context}

Related Entities: {entities}

Guidelines:
1. **Ground your answer in the context** - cite sources when possible
2. **Be accurate** - don't make up information not in the context
3. **Be helpful** - explain concepts clearly for practitioners
4. **Acknowledge limitations** - say if the context doesn't fully answer the question
5. **Use domain terminology** - maintain technical accuracy

Format citations as [Source N] where N is the source number from the context."""

RAG_GENERATION_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_GENERATION_SYSTEM),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{question}"),
    ]
)

RAG_GENERATION_METADATA = PromptMetadata(
    version="1.1.0",
    description=(
        "Generates grounded answers with citations from retrieved context, "
        "supports multi-turn conversation"
    ),
    # Note: 'history' is optional (via MessagesPlaceholder), so not listed in required variables
    input_variables=["context", "entities", "question"],
    output_format="text",
    evaluation_criteria=[
        "faithfulness",
        "answer_relevance",
        "context_utilization",
        "citation_accuracy",
    ],
    tags=["generation", "rag", "citation", "multi-turn"],
)


# =============================================================================
# TEXT2CYPHER PROMPT
# Generates Cypher queries from natural language
# =============================================================================

TEXT2CYPHER_SYSTEM: Final[
    str
] = """You are a Cypher query generator for a Requirements Management knowledge graph.

Database Schema:
{schema}

Key Node Types:
- Article: article_title, url, chapter_title
- Chunk: text, index (linked to Article via FROM_ARTICLE)
- Entity: name, display_name (subtypes: Tool, Concept, Standard, Industry, Methodology)
- Definition: term, definition, url, acronym
- Standard: name, display_name, organization
- Webinar: title, url, thumbnail_url
- Video: title, url
- Image: url, alt_text, context

Key Relationships:
- (Chunk)-[:FROM_ARTICLE]->(Article)
- (Entity)-[:MENTIONED_IN]->(Chunk)
- (Standard)-[:APPLIES_TO]->(Industry)
- (Article)-[:HAS_WEBINAR]->(Webinar)
- (Article)-[:HAS_VIDEO]->(Video)
- (Article)-[:HAS_IMAGE]->(Image)
- (Chunk)-[:NEXT_CHUNK]->(Chunk)

Few-Shot Examples:
{examples}

Guidelines:
1. Use MATCH for read queries only (no CREATE, MERGE, DELETE)
2. Use toLower() for case-insensitive matching
3. Use CONTAINS for partial text matching
4. Return meaningful property values, not just node counts
5. Limit results to 25 unless aggregating
6. For media queries (webinars, videos, images), traverse from Article
7. For media listing queries, use COLLECT(DISTINCT ...) to aggregate
   source articles and avoid duplicate rows

Generate only the Cypher query, no explanation."""

TEXT2CYPHER_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", TEXT2CYPHER_SYSTEM),
        ("human", "Question: {question}\n\nCypher:"),
    ]
)

TEXT2CYPHER_METADATA = PromptMetadata(
    version="1.0.0",
    description="Converts natural language questions to Cypher queries",
    input_variables=["schema", "examples", "question"],
    output_format="cypher",
    evaluation_criteria=[
        "syntactic_validity",
        "semantic_correctness",
        "result_accuracy",
        "read_only_compliance",
    ],
    tags=["text2cypher", "graph", "query_generation"],
)


# =============================================================================
# TEXT2CYPHER FEW-SHOT EXAMPLES
# =============================================================================

TEXT2CYPHER_EXAMPLES: Final[str] = """Example 1:
Question: How many articles are in the knowledge base?
Cypher: MATCH (a:Article)
RETURN count(a) AS article_count

Example 2:
Question: What chapters does the guide have?
Cypher: MATCH (a:Article)
WHERE a.article_title IS NOT NULL
RETURN DISTINCT a.article_title AS chapter
ORDER BY chapter

Example 3:
Question: Find all tools mentioned in the guide
Cypher: MATCH (t:Tool)
RETURN t.name AS tool, t.display_name AS display_name
ORDER BY t.display_name

Example 4:
Question: Which entities are most frequently mentioned?
Cypher: MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
WITH e, count(c) AS mentions
RETURN labels(e)[0] AS type, e.display_name AS entity, mentions
ORDER BY mentions DESC
LIMIT 15

Example 5:
Question: What standards apply to the automotive industry?
Cypher: MATCH (s:Standard)-[:APPLIES_TO]->(i:Industry)
WHERE toLower(i.name) CONTAINS 'automotive'
   OR toLower(s.display_name) CONTAINS 'automotive'
RETURN s.name AS standard, s.display_name AS display_name, s.organization AS organization

Example 6:
Question: Which articles mention ISO 26262?
Cypher: MATCH (e:Standard)-[:MENTIONED_IN]->(c:Chunk)-[:FROM_ARTICLE]->(a:Article)
WHERE toLower(e.name) CONTAINS 'iso 26262'
RETURN DISTINCT a.article_title AS article, a.url AS url

Example 7:
Question: What are the top 5 most mentioned entities?
Cypher: MATCH (entity)-[:MENTIONED_IN]->(c:Chunk)
WITH labels(entity)[0] AS entity_type, entity.display_name AS entity_name, count(c) AS mention_count
RETURN entity_type, entity_name, mention_count
ORDER BY mention_count DESC
LIMIT 5

Example 8:
Question: List all webinars
Cypher: MATCH (a:Article)-[:HAS_WEBINAR]->(w:Webinar)
RETURN w.title AS webinar_title, w.url AS webinar_url,
       COLLECT(DISTINCT a.article_title) AS source_articles
ORDER BY w.title

Example 9:
Question: Show me all videos in the knowledge base
Cypher: MATCH (a:Article)-[:HAS_VIDEO]->(v:Video)
RETURN v.title AS video_title, v.url AS video_url,
       COLLECT(DISTINCT a.article_title) AS source_articles
ORDER BY v.title

Example 10:
Question: What images are available about traceability?
Cypher: MATCH (a:Article)-[:HAS_IMAGE]->(img:Image)
WHERE toLower(img.alt_text) CONTAINS 'traceability'
RETURN img.alt_text AS description, img.url AS image_url,
       COLLECT(DISTINCT a.article_title) AS source_articles
ORDER BY description
LIMIT 10

Example 11:
Question: How many webinars and videos are there?
Cypher: MATCH (w:Webinar)
WITH count(w) AS webinar_count
MATCH (v:Video)
RETURN webinar_count, count(v) AS video_count
"""


# =============================================================================
# AGENT REASONING PROMPT
# Main agent loop for tool selection and reasoning
# =============================================================================

AGENT_REASONING_SYSTEM: Final[str] = """You are an intelligent Requirements Management assistant \
with access to specialized tools.

Your goal is to answer questions about requirements management, traceability, standards, \
and related concepts by selecting and using the appropriate tools.

## Available Tools

1. **graph_search** - Search the knowledge base using hybrid vector + graph retrieval
   - Use for: explanations, concepts, best practices, how-to questions
   - Returns: ranked content chunks with citations

2. **text2cypher** - Convert questions to Cypher queries for structured data
   - Use for: counts, lists, enumerations, specific lookups
   - Returns: query results from the knowledge graph

3. **explore_entity** - Deep dive into a specific entity
   - Use for: understanding relationships, finding related content
   - Returns: entity details, related entities, article mentions

4. **lookup_standard** - Get information about industry standards
   - Use for: specific standard details (ISO 26262, DO-178C, etc.)
   - Returns: standard info, applicable industries, related standards

5. **search_standards** - Search for standards by criteria
   - Use for: finding relevant standards for a domain
   - Returns: list of matching standards

6. **search_definitions** - Search terminology definitions
   - Use for: finding definitions of technical terms
   - Returns: matching term definitions

7. **lookup_term** - Get precise definition of a term
   - Use for: exact term/acronym definitions
   - Returns: term definition with source

## Decision Guidelines

1. **Start with the most likely tool** - Use graph_search for explanatory questions, \
text2cypher for structured queries
2. **Use multiple tools when needed** - Complex questions may require combining results
3. **Iterate if context is insufficient** - If first results don't fully answer, try different tools
4. **Know when to stop** - Don't over-iterate; 2-3 tool calls usually suffice
5. **Synthesize clearly** - Combine tool results into a coherent, cited answer

## Iteration Control

- Maximum iterations: {max_iterations}
- Current iteration: {iteration}
- Stop when you have sufficient context to answer confidently

Think step-by-step about which tool(s) will best answer the question."""

AGENT_REASONING_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_REASONING_SYSTEM),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

AGENT_REASONING_METADATA = PromptMetadata(
    version="1.0.0",
    description="Main agent loop prompt for tool selection and reasoning",
    input_variables=["max_iterations", "iteration", "messages"],
    output_format="text",
    evaluation_criteria=[
        "tool_selection_accuracy",
        "iteration_efficiency",
        "reasoning_quality",
    ],
    tags=["agentic", "reasoning", "tool_selection"],
)


# =============================================================================
# QUERY EXPANSION PROMPT
# Generates multiple search queries from user question (uses STEPBACK internally)
# =============================================================================

QUERY_EXPANSION_SYSTEM: Final[str] = """You are a query expansion specialist for a \
Requirements Management knowledge base.

Your task is to generate multiple search queries that together will retrieve \
comprehensive context for answering the user's question.

## Expansion Strategies

1. **Step-back Query**: A broader, more general version of the question
   - Removes specific details to find foundational information
   - Example: "What are ASIL levels in ISO 26262?" → "What is ISO 26262 safety classification?"

2. **Synonym/Alternate Terms**: Rephrase using different terminology
   - Requirements Management has many synonymous terms
   - Example: "traceability" ↔ "trace links" ↔ "requirements linking"

3. **Aspect-Specific Query**: Focus on one aspect of a multi-faceted question
   - Break complex questions into components
   - Example: "How to implement V&V?" → "What is verification?" + "What is validation?"

## Output Format

Generate 2-4 queries as a JSON array:
{{
    "queries": [
        {{"query": "...", "strategy": "stepback|synonym|aspect"}},
        {{"query": "...", "strategy": "stepback|synonym|aspect"}}
    ],
    "reasoning": "Brief explanation of why these queries will help"
}}

Always include at least one step-back query for foundational context."""

QUERY_EXPANSION_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", QUERY_EXPANSION_SYSTEM),
        ("human", "Original question: {question}\n\nGenerate expanded queries:"),
    ]
)

QUERY_EXPANSION_METADATA = PromptMetadata(
    version="1.0.0",
    description="Generates multiple search queries using step-back and synonym strategies",
    input_variables=["question"],
    output_format="json",
    evaluation_criteria=[
        "query_diversity",
        "retrieval_improvement",
        "stepback_quality",
    ],
    tags=["agentic", "query_expansion", "stepback"],
)


# =============================================================================
# SYNTHESIS PROMPT
# Generates final answer with self-critique (uses CRITIC internally)
# =============================================================================

SYNTHESIS_SYSTEM: Final[str] = """You are a Requirements Management expert synthesizing \
answers from retrieved context.

Your task is to generate a comprehensive, accurate answer and then critically evaluate it.

## Retrieved Context
{context}

## Related Entities
{entities}

## Previous Conversation (if multi-turn)
{previous_context}

## Synthesis Guidelines

1. **Ground in context** - Every claim should be supported by the retrieved context
2. **Cite sources** - Use [Source N] format for citations
3. **Acknowledge gaps** - If context is incomplete, say so
4. **Use technical terminology** - Maintain domain accuracy
5. **Structure clearly** - Use headings, bullets for complex answers

## Self-Critique (CRITICAL)

After drafting your answer, evaluate it:
- **Completeness**: Does it address all parts of the question?
- **Accuracy**: Is every claim supported by context?
- **Confidence**: How confident are you (0.0-1.0)?

If confidence < 0.7 or completeness is partial, indicate what additional information would help.

## Output Format

{{
    "answer": "Your synthesized answer with [Source N] citations...",
    "critique": {{
        "completeness": "complete|partial|insufficient",
        "confidence": 0.0-1.0,
        "missing_aspects": ["list any gaps"],
        "would_benefit_from": "suggested follow-up query if needed"
    }},
    "citations": ["Source 1 title", "Source 2 title", ...]
}}"""

SYNTHESIS_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYNTHESIS_SYSTEM),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "Question: {question}\n\nSynthesize your answer:"),
    ]
)

SYNTHESIS_METADATA = PromptMetadata(
    version="1.0.0",
    description="Synthesizes answers with self-critique and citations",
    input_variables=["context", "entities", "previous_context", "question"],
    output_format="json",
    evaluation_criteria=[
        "faithfulness",
        "completeness",
        "citation_accuracy",
        "self_critique_calibration",
    ],
    tags=["agentic", "synthesis", "critic", "citation"],
)


# =============================================================================
# ENTITY SELECTOR PROMPT
# Selects entities for deep exploration from context
# =============================================================================

ENTITY_SELECTOR_SYSTEM: Final[str] = """You are an entity analysis specialist for a \
Requirements Management knowledge base.

Your task is to identify entities from the context that warrant deeper exploration \
to better answer the question.

## Entity Types in Knowledge Base

- **Standard**: Industry standards (ISO 26262, DO-178C, IEC 62304, etc.)
- **Tool**: Requirements management tools (Jama Connect, DOORS, etc.)
- **Concept**: Domain concepts (traceability, verification, validation, etc.)
- **Methodology**: Development methodologies (Agile, V-Model, MBSE, etc.)
- **Industry**: Industries (automotive, aerospace, medical, etc.)

## Selection Criteria

Select entities that:
1. Are central to answering the question
2. Appear in the context but lack sufficient detail
3. Have relationships that would enrich the answer
4. The user might want to learn more about

Do NOT select:
- Entities already well-explained in context
- Generic terms that aren't specific entities
- More than 3 entities (focus on most important)

## Current Context
{context}

## Output Format

{{
    "entities": [
        {{"name": "...", "type": "Standard|Tool|Concept|...", "reason": "why explore"}},
        ...
    ],
    "exploration_priority": "high|medium|low",
    "reasoning": "Brief explanation of selection"
}}

If no entities need exploration, return empty list with reasoning."""

ENTITY_SELECTOR_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", ENTITY_SELECTOR_SYSTEM),
        ("human", "Question: {question}\n\nSelect entities to explore:"),
    ]
)

ENTITY_SELECTOR_METADATA = PromptMetadata(
    version="1.0.0",
    description="Identifies entities that warrant deeper exploration",
    input_variables=["context", "question"],
    output_format="json",
    evaluation_criteria=[
        "entity_relevance",
        "selection_precision",
        "exploration_value",
    ],
    tags=["agentic", "entity_selection", "research"],
)


# =============================================================================
# PROMPT DEFINITIONS REGISTRY
# =============================================================================

PROMPT_DEFINITIONS: Final[dict[PromptName, PromptDefinition]] = {
    # Existing prompts
    PromptName.INTENT_CLASSIFIER: PromptDefinition(
        name=PromptName.INTENT_CLASSIFIER,
        template=INTENT_CLASSIFIER_TEMPLATE,
        metadata=INTENT_CLASSIFIER_METADATA,
    ),
    PromptName.CRITIC: PromptDefinition(
        name=PromptName.CRITIC,
        template=CRITIC_TEMPLATE,
        metadata=CRITIC_METADATA,
    ),
    PromptName.STEPBACK: PromptDefinition(
        name=PromptName.STEPBACK,
        template=STEPBACK_TEMPLATE,
        metadata=STEPBACK_METADATA,
    ),
    PromptName.QUERY_UPDATER: PromptDefinition(
        name=PromptName.QUERY_UPDATER,
        template=QUERY_UPDATER_TEMPLATE,
        metadata=QUERY_UPDATER_METADATA,
    ),
    PromptName.RAG_GENERATION: PromptDefinition(
        name=PromptName.RAG_GENERATION,
        template=RAG_GENERATION_TEMPLATE,
        metadata=RAG_GENERATION_METADATA,
    ),
    PromptName.TEXT2CYPHER: PromptDefinition(
        name=PromptName.TEXT2CYPHER,
        template=TEXT2CYPHER_TEMPLATE,
        metadata=TEXT2CYPHER_METADATA,
    ),
    # Agentic prompts (Phase 2)
    PromptName.AGENT_REASONING: PromptDefinition(
        name=PromptName.AGENT_REASONING,
        template=AGENT_REASONING_TEMPLATE,
        metadata=AGENT_REASONING_METADATA,
    ),
    PromptName.QUERY_EXPANSION: PromptDefinition(
        name=PromptName.QUERY_EXPANSION,
        template=QUERY_EXPANSION_TEMPLATE,
        metadata=QUERY_EXPANSION_METADATA,
    ),
    PromptName.SYNTHESIS: PromptDefinition(
        name=PromptName.SYNTHESIS,
        template=SYNTHESIS_TEMPLATE,
        metadata=SYNTHESIS_METADATA,
    ),
    PromptName.ENTITY_SELECTOR: PromptDefinition(
        name=PromptName.ENTITY_SELECTOR,
        template=ENTITY_SELECTOR_TEMPLATE,
        metadata=ENTITY_SELECTOR_METADATA,
    ),
}


__all__ = [
    "PROMPT_DEFINITIONS",
    "TEXT2CYPHER_EXAMPLES",
    "PromptDefinition",
    "PromptMetadata",
    "PromptName",
]
