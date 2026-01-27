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

    INTENT_CLASSIFIER = "graphrag-intent-classifier"
    ROUTER = "graphrag-router"  # Deprecated: use INTENT_CLASSIFIER
    CRITIC = "graphrag-critic"
    STEPBACK = "graphrag-stepback"
    QUERY_UPDATER = "graphrag-query-updater"
    RAG_GENERATION = "graphrag-rag-generation"
    TEXT2CYPHER = "graphrag-text2cypher"


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
# ROUTER PROMPT (Deprecated - use INTENT_CLASSIFIER)
# Routes queries to the most appropriate retrieval tool(s)
# =============================================================================

ROUTER_SYSTEM: Final[
    str
] = """You are a retrieval router for a Requirements Management knowledge graph.

Your task is to analyze the user's question and select the best retrieval tool(s).

Available tools:
{tools}

Selection Guidelines:
1. For simple lookups or general questions -> graphrag_vector_search or graphrag_hybrid_search
2. For questions about how concepts relate -> graphrag_graph_enriched_search
3. For deep dives into specific entities -> graphrag_explore_entity
4. For regulatory/compliance questions -> graphrag_lookup_standard
5. For terminology definitions -> graphrag_lookup_term
6. For aggregations or complex patterns -> graphrag_text2cypher
7. For multi-faceted questions requiring synthesis -> graphrag_chat

You may select multiple tools if the question has multiple parts.

Respond with a JSON object:
{{
    "selected_tools": ["tool_name1", "tool_name2"],
    "reasoning": "Brief explanation of why these tools were selected",
    "tool_params": {{
        "tool_name1": {{"param": "value"}},
        "tool_name2": {{"param": "value"}}
    }}
}}"""

ROUTER_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM),
        ("human", "Question: {question}"),
    ]
)

ROUTER_METADATA = PromptMetadata(
    version="1.0.0",
    description="Routes queries to optimal retrieval tools based on question characteristics",
    input_variables=["tools", "question"],
    output_format="json",
    evaluation_criteria=[
        "correct_tool_selection",
        "appropriate_params",
        "valid_json_output",
    ],
    tags=["routing", "agentic", "classification"],
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
# PROMPT DEFINITIONS REGISTRY
# =============================================================================

PROMPT_DEFINITIONS: Final[dict[PromptName, PromptDefinition]] = {
    PromptName.INTENT_CLASSIFIER: PromptDefinition(
        name=PromptName.INTENT_CLASSIFIER,
        template=INTENT_CLASSIFIER_TEMPLATE,
        metadata=INTENT_CLASSIFIER_METADATA,
    ),
    PromptName.ROUTER: PromptDefinition(
        name=PromptName.ROUTER,
        template=ROUTER_TEMPLATE,
        metadata=ROUTER_METADATA,
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
}


__all__ = [
    "PROMPT_DEFINITIONS",
    "TEXT2CYPHER_EXAMPLES",
    "PromptDefinition",
    "PromptMetadata",
    "PromptName",
]
