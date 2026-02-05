"""Search endpoints for vector, hybrid, and graph-enriched search.

Updated Data Model (2026-01):
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
- Uses VectorRetriever instead of Neo4jVector
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from requirements_graphrag_api.core import (
    graph_enriched_search,
    hybrid_search,
    vector_search,
)
from requirements_graphrag_api.guardrails import (
    InjectionRisk,
    TopicClassification,
    check_prompt_injection,
    check_topic_relevance,
    check_toxicity,
    detect_and_redact_pii,
    log_guardrail_event,
    metrics,
)
from requirements_graphrag_api.guardrails.events import (
    create_injection_event,
    create_pii_event,
    create_topic_event,
    create_toxicity_event,
)
from requirements_graphrag_api.middleware.timeout import TIMEOUTS, with_timeout

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from requirements_graphrag_api.config import AppConfig, GuardrailConfig

logger = logging.getLogger(__name__)

router = APIRouter()


class SearchRequest(BaseModel):
    """Request body for search endpoints."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results",
    )


class HybridSearchRequest(SearchRequest):
    """Request body for hybrid search."""

    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword vs vector (0.0-1.0)",
    )


class EntityInfo(BaseModel):
    """Entity information from graph enrichment."""

    name: str | None = None
    type: str | None = None
    definition: str | None = None
    benefit: str | None = None
    impact: str | None = None


class GlossaryDefinition(BaseModel):
    """Glossary definition from graph enrichment."""

    term: str | None = None
    definition: str | None = None
    url: str | None = None


class SemanticRelationship(BaseModel):
    """Semantic relationship between entities."""

    from_entity: str | None = None
    relationship: str | None = None
    to_entity: str | None = None
    to_type: str | None = None
    to_definition: str | None = None


class ContextWindow(BaseModel):
    """Adjacent chunk context."""

    prev_context: str | None = None
    next_context: str | None = None


class ImageContent(BaseModel):
    """Image content from source articles."""

    url: str
    alt_text: str = ""
    context: str = ""


class WebinarContent(BaseModel):
    """Webinar content from source articles."""

    title: str | None = None
    url: str | None = None
    thumbnail_url: str | None = None


class VideoContent(BaseModel):
    """Video content from source articles."""

    title: str | None = None
    url: str | None = None


class MediaContent(BaseModel):
    """Media content from source articles."""

    images: list[ImageContent] = []
    webinars: list[WebinarContent] = []
    videos: list[VideoContent] = []


class SearchResult(BaseModel):
    """A single search result with graph enrichment."""

    content: str
    score: float
    metadata: dict[str, Any]
    # Level 1: Window context
    context_window: ContextWindow | None = None
    # Level 2: Entities with properties
    entities: list[EntityInfo] = []
    # Level 3: Semantic relationships
    semantic_relationships: list[SemanticRelationship] = []
    # Level 4: Domain context
    industry_standards: list[dict[str, Any]] = []
    media: MediaContent | None = None
    related_articles: list[dict[str, Any]] = []
    glossary_definitions: list[GlossaryDefinition] = []


class SearchResponse(BaseModel):
    """Response from search endpoints."""

    results: list[SearchResult]
    total: int


async def _apply_search_guardrails(
    query: str,
    guardrail_config: GuardrailConfig,
    config: AppConfig | None = None,
) -> str:
    """Apply input guardrails to a search query.

    Checks: prompt injection, PII redaction, toxicity, and topic relevance.

    Args:
        query: Raw user query.
        guardrail_config: Guardrail configuration from app state.
        config: Application config (needed for LLM-based topic guard).

    Returns:
        Sanitized query safe for processing.

    Raises:
        HTTPException: 400 if query is blocked by any guardrail.
    """
    request_id = str(uuid.uuid4())[:8]
    safe_query = query

    # 1. Check for prompt injection
    if guardrail_config.prompt_injection_enabled:
        injection_result = check_prompt_injection(
            query,
            block_threshold=InjectionRisk(guardrail_config.injection_block_threshold),
        )
        if injection_result.should_warn or injection_result.should_block:
            metrics.record_prompt_injection(blocked=injection_result.should_block)
            event = create_injection_event(
                request_id=request_id,
                risk_level=injection_result.risk_level.value,
                patterns=injection_result.detected_patterns,
                blocked=injection_result.should_block,
                input_text=query,
            )
            log_guardrail_event(event)

        if injection_result.should_block:
            metrics.record_request(blocked=True)
            raise HTTPException(
                status_code=400,
                detail="Request blocked by safety filter",
            )

    # 2. Detect and redact PII
    if guardrail_config.pii_detection_enabled:
        pii_result = detect_and_redact_pii(
            safe_query,
            entities=guardrail_config.pii_entities,
            score_threshold=guardrail_config.pii_score_threshold,
            anonymize_type=guardrail_config.pii_anonymize_type,
        )
        if pii_result.check_failed:
            logger.warning("PII detection failed — processing search with unchecked input")
        if pii_result.contains_pii:
            metrics.record_pii(redacted=True)
            entity_types = tuple(e.entity_type for e in pii_result.detected_entities)
            event = create_pii_event(
                request_id=request_id,
                entity_types=entity_types,
                entity_count=pii_result.entity_count,
                redacted=True,
                input_text=query,
            )
            log_guardrail_event(event)
            safe_query = pii_result.anonymized_text
            logger.info(
                "PII detected in search query: %d entities redacted",
                pii_result.entity_count,
            )

    # 3. Toxicity check
    if guardrail_config.toxicity_enabled:
        toxicity_result = await check_toxicity(
            safe_query,
            use_full_check=guardrail_config.toxicity_use_full_check,
        )
        if toxicity_result.should_block:
            metrics.record_toxicity(blocked=True)
            event = create_toxicity_event(
                request_id=request_id,
                categories=tuple(c.value for c in toxicity_result.categories),
                confidence=toxicity_result.confidence,
                blocked=True,
                check_type=toxicity_result.check_type,
                input_text=query,
            )
            log_guardrail_event(event)
            metrics.record_request(blocked=True)
            raise HTTPException(
                status_code=400,
                detail="Request blocked by content safety filter",
            )
        if toxicity_result.should_warn:
            metrics.record_toxicity(blocked=False)
            event = create_toxicity_event(
                request_id=request_id,
                categories=tuple(c.value for c in toxicity_result.categories),
                confidence=toxicity_result.confidence,
                blocked=False,
                check_type=toxicity_result.check_type,
                input_text=query,
            )
            log_guardrail_event(event)

    # 4. Topic guard (keyword + LLM classification for borderline)
    if guardrail_config.topic_guard_enabled:
        from langchain_openai import ChatOpenAI

        from requirements_graphrag_api.guardrails.topic_guard import TopicGuardConfig

        topic_config = TopicGuardConfig(
            enabled=True,
            use_llm_classification=guardrail_config.topic_guard_use_llm,
            allow_borderline=guardrail_config.topic_guard_allow_borderline,
        )
        topic_llm = None
        if guardrail_config.topic_guard_use_llm and config and config.openai_api_key:
            topic_llm = ChatOpenAI(
                model=config.chat_model,
                temperature=0,
                api_key=config.openai_api_key,
                max_tokens=20,
            )
        topic_result = await check_topic_relevance(
            safe_query,
            llm=topic_llm,
            config=topic_config,
        )
        if topic_result.classification == TopicClassification.OUT_OF_SCOPE:
            metrics.record_topic_out_of_scope()
            event = create_topic_event(
                request_id=request_id,
                classification=topic_result.classification.value,
                confidence=topic_result.confidence,
                reasoning=topic_result.reasoning,
                check_type=topic_result.check_type,
                input_text=query,
            )
            log_guardrail_event(event)
            metrics.record_request(blocked=True)
            raise HTTPException(
                status_code=400,
                detail="Query is outside the scope of this service",
            )

    metrics.record_request(blocked=False)
    return safe_query


@router.post("/search/vector", response_model=SearchResponse)
@with_timeout(TIMEOUTS["search"])
async def vector_search_endpoint(
    request: Request,
    body: SearchRequest,
) -> dict[str, Any]:
    """Search using semantic vector similarity.

    Performs vector similarity search on chunk embeddings to find
    semantically similar content.

    Args:
        request: FastAPI request object.
        body: Search request body.

    Returns:
        Search results with content and metadata.
    """
    retriever: VectorRetriever = request.app.state.retriever
    driver: Driver = request.app.state.driver
    guardrail_config: GuardrailConfig = request.app.state.guardrail_config
    config: AppConfig = request.app.state.config

    safe_query = await _apply_search_guardrails(body.query, guardrail_config, config)
    results = await vector_search(retriever, driver, safe_query, limit=body.limit)

    return {
        "results": [SearchResult(**r) for r in results],
        "total": len(results),
    }


@router.post("/search/hybrid", response_model=SearchResponse)
@with_timeout(TIMEOUTS["search"])
async def hybrid_search_endpoint(
    request: Request,
    body: HybridSearchRequest,
) -> dict[str, Any]:
    """Search using combined vector and keyword matching.

    Combines vector similarity with keyword matching for better
    results when queries contain specific terms.

    Args:
        request: FastAPI request object.
        body: Hybrid search request body.

    Returns:
        Search results with combined scores.
    """
    retriever: VectorRetriever = request.app.state.retriever
    driver: Driver = request.app.state.driver
    guardrail_config: GuardrailConfig = request.app.state.guardrail_config
    config: AppConfig = request.app.state.config

    safe_query = await _apply_search_guardrails(body.query, guardrail_config, config)
    results = await hybrid_search(
        retriever,
        driver,
        safe_query,
        limit=body.limit,
        keyword_weight=body.keyword_weight,
    )

    return {
        "results": [SearchResult(**r) for r in results],
        "total": len(results),
    }


@router.post("/search/graph", response_model=SearchResponse)
@with_timeout(TIMEOUTS["search"])
async def graph_search_endpoint(
    request: Request,
    body: SearchRequest,
) -> dict[str, Any]:
    """Search with knowledge graph enrichment.

    Combines hybrid search (vector + keyword) with multi-level graph
    traversal to provide rich context:
    - Level 1: Window expansion (adjacent chunks)
    - Level 2: Entity extraction with properties
    - Level 3: Semantic relationship traversal
    - Level 4: Industry standards, media, cross-references

    Args:
        request: FastAPI request object.
        body: Search request body.

    Returns:
        Search results enriched with comprehensive graph context.
    """
    retriever: VectorRetriever = request.app.state.retriever
    driver: Driver = request.app.state.driver
    guardrail_config: GuardrailConfig = request.app.state.guardrail_config
    config: AppConfig = request.app.state.config

    safe_query = await _apply_search_guardrails(body.query, guardrail_config, config)
    results = await graph_enriched_search(
        retriever,
        driver,
        safe_query,
        limit=body.limit,
    )

    return {
        "results": [SearchResult(**r) for r in results],
        "total": len(results),
    }
