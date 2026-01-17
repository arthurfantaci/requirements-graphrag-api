"""Chat endpoint for RAG-powered Q&A.

Updated Data Model (2026-01):
- Uses neo4j Driver directly instead of LangChain Neo4jGraph
- Uses VectorRetriever instead of Neo4jVector
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from jama_mcp_server_graphrag.core import chat as core_chat

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.retrievers import VectorRetriever

    from jama_mcp_server_graphrag.config import AppConfig

router = APIRouter()


class ChatOptions(BaseModel):
    """Options for chat request."""

    retrieval_strategy: str = Field(
        default="hybrid",
        description="Retrieval strategy: 'vector', 'hybrid', or 'graph'",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source citations",
    )
    max_sources: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of sources to cite",
    )


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message to respond to",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation ID for history tracking",
    )
    options: ChatOptions = Field(
        default_factory=ChatOptions,
        description="Chat options",
    )


class SourceInfo(BaseModel):
    """Information about a source citation."""

    title: str
    url: str | None
    chunk_id: str | None
    relevance_score: float


class EntityInfo(BaseModel):
    """Information about a related entity."""

    name: str
    type: str | None = None


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    answer: str
    sources: list[SourceInfo]
    entities: list[EntityInfo]
    conversation_id: str | None


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    body: ChatRequest,
) -> dict[str, Any]:
    """Chat with the requirements management knowledge base.

    This endpoint provides RAG-powered Q&A with source citations.
    It retrieves relevant content from the knowledge graph and
    generates answers grounded in the source material.

    Args:
        request: FastAPI request object.
        body: Chat request body.

    Returns:
        Answer with sources and related entities.
    """
    config: AppConfig = request.app.state.config
    retriever: VectorRetriever = request.app.state.retriever
    driver: Driver = request.app.state.driver

    # Call core chat function
    result = await core_chat(
        config,
        retriever,
        driver,
        body.message,
        max_sources=body.options.max_sources,
    )

    # Transform sources to response format
    sources = [
        SourceInfo(
            title=s.get("title", "Unknown"),
            url=s.get("url"),
            chunk_id=s.get("chunk_id"),
            relevance_score=s.get("relevance_score", 0.0),
        )
        for s in result.get("sources", [])
    ]

    # Transform entities to response format
    entities = [
        EntityInfo(name=e) if isinstance(e, str) else EntityInfo(**e)
        for e in result.get("entities", [])
    ]

    return {
        "answer": result["answer"],
        "sources": sources,
        "entities": entities,
        "conversation_id": body.conversation_id,
    }
