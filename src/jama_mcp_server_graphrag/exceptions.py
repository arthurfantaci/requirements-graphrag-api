"""Custom exception hierarchy for GraphRAG MCP Server.

Provides specific exceptions for different failure modes with
proper exception chaining support.
"""

from __future__ import annotations


class GraphRAGError(Exception):
    """Base exception for all GraphRAG errors."""


class ConfigurationError(GraphRAGError):
    """Raised when configuration is invalid or missing."""


class ServiceConnectionError(GraphRAGError):
    """Raised when connection to external service fails."""


class Neo4jConnectionError(ServiceConnectionError):
    """Raised when Neo4j connection fails."""


class InputValidationError(GraphRAGError):
    """Raised when input validation fails."""


class GraphTraversalError(GraphRAGError):
    """Raised when graph traversal operations fail."""


class VectorSearchError(GraphRAGError):
    """Raised when vector similarity search fails."""


class LLMError(GraphRAGError):
    """Raised when LLM operations fail."""


class Text2CypherError(LLMError):
    """Raised when text to Cypher generation fails."""


class EvaluationError(GraphRAGError):
    """Raised when evaluation operations fail."""
