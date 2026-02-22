"""Voyage AI embedding wrapper implementing neo4j-graphrag Embedder interface.

This module provides a VoyageAIEmbeddings class that wraps the Voyage AI SDK
for use with neo4j-graphrag's VectorRetriever. It replaces the previous
OpenAIEmbeddings for query-time vector search after the upstream pipeline
(graphrag-api-db) migrated to Voyage AI voyage-4 (1024d).

Key design decisions:
- Default input_type="query" — this repo is always query-side
- Explicit api_key parameter — matches how OpenAIEmbeddings receives keys
- Implements only embed_query() (required by ABC) + async convenience method
"""

from __future__ import annotations

import structlog
import voyageai
from neo4j_graphrag.embeddings.base import Embedder

logger = structlog.get_logger()


class VoyageAIEmbeddings(Embedder):
    """Voyage AI embeddings for neo4j-graphrag VectorRetriever.

    Wraps the Voyage AI SDK to implement the neo4j-graphrag Embedder interface.
    Uses asymmetric embeddings: input_type="query" for search queries,
    input_type="document" for indexing (handled by graphrag-api-db pipeline).

    Args:
        model: Voyage AI model name (e.g., "voyage-4").
        input_type: Embedding input type ("query" or "document").
        dimensions: Output embedding dimensions (voyage-4 supports 256/512/1024/2048).
        api_key: Voyage AI API key.
    """

    def __init__(
        self,
        model: str = "voyage-4",
        input_type: str = "query",
        dimensions: int = 1024,
        api_key: str = "",
    ) -> None:
        """Initialize VoyageAIEmbeddings.

        Args:
            model: Voyage AI model name.
            input_type: Embedding input type ("query" or "document").
            dimensions: Output embedding dimensions.
            api_key: Voyage AI API key.
        """
        super().__init__()
        self.model = model
        self.input_type = input_type
        self.dimensions = dimensions
        self._client = voyageai.Client(api_key=api_key)
        self._async_client = voyageai.AsyncClient(api_key=api_key)
        logger.info(
            "Initialized VoyageAIEmbeddings: model=%s, input_type=%s, dimensions=%d",
            model,
            input_type,
            dimensions,
        )

    def embed_query(self, text: str) -> list[float]:
        """Embed query text using Voyage AI.

        Args:
            text: Text to convert to vector embedding.

        Returns:
            A vector embedding as list of floats.
        """
        result = self._client.embed(
            [text],
            model=self.model,
            input_type=self.input_type,
            output_dimension=self.dimensions,
        )
        return result.embeddings[0]

    async def async_embed_query(self, text: str) -> list[float]:
        """Embed query text asynchronously using Voyage AI.

        Convenience method for async callers. Not required by the Embedder ABC.

        Args:
            text: Text to convert to vector embedding.

        Returns:
            A vector embedding as list of floats.
        """
        result = await self._async_client.embed(
            [text],
            model=self.model,
            input_type=self.input_type,
            output_dimension=self.dimensions,
        )
        return result.embeddings[0]
