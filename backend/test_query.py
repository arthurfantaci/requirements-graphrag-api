#!/usr/bin/env python3
"""Test script for querying the Jama Guide knowledge graph.

This script demonstrates three query approaches:
1. Vector similarity search on chunks
2. Graph traversal from chunks to entities
3. Direct entity search by name

Usage:
    uv run python test_query.py
    uv run python test_query.py "What is impact analysis?"
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.types import RetrieverResult

load_dotenv()

console = Console()


def get_driver() -> Driver:
    """Create Neo4j driver from environment variables."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(username, password))


def vector_search(driver: Driver, query: str, top_k: int = 5) -> RetrieverResult:
    """Perform vector similarity search on chunk embeddings."""
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    retriever = VectorRetriever(
        driver=driver,
        index_name="chunk_embeddings",
        embedder=embedder,
        return_properties=["text"],
    )
    return retriever.search(query_text=query, top_k=top_k)


def get_entities_from_chunks(driver: Driver, chunk_ids: list[str]) -> list[dict]:
    """Find entities mentioned in the retrieved chunks."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Chunk)<-[:MENTIONED_IN]-(entity)
            WHERE elementId(c) IN $chunk_ids
            WITH entity, labels(entity)[0] AS label, count(*) AS mentions
            RETURN label, entity.name AS name, entity.display_name AS display_name,
                   entity.definition AS definition, mentions
            ORDER BY mentions DESC, label, name
            LIMIT 20
            """,
            chunk_ids=chunk_ids,
        )
        return [dict(record) for record in result]


def search_entities_by_name(driver: Driver, search_term: str) -> list[dict]:
    """Direct search for entities containing the search term."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE (n:Concept OR n:Challenge OR n:Bestpractice OR n:Standard
                   OR n:Methodology OR n:Artifact OR n:Tool)
                  AND (n.name CONTAINS $term OR n.display_name CONTAINS $term)
            WITH n, labels(n)[0] AS label
            OPTIONAL MATCH (n)-[r]-(related)
            WITH n, label, count(DISTINCT related) AS connections
            RETURN label, n.name AS name, n.display_name AS display_name,
                   n.definition AS definition, connections
            ORDER BY connections DESC
            LIMIT 10
            """,
            term=search_term.lower(),
        )
        return [dict(record) for record in result]


def get_related_entities(driver: Driver, entity_name: str) -> list[dict]:
    """Get entities related to a specific entity."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n {name: $name})-[r]-(related)
            WITH type(r) AS rel_type,
                 labels(related)[0] AS related_label,
                 related.name AS related_name,
                 related.display_name AS related_display,
                 startNode(r) = n AS outgoing
            RETURN rel_type,
                   CASE WHEN outgoing THEN '->' ELSE '<-' END AS direction,
                   related_label, related_name, related_display
            ORDER BY rel_type, related_label
            """,
            name=entity_name.lower(),
        )
        return [dict(record) for record in result]


def display_vector_results(results: RetrieverResult) -> list[str | None]:
    """Display vector search results and return chunk IDs."""
    console.print("\n[bold green]1. Vector Similarity Search[/] (semantic match)")
    console.print("-" * 60)

    chunk_ids = []
    for i, item in enumerate(results.items, 1):
        chunk_ids.append(item.metadata.get("element_id") if item.metadata else None)
        text = (
            item.content.get("text", "")[:300]
            if isinstance(item.content, dict)
            else str(item.content)[:300]
        )
        score = item.metadata.get("score", 0.0) if item.metadata else 0.0
        console.print(f"\n[yellow]Result {i}[/] (score: {score:.3f})")
        console.print(f"{text}...")
    return chunk_ids


def display_entity_table(entities: list[dict], title: str, count_col: str) -> None:
    """Display entities in a formatted table."""
    console.print(f"\n\n[bold green]{title}[/]")
    console.print("-" * 60)

    if not entities:
        console.print("[dim]No entities found[/]")
        return

    table = Table(show_header=True)
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column(count_col, justify="right")
    table.add_column("Definition", max_width=50)

    for record in entities:
        definition = record.get("definition") or ""
        table.add_row(
            record["label"],
            record.get("display_name") or record["name"],
            str(record.get("mentions") or record.get("connections", 0)),
            f"{definition[:50]}..." if definition else "-",
        )
    console.print(table)


def display_relationships(relationships: list[dict], entity_name: str) -> None:
    """Display relationships for an entity."""
    console.print(f"\n[bold green]4. Relationships for '{entity_name}'[/]")
    console.print("-" * 60)

    if not relationships:
        console.print("[dim]No relationships found[/]")
        return

    table = Table(show_header=True)
    table.add_column("Direction", justify="center")
    table.add_column("Relationship", style="magenta")
    table.add_column("Type", style="cyan")
    table.add_column("Related Entity", style="green")

    for rel in relationships:
        table.add_row(
            rel["direction"],
            rel["rel_type"],
            rel["related_label"],
            rel.get("related_display") or rel["related_name"],
        )
    console.print(table)


def main() -> None:
    """Run the knowledge graph query demonstration."""
    query = sys.argv[1] if len(sys.argv) > 1 else "What can you tell me about Requirements Tracing?"
    console.print(Panel(f"[bold cyan]Query:[/] {query}", title="Jama Guide Knowledge Graph Test"))

    driver = get_driver()
    try:
        # 1. Vector similarity search
        results = vector_search(driver, query)
        chunk_ids = display_vector_results(results)

        # 2. Entities from retrieved chunks
        valid_chunk_ids = [cid for cid in chunk_ids if cid]
        entities = get_entities_from_chunks(driver, valid_chunk_ids) if valid_chunk_ids else []
        display_entity_table(entities, "2. Entities Mentioned in Retrieved Chunks", "Mentions")

        # 3. Direct entity search
        direct_results = search_entities_by_name(driver, "trac")
        display_entity_table(
            direct_results,
            "3. Direct Entity Search (name contains 'trac')",
            "Connections",
        )

        # 4. Relationships for top result
        if direct_results:
            top_entity = direct_results[0]["name"]
            relationships = get_related_entities(driver, top_entity)
            display_relationships(relationships, top_entity)

        console.print("\n[bold green]✓ Query test complete![/]")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
