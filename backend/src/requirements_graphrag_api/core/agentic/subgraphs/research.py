"""Entity exploration subgraph for the agentic system.

This subgraph handles deep entity exploration:
1. Identify entities from context that need exploration
2. Explore each entity's relationships
3. Conditional loop for additional entities

Flow:
    START -> identify_entities -> explore_next -> (more_entities? -> explore_next) -> END

State:
    ResearchState with entities, explored_entities, entity_contexts
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from requirements_graphrag_api.core.agentic.state import EntityInfo, ResearchState
from requirements_graphrag_api.evaluation.cost_analysis import get_global_cost_tracker
from requirements_graphrag_api.prompts import PromptName, get_prompt

if TYPE_CHECKING:
    from neo4j import Driver

    from requirements_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)

# Constants
MAX_ENTITIES_TO_EXPLORE = 3
MAX_EXPLORATION_ITERATIONS = 5


def create_research_subgraph(
    config: AppConfig,
    driver: Driver,
) -> StateGraph:
    """Create the Research entity exploration subgraph.

    Args:
        config: Application configuration.
        driver: Neo4j driver instance.

    Returns:
        Compiled Research subgraph.
    """
    # Import here to avoid circular imports
    from requirements_graphrag_api.core.retrieval import explore_entity

    # -------------------------------------------------------------------------
    # Node: identify_entities
    # -------------------------------------------------------------------------
    async def identify_entities(state: ResearchState) -> dict[str, Any]:
        """Identify entities from context that warrant deeper exploration.

        Uses the ENTITY_SELECTOR prompt to analyze context and select
        entities for exploration.
        """
        query = state["query"]
        context = state.get("context", "")
        logger.info("Identifying entities for query: %s", query[:50])

        if not context:
            logger.warning("No context provided for entity identification")
            return {
                "identified_entities": [],
                "exploration_complete": True,
            }

        try:
            prompt_template = await get_prompt(PromptName.ENTITY_SELECTOR)
            llm = ChatOpenAI(
                model=config.chat_model,
                temperature=0.2,
                api_key=config.openai_api_key,
            )

            chain = prompt_template | llm
            response = await chain.ainvoke(
                {
                    "context": context,
                    "question": query,
                }
            )
            get_global_cost_tracker().record_from_response(
                config.chat_model, response, operation="entity_identification"
            )
            result = response.content

            # Parse JSON response
            try:
                parsed = json.loads(result)
                entities = [e["name"] for e in parsed.get("entities", [])]
                priority = parsed.get("exploration_priority", "medium")
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to parse entity selection, skipping exploration")
                entities = []
                priority = "low"

            # Limit entities
            entities = entities[:MAX_ENTITIES_TO_EXPLORE]

            logger.info(
                "Identified %d entities for exploration (priority: %s)",
                len(entities),
                priority,
            )

            # If no entities or low priority, mark as complete
            exploration_complete = len(entities) == 0 or priority == "low"

            return {
                "identified_entities": entities,
                "explored_entities": [],
                "entity_contexts": [],
                "exploration_complete": exploration_complete,
            }

        except Exception:
            logger.exception("Entity identification failed")
            return {
                "identified_entities": [],
                "exploration_complete": True,
            }

    # -------------------------------------------------------------------------
    # Node: explore_next
    # -------------------------------------------------------------------------
    async def explore_next(state: ResearchState) -> dict[str, Any]:
        """Explore the next unvisited entity.

        Calls explore_entity for the next entity in the queue and
        adds results to entity_contexts.
        """
        identified = state.get("identified_entities", [])
        explored = state.get("explored_entities", [])

        # Find next entity to explore
        remaining = [e for e in identified if e not in explored]

        if not remaining:
            logger.info("No more entities to explore")
            return {"exploration_complete": True}

        entity_name = remaining[0]
        logger.info("Exploring entity: %s", entity_name)

        try:
            result = await explore_entity(
                driver=driver,
                entity_name=entity_name,
                include_related=True,
                related_limit=5,
            )

            if result:
                entity_info = EntityInfo(
                    name=entity_name,
                    entity_type=result.get("type", "Unknown"),
                    description=result.get("display_name", entity_name),
                    related_entities=[
                        r.get("name", "") for r in result.get("related_entities", [])
                    ],
                    mentioned_in=[
                        m if isinstance(m, str) else m.get("title", str(m))
                        for m in result.get("mentioned_in", [])[:5]
                    ],
                )
                logger.info(
                    "Explored %s: %d related, %d articles",
                    entity_name,
                    len(entity_info.related_entities),
                    len(entity_info.mentioned_in),
                )
            else:
                # Entity not found, create minimal info
                entity_info = EntityInfo(
                    name=entity_name,
                    entity_type="Unknown",
                    description=f"Entity '{entity_name}' not found in knowledge graph",
                )
                logger.warning("Entity not found: %s", entity_name)

            # Check if we should continue
            new_explored = [*explored, entity_name]
            remaining_after = [e for e in identified if e not in new_explored]
            exploration_complete = (
                len(remaining_after) == 0 or len(new_explored) >= MAX_EXPLORATION_ITERATIONS
            )

            return {
                "explored_entities": [entity_name],  # Reducer will append
                "entity_contexts": [entity_info],  # Reducer will append
                "exploration_complete": exploration_complete,
            }

        except Exception:
            logger.exception("Entity exploration failed for %s", entity_name)
            # Mark as explored to avoid infinite loop
            return {
                "explored_entities": [entity_name],
                "exploration_complete": True,
            }

    # -------------------------------------------------------------------------
    # Conditional edge: should_continue_exploring
    # -------------------------------------------------------------------------
    def should_continue_exploring(state: ResearchState) -> Literal["explore_next", END]:
        """Determine if we should continue exploring entities."""
        if state.get("exploration_complete", False):
            return END

        identified = state.get("identified_entities", [])
        explored = state.get("explored_entities", [])
        remaining = [e for e in identified if e not in explored]

        if remaining and len(explored) < MAX_EXPLORATION_ITERATIONS:
            return "explore_next"
        return END

    # -------------------------------------------------------------------------
    # Build the subgraph
    # -------------------------------------------------------------------------
    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("identify_entities", identify_entities)
    builder.add_node("explore_next", explore_next)

    # Add edges
    builder.add_edge(START, "identify_entities")
    builder.add_conditional_edges(
        "identify_entities",
        should_continue_exploring,
        ["explore_next", END],
    )
    builder.add_conditional_edges(
        "explore_next",
        should_continue_exploring,
        ["explore_next", END],
    )

    return builder.compile()


__all__ = ["create_research_subgraph"]
