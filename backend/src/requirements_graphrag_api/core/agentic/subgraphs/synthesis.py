"""Answer synthesis subgraph for the agentic system.

This subgraph handles answer generation with self-critique:
1. Draft answer using SYNTHESIS prompt (includes self-critique)
2. Evaluate critique results
3. Revise if confidence is low or completeness is partial
4. Format final output

Flow:
    START -> draft_answer -> (needs_revision? -> revise ->) format_output -> END

State:
    SynthesisState with query, context, draft_answer, critique, final_answer
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from requirements_graphrag_api.core.agentic.state import CriticEvaluation, SynthesisState
from requirements_graphrag_api.evaluation.cost_analysis import get_global_cost_tracker
from requirements_graphrag_api.prompts import PromptName, get_prompt

if TYPE_CHECKING:
    from requirements_graphrag_api.config import AppConfig

logger = logging.getLogger(__name__)

# Constants
MAX_REVISIONS = 1
CONFIDENCE_THRESHOLD = 0.65


def create_synthesis_subgraph(config: AppConfig) -> StateGraph:
    """Create the Synthesis answer generation subgraph.

    Args:
        config: Application configuration.

    Returns:
        Compiled Synthesis subgraph.
    """

    # -------------------------------------------------------------------------
    # Node: draft_answer
    # -------------------------------------------------------------------------
    async def draft_answer(state: SynthesisState) -> dict[str, Any]:
        """Generate initial answer using SYNTHESIS prompt.

        The SYNTHESIS prompt includes self-critique in its output,
        so we get both the answer and evaluation in one call.
        """
        query = state["query"]
        context = state.get("context", "")
        logger.info("Drafting answer for query: %s", query[:50])

        if not context:
            logger.warning("No context provided for synthesis")
            return {
                "draft_answer": "I don't have enough context to answer this question.",
                "critique": CriticEvaluation(
                    answerable=False,
                    confidence=0.0,
                    completeness="insufficient",
                    missing_aspects=["No context was retrieved"],
                    reasoning="No context provided",
                ),
                "revision_count": 0,
            }

        try:
            prompt_template = await get_prompt(PromptName.SYNTHESIS)
            llm = ChatOpenAI(
                model=config.chat_model,
                temperature=0.3,
                api_key=config.openai_api_key,
            )

            chain = prompt_template | llm
            response = await chain.ainvoke(
                {
                    "context": context,
                    "entities": state.get("entities_str", ""),
                    "previous_context": state.get("previous_context", ""),
                    "question": query,
                }
            )
            get_global_cost_tracker().record_from_response(
                config.chat_model, response, operation="synthesis_draft"
            )
            result = response.content

            # Parse JSON response
            try:
                parsed = json.loads(result)
                answer = parsed.get("answer", "")
                critique_data = parsed.get("critique", {})
                citations = parsed.get("citations", [])

                critique = CriticEvaluation(
                    answerable=True,
                    confidence=critique_data.get("confidence", 0.5),
                    completeness=critique_data.get("completeness", "partial"),
                    missing_aspects=critique_data.get("missing_aspects", []),
                    followup_query=critique_data.get("would_benefit_from"),
                    reasoning="Self-critique during synthesis",
                )

                logger.info(
                    "Draft complete - confidence: %.2f, completeness: %s",
                    critique.confidence,
                    critique.completeness,
                )

                return {
                    "draft_answer": answer,
                    "critique": critique,
                    "citations": citations,
                    "revision_count": 0,
                }

            except (json.JSONDecodeError, KeyError) as e:
                # Fallback: treat the whole response as the answer
                logger.warning("Failed to parse synthesis JSON: %s", e)
                return {
                    "draft_answer": result,
                    "critique": CriticEvaluation(
                        answerable=True,
                        confidence=0.5,
                        completeness="partial",
                        reasoning="Could not parse structured output",
                    ),
                    "citations": [],
                    "revision_count": 0,
                }

        except Exception:
            logger.exception("Answer drafting failed")
            return {
                "draft_answer": "I encountered an error generating the answer.",
                "critique": CriticEvaluation(
                    answerable=False,
                    confidence=0.0,
                    completeness="insufficient",
                    reasoning="Exception during synthesis",
                ),
                "revision_count": 0,
            }

    # -------------------------------------------------------------------------
    # Node: revise
    # -------------------------------------------------------------------------
    async def revise(state: SynthesisState) -> dict[str, Any]:
        """Revise the answer based on critique feedback.

        Uses the CRITIC prompt to get more detailed feedback, then
        regenerates with that guidance.
        """
        query = state["query"]
        context = state.get("context", "")
        draft = state.get("draft_answer", "")
        critique = state.get("critique")
        revision_count = state.get("revision_count", 0)

        logger.info("Revising answer (revision %d)", revision_count + 1)

        try:
            # Get detailed critique using CRITIC prompt
            critic_template = await get_prompt(PromptName.CRITIC)
            llm = ChatOpenAI(
                model=config.chat_model,
                temperature=0.2,
                api_key=config.openai_api_key,
            )

            critic_chain = critic_template | llm
            critic_response = await critic_chain.ainvoke(
                {
                    "context": context,
                    "question": query,
                }
            )
            get_global_cost_tracker().record_from_response(
                config.chat_model, critic_response, operation="critic"
            )
            critic_result = critic_response.content

            # Parse critic feedback
            try:
                critic_data = json.loads(critic_result)
            except json.JSONDecodeError:
                critic_data = {}

            # Build revision guidance
            missing = critique.missing_aspects if critique else []
            guidance = ""
            if missing:
                guidance = f"\n\nAddress these gaps: {', '.join(missing)}"
            if critic_data.get("followup_query"):
                guidance += f"\nConsider: {critic_data.get('followup_query')}"

            # Regenerate with guidance
            synth_template = await get_prompt(PromptName.SYNTHESIS)
            synth_chain = synth_template | llm

            # Augment context with previous draft and critique
            augmented_context = f"""{context}

## Previous Draft (needs improvement)
{draft}

## Critic Feedback
{guidance}

Focus on improving completeness and addressing the gaps identified above."""

            synth_response = await synth_chain.ainvoke(
                {
                    "context": augmented_context,
                    "entities": state.get("entities_str", ""),
                    "previous_context": state.get("previous_context", ""),
                    "question": query,
                }
            )
            get_global_cost_tracker().record_from_response(
                config.chat_model, synth_response, operation="synthesis_revision"
            )
            result = synth_response.content

            # Parse revised response
            try:
                parsed = json.loads(result)
                answer = parsed.get("answer", "")
                critique_data = parsed.get("critique", {})
                citations = parsed.get("citations", [])

                new_critique = CriticEvaluation(
                    answerable=True,
                    confidence=critique_data.get("confidence", 0.6),
                    completeness=critique_data.get("completeness", "partial"),
                    missing_aspects=critique_data.get("missing_aspects", []),
                    followup_query=critique_data.get("would_benefit_from"),
                    reasoning=f"Revision {revision_count + 1}",
                )

                logger.info(
                    "Revision complete - confidence: %.2f, completeness: %s",
                    new_critique.confidence,
                    new_critique.completeness,
                )

                return {
                    "draft_answer": answer,
                    "critique": new_critique,
                    "citations": citations,
                    "revision_count": revision_count + 1,
                }

            except (json.JSONDecodeError, KeyError):
                return {
                    "draft_answer": result,
                    "critique": CriticEvaluation(
                        answerable=True,
                        confidence=0.6,
                        completeness="partial",
                        reasoning=f"Revision {revision_count + 1} (unparsed)",
                    ),
                    "citations": [],
                    "revision_count": revision_count + 1,
                }

        except Exception:
            logger.exception("Revision failed")
            # Keep the draft as-is if revision fails
            return {
                "revision_count": revision_count + 1,
            }

    # -------------------------------------------------------------------------
    # Node: format_output
    # -------------------------------------------------------------------------
    async def format_output(state: SynthesisState) -> dict[str, Any]:
        """Format the final answer with citations.

        Takes the draft answer and formats it for output,
        adding citation references and metadata.
        """
        draft = state.get("draft_answer", "")
        citations = state.get("citations", [])
        critique = state.get("critique")

        logger.info("Formatting final output")

        # Build final answer with citation footer
        final = draft
        if citations:
            citation_text = "\n\n**Sources:**\n"
            for i, source in enumerate(citations, 1):
                citation_text += f"- [{i}] {source}\n"
            final += citation_text

        # Add confidence indicator if low
        if critique and critique.confidence < CONFIDENCE_THRESHOLD:
            final += (
                "\n\n*Note: This answer is based on limited context. "
                "Consider asking follow-up questions for more detail.*"
            )

        return {
            "final_answer": final,
        }

    # -------------------------------------------------------------------------
    # Conditional edge: needs_revision
    # -------------------------------------------------------------------------
    def needs_revision(state: SynthesisState) -> Literal["revise", "format_output"]:
        """Determine if the answer needs revision based on critique."""
        critique = state.get("critique")
        revision_count = state.get("revision_count", 0)

        # Don't revise if we've hit the limit
        if revision_count >= MAX_REVISIONS:
            logger.info("Max revisions reached (%d), proceeding to format", revision_count)
            return "format_output"

        # Check critique for revision triggers
        if critique:
            needs_work = (
                critique.confidence < CONFIDENCE_THRESHOLD
                or critique.completeness == "insufficient"
            )
            if needs_work:
                logger.info(
                    "Revision needed - confidence: %.2f, completeness: %s",
                    critique.confidence,
                    critique.completeness,
                )
                return "revise"

        return "format_output"

    # -------------------------------------------------------------------------
    # Build the subgraph
    # -------------------------------------------------------------------------
    builder = StateGraph(SynthesisState)

    # Add nodes
    builder.add_node("draft_answer", draft_answer)
    builder.add_node("revise", revise)
    builder.add_node("format_output", format_output)

    # Add edges
    builder.add_edge(START, "draft_answer")
    builder.add_conditional_edges(
        "draft_answer",
        needs_revision,
        ["revise", "format_output"],
    )
    builder.add_conditional_edges(
        "revise",
        needs_revision,
        ["revise", "format_output"],
    )
    builder.add_edge("format_output", END)

    return builder.compile()


__all__ = ["create_synthesis_subgraph"]
